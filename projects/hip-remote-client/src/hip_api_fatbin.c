/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file hip_api_fatbin.c
 * @brief Fat binary registration and call configuration for remote HIP
 *
 * Implements the HIP fat binary registration API used by compiled HIP programs
 * (including PyTorch's <<<>>> kernel launch syntax). When a HIP program is
 * compiled, the compiler generates calls to __hipRegisterFatBinary,
 * __hipRegisterFunction, etc. at static initialization time. At kernel launch
 * time, __hipPushCallConfiguration and __hipPopCallConfiguration manage the
 * launch configuration stack.
 *
 * This file bridges the gap between the compiler-generated registration calls
 * and the remote execution model: fat binaries are sent to the worker via
 * hipModuleLoadData, and function lookups are resolved via hipModuleGetFunction.
 */

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"
#include "hip_remote/hip_remote_platform.h"

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Type Definitions
 *
 * uint3 is used in __hipRegisterFunction's signature but is not defined
 * in our minimal HIP type definitions. dim3 is already provided by
 * hip_remote_client.h.
 * ============================================================================ */

typedef struct {
    unsigned int x, y, z;
} uint3;

/* ============================================================================
 * Fat Binary Module Registry
 *
 * When __hipRegisterFatBinary is called, we send the fat binary data to the
 * worker via hipModuleLoadData and store the resulting module handle. The
 * return value is a void** "cookie" pointing to the stored module handle.
 * ============================================================================ */

#define INITIAL_FATBIN_MODULES 1024
#define MAX_FATBIN_DATA_SIZE (16 * 1024 * 1024)

typedef struct {
    hipModule_t module;         /* Remote module handle (NULL until loaded) */
    const void* fatbin_data;    /* Pointer to original fat binary data */
    size_t fatbin_size;         /* Size of fat binary data */
    int in_use;
    int loaded;                 /* Whether module has been sent to worker */
} FatBinModule;

static FatBinModule* g_fatbin_modules = NULL;
static uint32_t g_fatbin_count = 0;
static uint32_t g_fatbin_capacity = 0;
static hip_mutex_t g_fatbin_lock = HIP_MUTEX_INIT;

static int fatbin_ensure_capacity(void) {
    if (g_fatbin_count < g_fatbin_capacity) return 0;
    uint32_t new_cap = g_fatbin_capacity == 0 ? INITIAL_FATBIN_MODULES : g_fatbin_capacity * 2;
    FatBinModule* new_arr = (FatBinModule*)realloc(g_fatbin_modules, new_cap * sizeof(FatBinModule));
    if (!new_arr) return -1;
    memset(new_arr + g_fatbin_capacity, 0, (new_cap - g_fatbin_capacity) * sizeof(FatBinModule));
    g_fatbin_modules = new_arr;
    g_fatbin_capacity = new_cap;
    return 0;
}

/**
 * Lazily load a fat binary module to the worker. Called on first use
 * (e.g., when __hipRegisterFunction or hipLaunchKernel needs the module).
 * This defers the network call until the worker connection is established.
 */
static hipError_t fatbin_ensure_loaded(FatBinModule* mod) {
    if (mod->loaded) return hipSuccess;
    if (!mod->fatbin_data) return hipErrorInvalidValue;

    /* Send the fat binary data with the KNOWN size from registration,
     * bypassing hipModuleLoadData's size guessing which can be wrong
     * for compressed offload bundles (CCOB). */
    HipRemoteModuleLoadRequest req;
    req.data_size = (uint64_t)mod->fatbin_size;

    HipRemoteModuleLoadResponse resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request_with_data(
        HIP_OP_MODULE_LOAD_DATA,
        &req, sizeof(req),
        mod->fatbin_data, mod->fatbin_size,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        mod->module = (hipModule_t)(uintptr_t)resp.module;
        mod->loaded = 1;
        hip_remote_log_debug("fatbin_ensure_loaded: module %p loaded (%zu bytes)", (void*)mod->module, mod->fatbin_size);
    } else {
        hip_remote_log_debug("fatbin_ensure_loaded: load failed err=%d size=%zu", err, mod->fatbin_size);
    }
    return err;
}

/* ============================================================================
 * Host Function -> Remote Function Registry
 *
 * When __hipRegisterFunction is called, we call hipModuleGetFunction on the
 * worker and store the mapping from host function pointer to remote
 * hipFunction_t handle. hipLaunchKernel uses this registry to translate
 * host function pointers into remote handles at launch time.
 * ============================================================================ */

#define INITIAL_REGISTERED_FUNCTIONS 4096

typedef struct {
    const void* host_function;
    hipFunction_t remote_function;  /* NULL until lazily resolved */
    uint32_t module_index;          /* Index into g_fatbin_modules (survives realloc) */
    char device_name[1024];
    int resolved;                   /* Whether remote_function is valid */
} RegisteredFunction;

static RegisteredFunction* g_registered_functions = NULL;
static uint32_t g_registered_function_count = 0;
static uint32_t g_registered_function_capacity = 0;
static hip_mutex_t g_registered_function_lock = HIP_MUTEX_INIT;

static int functions_ensure_capacity(void) {
    if (g_registered_function_count < g_registered_function_capacity) return 0;
    uint32_t new_cap = g_registered_function_capacity == 0 ? INITIAL_REGISTERED_FUNCTIONS : g_registered_function_capacity * 2;
    RegisteredFunction* new_arr = (RegisteredFunction*)realloc(g_registered_functions, new_cap * sizeof(RegisteredFunction));
    if (!new_arr) return -1;
    memset(new_arr + g_registered_function_capacity, 0, (new_cap - g_registered_function_capacity) * sizeof(RegisteredFunction));
    g_registered_functions = new_arr;
    g_registered_function_capacity = new_cap;
    return 0;
}

/* ============================================================================
 * Call Configuration Stack
 *
 * The <<<gridDim, blockDim, sharedMem, stream>>> syntax generates a call to
 * __hipPushCallConfiguration before the kernel stub, and the stub calls
 * __hipPopCallConfiguration to retrieve the configuration before calling
 * hipLaunchKernel.
 *
 * This is a simple static stack. For multi-threaded use, the mutex provides
 * safety, though a proper implementation would use thread-local storage.
 * ============================================================================ */

#define MAX_CALL_CONFIG_DEPTH 32

typedef struct {
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
    hipStream_t stream;
} CallConfig;

/* The call config stack MUST be thread-local. The <<<>>> syntax generates
 * push/pop pairs on the calling thread, and PyTorch uses multiple threads
 * for kernel launches. A global stack causes cross-thread corruption. */
#ifdef _WIN32
static __declspec(thread) CallConfig g_call_config_stack[MAX_CALL_CONFIG_DEPTH];
static __declspec(thread) int g_call_config_top = 0;
#else
static _Thread_local CallConfig g_call_config_stack[MAX_CALL_CONFIG_DEPTH];
static _Thread_local int g_call_config_top = 0;
#endif
/* No mutex needed — the stack is thread-local */

/* ============================================================================
 * Fat Binary Registration
 * ============================================================================ */

/* HIP fat binary wrapper — matches the struct generated by hipcc/amdclang. */
#define HIP_FATBIN_MAGIC 0x48495046  /* "HIPF" */

typedef struct {
    unsigned int magic;
    unsigned int version;
    void* binary;
    void* dummy1;
} HipFatBinaryWrapper;

void** __hipRegisterFatBinary(const void* data) {
    if (!data) {
        return NULL;
    }

    /* The data pointer is a __CudaFatBinaryWrapper, not the raw code object.
     * Dereference wrapper->binary to get the actual code object. */
    const HipFatBinaryWrapper* wrapper = (const HipFatBinaryWrapper*)data;
    const void* binary = NULL;

    if (wrapper->magic == HIP_FATBIN_MAGIC && wrapper->version == 1) {
        binary = wrapper->binary;
        hip_remote_log_debug("__hipRegisterFatBinary: HIPF wrapper, binary=%p", binary);
    } else {
        /* Might be a raw code object (e.g., from hipModuleLoadData) */
        binary = data;
        hip_remote_log_debug("__hipRegisterFatBinary: raw binary (magic=0x%x)", wrapper->magic);
    }

    if (!binary) {
        return NULL;
    }

    /* Determine size of the code object for later transmission. */
    const unsigned char* bytes = (const unsigned char*)binary;
    size_t data_size = 0;

    /* Check for ELF magic */
    if (bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F') {
        uint64_t e_shoff = *(uint64_t*)(bytes + 40);
        uint16_t e_shentsize = *(uint16_t*)(bytes + 58);
        uint16_t e_shnum = *(uint16_t*)(bytes + 60);
        data_size = (size_t)(e_shoff + e_shnum * e_shentsize);
        hip_remote_log_debug("__hipRegisterFatBinary: ELF size=%zu (shoff=%lu, shentsize=%u, shnum=%u)",
                             data_size, (unsigned long)e_shoff, e_shentsize, e_shnum);
    }
    /* Check for Clang offload bundle */
    else if (memcmp(bytes, "__CLANG_OFFLOAD_BUNDLE__", 24) == 0) {
        uint64_t num_bundles = *(uint64_t*)(bytes + 24);
        const unsigned char* ptr = bytes + 32;
        uint64_t max_end = 0;
        for (uint64_t i = 0; i < num_bundles && i < 16; i++) {
            uint64_t offset = *(uint64_t*)ptr;
            uint64_t bsize = *(uint64_t*)(ptr + 8);
            uint64_t triple_len = *(uint64_t*)(ptr + 16);
            uint64_t end_pos = offset + bsize;
            if (end_pos > max_end) max_end = end_pos;
            ptr += 24 + triple_len;
        }
        data_size = (size_t)max_end;
        hip_remote_log_debug("__hipRegisterFatBinary: bundle size=%zu (bundles=%lu)", data_size, (unsigned long)num_bundles);
    }
    /* Check for compressed offload bundle (CCOB) */
    else if (bytes[0] == 'C' && bytes[1] == 'C' && bytes[2] == 'O' && bytes[3] == 'B') {
        uint16_t version = *(uint16_t*)(bytes + 4);
        if (version >= 3) {
            uint64_t file_size = *(uint64_t*)(bytes + 8);
            data_size = (size_t)file_size;
        } else if (version == 2) {
            uint32_t file_size = *(uint32_t*)(bytes + 8);
            data_size = (size_t)file_size;
        }
        hip_remote_log_debug("__hipRegisterFatBinary: CCOB v%u, size=%zu", version, data_size);
    } else {
        hip_remote_log_debug("__hipRegisterFatBinary: unknown format magic=0x%02x%02x%02x%02x",
                             bytes[0], bytes[1], bytes[2], bytes[3]);
    }

    if (data_size == 0 || data_size > MAX_FATBIN_DATA_SIZE) {
        data_size = MAX_FATBIN_DATA_SIZE;
    }
    /* No minimum size check — trust the parser for known formats. */

    hip_mutex_lock(&g_fatbin_lock);

    if (fatbin_ensure_capacity() != 0) {
        hip_mutex_unlock(&g_fatbin_lock);
        return NULL;
    }

    uint32_t idx = g_fatbin_count++;
    g_fatbin_modules[idx].module = NULL;
    g_fatbin_modules[idx].fatbin_data = binary;
    g_fatbin_modules[idx].fatbin_size = data_size;
    g_fatbin_modules[idx].in_use = 1;
    g_fatbin_modules[idx].loaded = 0;

    hip_mutex_unlock(&g_fatbin_lock);

    hip_remote_log_debug("__hipRegisterFatBinary: stored at slot %u (%zu bytes)", idx, data_size);

    /* Return the 1-based index as the cookie (cast to void**).
     * Using the index avoids invalidation when the array is reallocated. */
    return (void**)(uintptr_t)(idx + 1);
}

/* ============================================================================
 * Function Registration
 * ============================================================================ */

void __hipRegisterFunction(void** modules,
                           const void* hostFunction,
                           char* deviceFunction,
                           const char* deviceName,
                           unsigned int threadLimit,
                           uint3* tid,
                           uint3* bid,
                           dim3* blockDim,
                           dim3* gridDim,
                           int* wSize) {
    (void)deviceFunction;
    (void)threadLimit;
    (void)tid;
    (void)bid;
    (void)blockDim;
    (void)gridDim;
    (void)wSize;

    if (!modules || !hostFunction || !deviceName) {
        return;
    }

    /* Decode the cookie: it's a 1-based index into g_fatbin_modules */
    uint32_t mod_idx = (uint32_t)(uintptr_t)modules;
    if (mod_idx == 0 || mod_idx > g_fatbin_count) {
        return;
    }
    FatBinModule* mod = &g_fatbin_modules[mod_idx - 1];
    if (!mod->in_use) {
        return;
    }

    hipFunction_t func = NULL; /* Will be resolved lazily */

    hip_mutex_lock(&g_registered_function_lock);

    if (functions_ensure_capacity() != 0) {
        hip_mutex_unlock(&g_registered_function_lock);
        return;
    }

    uint32_t idx = g_registered_function_count++;
    g_registered_functions[idx].host_function = hostFunction;
    g_registered_functions[idx].remote_function = NULL;
    g_registered_functions[idx].module_index = mod_idx - 1;  /* 0-based index, survives realloc */
    g_registered_functions[idx].resolved = 0;
    strncpy(g_registered_functions[idx].device_name, deviceName,
            sizeof(g_registered_functions[idx].device_name) - 1);
    g_registered_functions[idx].device_name[
        sizeof(g_registered_functions[idx].device_name) - 1] = '\0';

    hip_mutex_unlock(&g_registered_function_lock);
}

/* ============================================================================
 * Variable Registration (stub)
 * ============================================================================ */

void __hipRegisterVar(void** modules,
                      void* var,
                      char* hostVar,
                      const char* deviceName,
                      int ext,
                      size_t size,
                      int constant,
                      int global) {
    (void)modules;
    (void)var;
    (void)hostVar;
    (void)ext;
    (void)size;
    (void)constant;
    (void)global;

    hip_remote_log_debug("__hipRegisterVar: '%s' (stub, not implemented)",
                         deviceName ? deviceName : "(null)");
}

/* ============================================================================
 * Fat Binary Unregistration
 * ============================================================================ */

void __hipUnregisterFatBinary(void** modules) {
    if (!modules) {
        return;
    }

    uint32_t mod_idx = (uint32_t)(uintptr_t)modules;
    if (mod_idx == 0 || mod_idx > g_fatbin_count) {
        return;
    }

    hip_mutex_lock(&g_fatbin_lock);

    FatBinModule* mod = &g_fatbin_modules[mod_idx - 1];
    if (mod->in_use) {
        if (mod->loaded && mod->module) {
            hip_remote_log_debug("__hipUnregisterFatBinary: unloading module %p at slot %u",
                                 (void*)mod->module, mod_idx - 1);
            hipModuleUnload(mod->module);
        }
        mod->module = NULL;
        mod->in_use = 0;
        mod->loaded = 0;
    }

    hip_mutex_unlock(&g_fatbin_lock);
}

/* ============================================================================
 * Call Configuration Stack
 * ============================================================================ */

hipError_t __hipPushCallConfiguration(dim3 gridDim,
                                      dim3 blockDim,
                                      size_t sharedMem,
                                      hipStream_t stream) {
    if (g_call_config_top >= MAX_CALL_CONFIG_DEPTH) {
        hip_remote_log_error(
            "__hipPushCallConfiguration: stack overflow (%d deep)",
            MAX_CALL_CONFIG_DEPTH);
        return hipErrorInvalidConfiguration;
    }

    g_call_config_stack[g_call_config_top].grid_dim = gridDim;
    g_call_config_stack[g_call_config_top].block_dim = blockDim;
    g_call_config_stack[g_call_config_top].shared_mem = sharedMem;
    g_call_config_stack[g_call_config_top].stream = stream;
    g_call_config_top++;

    hip_remote_log_debug(
        "__hipPushCallConfiguration: grid=(%u,%u,%u) block=(%u,%u,%u) "
        "shared=%zu",
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem);

    return hipSuccess;
}

hipError_t __hipPopCallConfiguration(dim3* gridDim,
                                     dim3* blockDim,
                                     size_t* sharedMem,
                                     hipStream_t* stream) {
    if (!gridDim || !blockDim || !sharedMem || !stream) {
        return hipErrorInvalidValue;
    }

    if (g_call_config_top <= 0) {
        hip_remote_log_error("__hipPopCallConfiguration: stack underflow");
        return hipErrorInvalidConfiguration;
    }

    g_call_config_top--;
    *gridDim = g_call_config_stack[g_call_config_top].grid_dim;
    *blockDim = g_call_config_stack[g_call_config_top].block_dim;
    *sharedMem = g_call_config_stack[g_call_config_top].shared_mem;
    *stream = g_call_config_stack[g_call_config_top].stream;

    hip_remote_log_debug(
        "__hipPopCallConfiguration: grid=(%u,%u,%u) block=(%u,%u,%u) "
        "shared=%zu",
        gridDim->x, gridDim->y, gridDim->z,
        blockDim->x, blockDim->y, blockDim->z,
        *sharedMem);

    return hipSuccess;
}

/* ============================================================================
 * Function Lookup for hipLaunchKernel
 *
 * Called by hipLaunchKernel to translate a host function pointer (which is
 * meaningless on the remote worker) into the corresponding remote
 * hipFunction_t handle that was registered via __hipRegisterFunction.
 * ============================================================================ */

/* Return the number of registered functions (for diagnostics) */
uint32_t hip_fatbin_get_registered_count(void) {
    return g_registered_function_count;
}

/* Return the number of registered modules (for diagnostics) */
uint32_t hip_fatbin_get_module_count(void) {
    return g_fatbin_count;
}

hipFunction_t hip_fatbin_lookup_function(const void* hostFunction) {
    hip_mutex_lock(&g_registered_function_lock);

    for (uint32_t i = 0; i < g_registered_function_count; i++) {
        if (g_registered_functions[i].host_function == hostFunction) {
            RegisteredFunction* rf = &g_registered_functions[i];

            /* Lazily resolve the remote function handle on first use */
            if (!rf->resolved) {
                /* Get the module by index (safe across reallocs) */
                FatBinModule* mod = &g_fatbin_modules[rf->module_index];

                /* Ensure the fat binary is loaded to the worker */
                hipError_t err = fatbin_ensure_loaded(mod);
                if (err != hipSuccess) {
                    hip_remote_log_debug("hip_fatbin_lookup_function: module load failed for '%s'", rf->device_name);
                    hip_mutex_unlock(&g_registered_function_lock);
                    return NULL;
                }

                /* Look up the function on the worker */
                hipFunction_t func = NULL;
                err = hipModuleGetFunction(&func, mod->module, rf->device_name);
                if (err == hipSuccess && func) {
                    rf->remote_function = func;
                    rf->resolved = 1;
                    hip_remote_log_debug("hip_fatbin_lookup_function: resolved '%s' -> %p", rf->device_name, (void*)func);
                } else {
                    hip_remote_log_debug("hip_fatbin_lookup_function: resolve failed for '%s' (err=%d)", rf->device_name, err);
                    hip_mutex_unlock(&g_registered_function_lock);
                    return NULL;
                }
            }

            hipFunction_t func = rf->remote_function;
            hip_mutex_unlock(&g_registered_function_lock);
            return func;
        }
    }

    hip_mutex_unlock(&g_registered_function_lock);
    hip_remote_log_debug("hip_fatbin_lookup_function: host %p not found", hostFunction);
    return NULL;
}
