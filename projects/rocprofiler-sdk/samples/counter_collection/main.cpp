// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <thread>

#include <libgen.h>
#include "client.hpp"

#define HIP_CALL(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr, "%s\n", hipGetErrorString(err));                                       \
            abort();                                                                               \
        }                                                                                          \
    } while(0)

__global__ void
kernelA(int* flag)
{
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        atomicAdd(flag, 1);
        asm volatile("s_dcache_wb;"
                    "s_waitcnt lgkmcnt(0)"
                    "buffer_wbl2;"
                    "s_waitcnt vmcnt(0)");
    }
    __syncthreads();

    int result = 1;
    while (result != 0)
    {
        for (int i=0; i<25; i++) asm volatile("s_sleep 63"); // 50us

        asm volatile("s_dcache_inv; "
                     "s_waitcnt lgkmcnt(0)");
        asm volatile("s_load_dword %0, %1, 0" : "=r"(result) : "r"(flag));
        asm volatile("s_waitcnt lgkmcnt(0)");
    }
}

int* flag = nullptr;

void LaunchAndConfirm()
{
    int thr = 256;
    int blk = 2;
    *flag = 0;
    hipLaunchKernelGGL(kernelA, blk, thr, 0, 0, (int*)flag);
    while (*flag != blk) std::this_thread::sleep_for(std::chrono::microseconds(50));
}

int
main(int argc, char** argv)
{
    int ntotdevice = 0;
    HIP_CALL(hipGetDeviceCount(&ntotdevice));

    // Normal HIP Calls
    HIP_CALL(hipSetDevice(0));
    [[maybe_unused]] hipDeviceProp_t devProp;
    HIP_CALL(hipGetDeviceProperties(&devProp, 0));

    HIP_CALL(hipMallocHost((void**)&flag, 4096));
    HIP_CALL(hipDeviceSynchronize());

    std::cout << "Start before kernel, read after kernel: " << std::endl;
    start();
    LaunchAndConfirm();
    *flag = 0;
    HIP_CALL(hipDeviceSynchronize());
    read();
    
    std::cout << "Start before kernel, read during kernel: " << std::endl;
    start();
    LaunchAndConfirm();
    read();
    *flag = 0;
    HIP_CALL(hipDeviceSynchronize());

    std::cout << "Start during kernel, read after kernel: " << std::endl;
    LaunchAndConfirm();
    start();
    *flag = 0;
    HIP_CALL(hipDeviceSynchronize()); 
    read();
    
    std::cout << "Start during kernel, read during kernel: " << std::endl;
    LaunchAndConfirm();
    start();
    read();
    *flag = 0;
    HIP_CALL(hipDeviceSynchronize());
}
