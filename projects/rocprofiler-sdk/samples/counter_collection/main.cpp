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
    for (int i=0; i<10000; i++) asm volatile("s_sleep 63"); // 2us * 1000 * 100 = 200ms
}

int
main(int argc, char** argv)
{
    int thr = 256;
    int blk = 256;
    int ntotdevice = 0;
    HIP_CALL(hipGetDeviceCount(&ntotdevice));

    // Normal HIP Calls
    HIP_CALL(hipSetDevice(0));
    [[maybe_unused]] hipDeviceProp_t devProp;
    HIP_CALL(hipGetDeviceProperties(&devProp, 0));

    HIP_CALL(hipDeviceSynchronize());

    std::cout << "Start before kernel, read after kernel: " << std::endl;
    start();
    hipLaunchKernelGGL(kernelA, blk, thr, 0, 0, nullptr);
    HIP_CALL(hipDeviceSynchronize());
    read();
    
    std::cout << "Start before kernel, read during kernel: " << std::endl;
    start();
    hipLaunchKernelGGL(kernelA, blk, thr, 0, 0, nullptr);
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    read();
    HIP_CALL(hipDeviceSynchronize());

    std::cout << "Start during kernel, read after kernel: " << std::endl;
    hipLaunchKernelGGL(kernelA, blk, thr, 0, 0, nullptr);
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    start();
    HIP_CALL(hipDeviceSynchronize()); 
    read();

    HIP_CALL(hipDeviceSynchronize());
}
