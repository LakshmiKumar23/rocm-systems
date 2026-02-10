/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

namespace amd::device {

#define BLIT_KERNELS(...) #__VA_ARGS__

const char* BlitLinearSourceCode = BLIT_KERNELS(
    // Extern
    extern void __amd_fillBufferAligned(__global uchar*, __global ushort*, __global uint*,
                                        __global ulong*, __constant uchar*, uint, ulong, ulong);

    extern void __amd_fillBufferAligned2D(__global uchar*, __global ushort*, __global uint*,
                                          __global ulong*, __constant uchar*, uint, ulong, ulong,
                                          ulong, ulong);

    extern void __amd_copyBuffer(__global uchar*, __global uchar*, ulong, ulong, ulong, uint);

    extern void __amd_copyBufferAligned(__global uint*, __global uint*, ulong, ulong, ulong, uint);

    extern void __amd_copyBufferRect(__global uchar*, __global uchar*, ulong4, ulong4, ulong4);

    extern void __amd_copyBufferRectAligned(__global uint*, __global uint*, ulong4, ulong4, ulong4);

    extern void __amd_streamOpsWrite(__global uint*, __global ulong*, ulong);

    extern void __amd_streamOpsWait(__global uint*, __global ulong*, ulong, ulong, ulong);

    extern void __amd_batchMemOp(__global void*, uint count);

    extern void __ockl_dm_init_v1(ulong, ulong, uint, uint);

__attribute__((always_inline)) inline ulong __amd_alignUp(ulong ptr, ulong alignment) {
  return (ptr + alignment - 1) & ~(alignment - 1);
}
// Note: alignement of arguments matters! Have small data types at end
// Otherwise loading values after arn't aligned in the argument buffer!// Assume pattern is an 32 bit int for now
__kernel void __amd_rocclr_fillBufferUnAligned(
    __global void* __restrict buf, __constant uchar* __restrict pattern,
    int body_pattern, ulong2 body_tile_pattern, ulong body_tile_count, ulong body_tile_passes,
    ulong stride, ushort body_count, ushort body_tail_count, ushort head_count,
    ushort tail_count) {
  uint l = __builtin_amdgcn_workitem_id_x();
  uint g = __builtin_amdgcn_workgroup_id_x();
  ulong id = (g * 256 + l);

  __global uchar* head_tail_element = (__global uchar*)buf;
  __global ulong2* element_tiled =
      ((__global ulong2*)__amd_alignUp((ulong)buf, sizeof(ulong2)));

  // Handle head, body and tail in warp 1 in first 12 threads
  // TO TEST: load pattern once then warp broad cast it to all warp 1 threads
  // Currently there is a wait in the code-gen for the head and tail writes
  if (id < head_count) {  // Copy head
    head_tail_element[id] = pattern[id];
  } else if (id >= head_count && id < head_count + tail_count) {  // Copy remainder to tail
    ulong tail_offset =
        head_count + body_count * sizeof(int) + body_tile_count * sizeof(ulong2) +
        body_tail_count * sizeof(int);
    head_tail_element[id + tail_offset] = pattern[id];
  } else if ((id >= 11) && (id < 11 + body_tail_count)) {
    // Copy shifted body_pattern to the region just before the final tail bytes
    ulong body_tail_offset =
        head_count + body_count * sizeof(int) + body_tile_count * sizeof(ulong2);
    __global int* body_tail_element =
        (__global int*)(head_tail_element + body_tail_offset);
    body_tail_element[id - 11] = body_pattern;
  } else if ((id >= 8) && (id < body_count + 8)) {
    // Copy shifted body_pattern
    __global int* body_element = (__global int*)__amd_alignUp((ulong)buf, sizeof(int));
    body_element[id - 8] = body_pattern;
  }

  // We pass in the number of passes from the CPU to get the best code-gen
  // We use the number of passes and the size to get correct behiaviour
  for (ulong j = 0; (j < body_tile_passes) && (j * stride + id < body_tile_count); ++j) {
    element_tiled[j * stride + id] = body_tile_pattern;
  }
}

    __kernel void __amd_rocclr_fillBufferAligned(__global void* buf, __constant uchar* pattern,
                                                 uint pattern_size, uint alignment, ulong end_ptr,
                                                 uint next_chunk, uint workgroup_size) {
      uint l = __builtin_amdgcn_workitem_id_x();
      uint g = __builtin_amdgcn_workgroup_id_x();
      ulong id = (g * workgroup_size + l);
      long cur_id = id * pattern_size;
      if (alignment == sizeof(ulong2)) {
        __global ulong2* bufULong2 = (__global ulong2*)buf;
        __global ulong2* element = &bufULong2[cur_id];
        __constant ulong2* pt = (__constant ulong2*)pattern;
        while ((ulong)element < end_ptr) {
          for (uint i = 0; i < pattern_size; ++i) {
            element[i] = pt[i];
          }
          element += next_chunk;
        }
      } else if (alignment == sizeof(ulong)) {
        __global ulong* bufULong = (__global ulong*)buf;
        __global ulong* element = &bufULong[cur_id];
        __constant ulong* pt = (__constant ulong*)pattern;
        while ((ulong)element < end_ptr) {
          for (uint i = 0; i < pattern_size; ++i) {
            element[i] = pt[i];
          }
          element += next_chunk;
        }
      } else if (alignment == sizeof(uint)) {
        __global uint* bufUInt = (__global uint*)buf;
        __global uint* element = &bufUInt[cur_id];
        __constant uint* pt = (__constant uint*)pattern;
        while ((ulong)element < end_ptr) {
          for (uint i = 0; i < pattern_size; ++i) {
            element[i] = pt[i];
          }
          element += next_chunk;
        }
      } else if (alignment == sizeof(ushort)) {
        __global ushort* bufUShort = (__global ushort*)buf;
        __global ushort* element = &bufUShort[cur_id];
        __constant ushort* pt = (__constant ushort*)pattern;
        while ((ulong)element < end_ptr) {
          for (uint i = 0; i < pattern_size; ++i) {
            element[i] = pt[i];
          }
          element += next_chunk;
        }
      } else {
        __global uchar* bufUChar = (__global uchar*)buf;
        __global uchar* element = &bufUChar[cur_id];
        while ((ulong)element < end_ptr) {
          for (uint i = 0; i < pattern_size; ++i) {
            element[i] = pattern[i];
          }
          element += next_chunk;
        }
      }
    }

    __kernel void __amd_rocclr_fillBufferAligned2D(
        __global uchar* bufUChar, __global ushort* bufUShort, __global uint* bufUInt,
        __global ulong* bufULong, __constant uchar* pattern, uint patternSize, ulong offset,
        ulong width, ulong height, ulong pitch) {
      __amd_fillBufferAligned2D(bufUChar, bufUShort, bufUInt, bufULong, pattern, patternSize,
                                offset, width, height, pitch);
    }

    __kernel void __amd_rocclr_copyBuffer(__global uchar* src, __global uchar* dst, ulong size,
                                          uint remainder, uint aligned_size, ulong end_ptr,
                                          uint next_chunk, uint workgroup_size) {
      uint l = __builtin_amdgcn_workitem_id_x();
      uint g = __builtin_amdgcn_workgroup_id_x();
      ulong id = (g * workgroup_size + l);
      ulong id_remainder = id;

      if (aligned_size == sizeof(ulong2)) {
        __global ulong2* srcD = (__global ulong2*)(src);
        __global ulong2* dstD = (__global ulong2*)(dst);
        while ((ulong)(&dstD[id]) < end_ptr) {
          dstD[id] = srcD[id];
          id += next_chunk;
        }
      } else {
        __global uint* srcD = (__global uint*)(src);
        __global uint* dstD = (__global uint*)(dst);
        while ((ulong)(&dstD[id]) < end_ptr) {
          dstD[id] = srcD[id];
          id += next_chunk;
        }
      }
      if ((remainder != 0) && (id_remainder == 0)) {
        for (ulong i = size - remainder; i < size; ++i) {
          dst[i] = src[i];
        }
      }
    }

    __kernel void __amd_rocclr_copyBufferAligned(__global uint* src, __global uint* dst,
                                                 ulong srcOrigin, ulong dstOrigin, ulong size,
                                                 uint alignment) {
      __amd_copyBufferAligned(src, dst, srcOrigin, dstOrigin, size, alignment);
    }

    __kernel void __amd_rocclr_copyBufferRect(__global uchar* src, __global uchar* dst,
                                              ulong4 srcRect, ulong4 dstRect, ulong4 size) {
      __amd_copyBufferRect(src, dst, srcRect, dstRect, size);
    }

    __kernel void __amd_rocclr_copyBufferRectAligned(__global uint* src, __global uint* dst,
                                                     ulong4 srcRect, ulong4 dstRect, ulong4 size) {
      __amd_copyBufferRectAligned(src, dst, srcRect, dstRect, size);
    }

    __kernel void __amd_rocclr_batchMemOp(__global void* params, uint count) {
      __amd_batchMemOp(params, count);
    });

const char* HipExtraSourceCode = BLIT_KERNELS(
    __kernel void __amd_rocclr_streamOpsWrite(__global uint* ptrInt, __global ulong* ptrUlong,
                                              ulong value) {
      __amd_streamOpsWrite(ptrInt, ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsWait(__global uint* ptrInt, __global ulong* ptrUlong,
                                             ulong value, ulong flags, ulong mask) {
      __amd_streamOpsWait(ptrInt, ptrUlong, value, flags, mask);
    }

    __kernel void __amd_rocclr_initHeap(ulong heap_to_initialize, ulong initial_blocks,
                                        uint heap_size, uint number_of_initial_blocks) {
      __ockl_dm_init_v1(heap_to_initialize, initial_blocks, heap_size, number_of_initial_blocks);
    }

    __kernel void __amd_rocclr_gwsInit(uint value) { __builtin_amdgcn_ds_gws_init(value, 0); });

const char* HipExtraSourceCodeNoGWS = BLIT_KERNELS(
    __kernel void __amd_rocclr_streamOpsWrite(__global uint* ptrInt, __global ulong* ptrUlong,
                                              ulong value) {
      __amd_streamOpsWrite(ptrInt, ptrUlong, value);
    }

    __kernel void __amd_rocclr_streamOpsWait(__global uint* ptrInt, __global ulong* ptrUlong,
                                             ulong value, ulong flags, ulong mask) {
      __amd_streamOpsWait(ptrInt, ptrUlong, value, flags, mask);
    }

    __kernel void __amd_rocclr_initHeap(ulong heap_to_initialize, ulong initial_blocks,
                                        uint heap_size, uint number_of_initial_blocks) {
      __ockl_dm_init_v1(heap_to_initialize, initial_blocks, heap_size, number_of_initial_blocks);
    });

const char* BlitImageSourceCode = BLIT_KERNELS(
    // Extern
    extern void __amd_fillImage(__write_only image2d_array_t, float4, int4, uint4, int4, int4,
                                uint);

    extern void __amd_copyImage(__read_only image2d_array_t, __write_only image2d_array_t, int4,
                                int4, int4);

    extern void __amd_copyImage1DA(__read_only image2d_array_t, __write_only image2d_array_t, int4,
                                   int4, int4);

    extern void __amd_copyBufferToImage(__global uint*, __write_only image2d_array_t, ulong4, int4,
                                        int4, uint4, ulong4);

    extern void __amd_copyImageToBuffer(__read_only image2d_array_t, __global uint*,
                                        __global ushort*, __global uchar*, int4, ulong4, int4,
                                        uint4, ulong4);

    __kernel void __amd_rocclr_fillImage(__write_only image2d_array_t image, float4 patternFLOAT4,
                                         int4 patternINT4, uint4 patternUINT4, int4 origin,
                                         int4 size, uint type) {
      __amd_fillImage(image, patternFLOAT4, patternINT4, patternUINT4, origin, size, type);
    }

    __kernel void __amd_rocclr_copyImage(
        __read_only image2d_array_t src, __write_only image2d_array_t dst, int4 srcOrigin,
        int4 dstOrigin, int4 size) { __amd_copyImage(src, dst, srcOrigin, dstOrigin, size); }

    __kernel void __amd_rocclr_copyImage1DA(
        __read_only image2d_array_t src, __write_only image2d_array_t dst, int4 srcOrigin,
        int4 dstOrigin, int4 size) { __amd_copyImage1DA(src, dst, srcOrigin, dstOrigin, size); }

    __kernel void __amd_rocclr_copyBufferToImage(
        __global uint* src, __write_only image2d_array_t dst, ulong4 srcOrigin, int4 dstOrigin,
        int4 size, uint4 format, ulong4 pitch) {
      __amd_copyBufferToImage(src, dst, srcOrigin, dstOrigin, size, format, pitch);
    }

    __kernel void __amd_rocclr_copyImageToBuffer(
        __read_only image2d_array_t src, __global uint* dstUInt, __global ushort* dstUShort,
        __global uchar* dstUChar, int4 srcOrigin, ulong4 dstOrigin, int4 size, uint4 format,
        ulong4 pitch) {
      __amd_copyImageToBuffer(src, dstUInt, dstUShort, dstUChar, srcOrigin, dstOrigin, size, format,
                              pitch);
    });

}  // namespace amd::device
