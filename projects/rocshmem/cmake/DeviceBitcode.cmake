###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# Function to add device bitcode compilation target
# Creates librocshmem_device.bc from rocSHMEM source files

option(BUILD_DEVICE_BITCODE "Build device bitcode library for JIT linking" ON)

if(NOT BUILD_DEVICE_BITCODE)
  return()
endif()

# Find LLVM tools
find_program(LLVM_CLANG clang++ PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH REQUIRED)
find_program(LLVM_LINK llvm-link PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH REQUIRED)

if(NOT LLVM_CLANG OR NOT LLVM_LINK)
  message(WARNING "LLVM tools not found. Skipping device bitcode compilation.")
  return()
endif()

# Set default GPU architecture for bitcode
set(BITCODE_GPU_ARCH "gfx942" CACHE STRING "GPU architecture for device bitcode")

# Compiler flags for device-only compilation
set(BITCODE_COMPILE_FLAGS
    -x hip
    --cuda-device-only
    -std=c++20
    -emit-llvm
    --offload-arch=${BITCODE_GPU_ARCH}
    -I${CMAKE_CURRENT_SOURCE_DIR}/include/rocshmem
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${CMAKE_BINARY_DIR}/include
    -I${CMAKE_BINARY_DIR}/include/rocshmem
)

# Add MPI include directories if available
if(MPI_CXX_FOUND)
  foreach(mpi_include_dir ${MPI_CXX_INCLUDE_DIRS})
    list(APPEND BITCODE_COMPILE_FLAGS -I${mpi_include_dir})
  endforeach()
endif()

# List of source files to compile into bitcode
# Core device sources
set(BITCODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rocshmem_gpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/team.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sync/abql_block_mutex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/context_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/device/rocshmem_wrapper.cc
)

# Add reverse offload backend sources if enabled
if(USE_RO)
  list(APPEND BITCODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reverse_offload/backend_ro.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reverse_offload/context_ro_device.cpp
  )
endif()

# Add IPC backend sources if enabled
if(USE_IPC)
  list(APPEND BITCODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/backend_ipc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/context_ipc_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/context_ipc_device_coll.cpp
  )
endif()

# Add GDA backend sources if enabled
if(USE_GDA)
  list(APPEND BITCODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/context_gda_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/context_gda_device_coll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/backend_gda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/queue_pair.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/ionic/queue_pair_ionic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/mlx5/queue_pair_mlx5.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/mlx5/segment_builder.cpp
  )
endif()

# Generate list of bitcode outputs
set(BITCODE_OBJECTS)
foreach(src_file ${BITCODE_SOURCES})
  get_filename_component(src_name ${src_file} NAME_WE)
  set(bc_file ${CMAKE_CURRENT_BINARY_DIR}/bitcode/${src_name}.bc)
  list(APPEND BITCODE_OBJECTS ${bc_file})

  # Add custom command to compile each source to bitcode
  add_custom_command(
    OUTPUT ${bc_file}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/bitcode
    COMMAND ${LLVM_CLANG} ${BITCODE_COMPILE_FLAGS} -c ${src_file} -o ${bc_file}
    DEPENDS ${src_file}
    COMMENT "Compiling ${src_name} to bitcode"
    VERBATIM
  )
endforeach()

# Final bitcode library output
set(BITCODE_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/librocshmem_device.bc)

# Add custom command to link all bitcode files
add_custom_command(
  OUTPUT ${BITCODE_OUTPUT}
  COMMAND ${LLVM_LINK} ${BITCODE_OBJECTS} -o ${BITCODE_OUTPUT}
  DEPENDS ${BITCODE_OBJECTS}
  COMMENT "Linking device bitcode into librocshmem_device.bc"
  VERBATIM
)

# Create custom target for bitcode compilation
add_custom_target(rocshmem_device_bitcode ALL
  DEPENDS ${BITCODE_OUTPUT}
)

# Install the bitcode library
install(
  FILES ${BITCODE_OUTPUT}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT runtime
)

message(STATUS "Device bitcode will be built for ${BITCODE_GPU_ARCH}")
message(STATUS "Device bitcode output: ${BITCODE_OUTPUT}")
