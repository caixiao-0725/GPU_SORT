cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Sort CUDA CXX)

# Enable CUDA
enable_language(CUDA)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)  # Adjust based on your target GPU architecture

# Add executable
add_executable(RadixSort
    common.cuh
    dataInit.cuh
    oneSweep.cuh
    oneSweep.cu
    main.cu
)

# Add include directories
target_include_directories(RadixSort PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set CUDA specific properties
set_target_properties(RadixSort PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
