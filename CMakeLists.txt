cmake_minimum_required(VERSION 3.21)
project(DE_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#add_compile_definitions(EXPONENTIAL_CROSSOVER)

add_subdirectory(gpu)
add_subdirectory(cpu)