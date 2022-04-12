# Find compute capability
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

file(GLOB SOURCE "src/*.cu" "src/*.cuh" "./*.cu")

add_executable(DE_cuda ${SOURCE})

set_target_properties(DE_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(DE_cuda CPU_DE)