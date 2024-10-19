cmake_minimum_required(VERSION 3.17)

project(ascii_rt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories("/usr/include/eigen3")
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

add_executable(${PROJECT_NAME} 
    src/main.cu
    src/utils/load_mesh.cu 
    src/shapes/triangle.cu 
    src/shapes/mesh.cu     
    src/cuda_rt.cu
    src/utils/bvh.cu
    src/utils/cuda_rt_util.cu
    src/utils/ascii_print.cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        -dc
        -diag-suppress 20012 
        -O2
        -Wall
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -Wall
        -Wextra
    >
)

set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_ARCHITECTURES "50;86"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen cuda)
