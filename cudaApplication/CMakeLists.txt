cmake_minimum_required(VERSION 3.17)

project(ascii_rt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Include Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include CUDA
find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(${PROJECT_NAME} 
    src/main.cu
    src/utils/load_mesh.cu 
    src/shapes/triangle.cu 
    src/shapes/mesh.cu     
    src/cuda_rt.cu
    src/utils/bvh.cu
    src/utils/cuda_rt_util.cu
    src/utils/ascii_print.cu
    src/utils/main_util.cu
    src/utils/matrix_util.cu
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${EIGEN3_INCLUDE_DIR} 
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Compile options
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

# Set CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_ARCHITECTURES "50;86"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    Eigen3::Eigen 
    ${CUDAToolkit_LIBRARIES}
)
