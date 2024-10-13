cmake_minimum_required(VERSION 3.17)

project(ascii_rt LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Find and include Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories("/usr/include/eigen3")
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")


# Add executable and make sure all relevant source files are included
add_executable(${PROJECT_NAME} 
    src/main.cu 
    src/utils/ray.cu 
    src/utils/load_mesh.cu 
    src/shapes/triangle.cu 
    src/shapes/mesh.cu        # Add this line to include mesh.cpp
    src/shapes/sphere.cu      # Add this line to include sphere.cpp
    src/utils/intersect.cu
    src/cuda_rt.cu
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       -dc
                       -diag-suppress 20012 # ignore eigen known issue
                       -O2
                       -Wall
                       --expt-relaxed-constexpr
                       >)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;61;70")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} cuda)
