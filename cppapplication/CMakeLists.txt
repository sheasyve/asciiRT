cmake_minimum_required(VERSION 3.10)

project(ascii_rt)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories("/usr/include/eigen3")

add_executable(${PROJECT_NAME} 
    src/main.cpp 
    src/utils/bvh.cpp 
    src/utils/ascii_print.cpp 
    src/utils/main_util.cpp 
    src/utils/matrix_util.cpp 
    src/utils/load_mesh.cpp 
    src/shapes/triangle.cpp 
    src/shapes/mesh.cpp
    src/utils/rt_util.cpp        
)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
