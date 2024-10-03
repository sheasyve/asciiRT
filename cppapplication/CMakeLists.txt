cmake_minimum_required(VERSION 3.10)

project(ascii_rt)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories("/usr/include/eigen3")

add_executable(${PROJECT_NAME} src/main.cpp src/ray.cpp src/obj.cpp src/triangle.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE Eigen3::Eigen)