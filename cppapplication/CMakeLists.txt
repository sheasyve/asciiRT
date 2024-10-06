cmake_minimum_required(VERSION 3.10)

project(ascii_rt)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# Find and include Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories("/usr/include/eigen3")

# Add executable and make sure all relevant source files are included
add_executable(${PROJECT_NAME} 
    src/main.cpp 
    src/ray.cpp 
    src/obj.cpp 
    src/triangle.cpp 
    src/mesh.cpp        # Add this line to include mesh.cpp
    src/sphere.cpp      # Add this line to include sphere.cpp
)

# Link Eigen3
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
