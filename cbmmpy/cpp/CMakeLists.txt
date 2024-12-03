cmake_minimum_required(VERSION 3.22)

# Give the project a name
project(template)

# Set C++ version
set(CMAKE_CXX_STANDARD 20)

# Bring the headers into the project
include_directories(include)
include_directories(include/Eigen)

# For Windows
include_directories(C:/Users/Daniel/anaconda3/pkgs/pybind11-2.9.2-py38h757b272_0/Lib/site-packages/pybind11/include)

# Add source files
file(GLOB SOURCES "src/*.cpp")

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES})
