cmake_minimum_required(VERSION 3.14)

project(KMLCPPLib VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

add_subdirectory(src)


add_subdirectory(cmd)

# Testing Integration
include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

