cmake_minimum_required(VERSION 3.10)

project(KMLCPPLib VERSION 1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen5 5.0.0 REQUIRED NO_MODULE)

add_library(${PROJECT_NAME} STATIC
    src/core.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC include)

add_executable(TestingNN cmd/main.cpp)

target_link_libraries(TestingNN PRIVATE ${PROJECT_NAME})
