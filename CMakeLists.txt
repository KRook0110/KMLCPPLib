cmake_minimum_required(VERSION 3.14)

project(KMLCPPLib VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} STATIC
    src/core.cpp
    src/Parameter.cpp
    src/Layers/LinearLayer.cpp
    src/LossFunctions/MSELossFunction.cpp
    src/Layers/SigmoidLayer.cpp
    src/Layers/SoftmaxLayer.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

add_executable(TestingNN cmd/main.cpp)
target_link_libraries(TestingNN PRIVATE ${PROJECT_NAME})

# GTest Integration
include(CTest)
include(GoogleTest)

if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.0.zip
        FIND_PACKAGE_ARGS NAMES GTest
    )
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()
