cmake_minimum_required(VERSION 3.10)

project(KMLCPPLib VERSION 1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_library(${PROJECT_NAME} STATIC
    src/core.cpp
    src/Parameter.cpp
    src/Layers/LinearLayer.cpp
    src/LossFunctions/MSELossLayer.cpp
    src/Layers/SigmoidLayer.cpp
    src/Layers/SoftmaxLayer.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC include)

add_executable(TestingNN cmd/main.cpp)

target_link_libraries(TestingNN PRIVATE ${PROJECT_NAME})
