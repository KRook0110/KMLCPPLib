add_executable(TestingNN main.cpp)

target_link_libraries(TestingNN PRIVATE ${PROJECT_NAME})

if(MNIST)
    add_subdirectory(MNIST)
endif()
