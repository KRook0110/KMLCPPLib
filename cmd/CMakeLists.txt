add_executable(TestingNN main.cpp)
add_executable(ViewMnistImages ./ViewMnistImages.cpp)
target_link_libraries(TestingNN PRIVATE ${PROJECT_NAME})
target_link_libraries(ViewMnistImages PRIVATE ${PROJECT_NAME})
