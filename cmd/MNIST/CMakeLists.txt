add_executable(ViewMnistImages ./ViewMnistImages.cpp)

add_executable(SolveMnist ./SolveMnist.cpp)

target_link_libraries(ViewMnistImages PRIVATE ${PROJECT_NAME})
target_link_libraries(SolveMnist PRIVATE ${PROJECT_NAME})
