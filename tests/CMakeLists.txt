add_executable(unit_tests
    test_parameter.cpp
    test_layers.cpp
    test_loss.cpp
)

target_link_libraries(unit_tests PRIVATE ${PROJECT_NAME} GTest::gtest_main GTest::gtest)

gtest_discover_tests(unit_tests)
