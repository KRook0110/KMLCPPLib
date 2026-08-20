include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.0.zip
    FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(unit_tests
    test_parameter.cpp
    test_layers.cpp
    test_loss.cpp
    test_randomized_weight.cpp
    test_optimizer.cpp
)

target_link_libraries(unit_tests PRIVATE ${PROJECT_NAME} GTest::gtest_main GTest::gtest)

gtest_discover_tests(unit_tests)
