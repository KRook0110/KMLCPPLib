add_library(${PROJECT_NAME} STATIC
    core.cpp
    Parameter.cpp
    Layers/LinearLayer.cpp
    LossFunctions/MSELossFunction.cpp
    Layers/SigmoidLayer.cpp
    Layers/SoftmaxLayer.cpp
    Layers/ReluLayer.cpp
    Layers/SequenceLayer.cpp
    Optimizers/GradientDescentOptimizer.cpp
    WeightInitializers/RandomizedWeight.cpp
    Parsers/MNIST/MNISTReader.cpp
    Parsers/MNIST/MNISTImageViewer.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
