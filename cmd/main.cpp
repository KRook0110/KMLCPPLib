#include <Eigen/Dense>
#include <iostream>
#include "KMLCPPLib/Layers/SigmoidLayer.hpp"

int main() {
    // 1. Define an input vector (4 elements)
    Eigen::VectorXd input(4);
    input << -2.0, 0.0, 1.0, 5.0;

    // 2. Initialize SigmoidLayer with 4 input nodes
    kmlcpplib::SigmoidLayer sigmoid(4);

    // 3. Forward pass: computes sigmoid(x) = 1 / (1 + e^-x)
    Eigen::VectorXd output = sigmoid.forward(input);

    std::cout << "=== Sigmoid Layer Forward Pass ===" << std::endl;
    for (int i = 0; i < input.size(); ++i) {
        std::cout << "x[" << i << "] = " << input[i]
                  << "  -->  sigmoid(x[" << i << "]) = " << output[i] << std::endl;
    }

    // 4. Backward pass: computes dx = output * (1 - output) * upstream_grad
    Eigen::VectorXd upstream_grad = Eigen::VectorXd::Ones(4);
    Eigen::VectorXd input_grad = sigmoid.backward(upstream_grad);

    std::cout << "\n=== Sigmoid Layer Backward Pass ===" << std::endl;
    std::cout << "Upstream Gradient: " << upstream_grad.transpose() << std::endl;
    for (int i = 0; i < input.size(); ++i) {
        std::cout << "dx[" << i << "] = " << input_grad[i] << std::endl;
    }

    return 0;
}


