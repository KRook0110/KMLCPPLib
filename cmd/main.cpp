#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>
#include <random>

#include "KMLCPPLib/Layers/LinearLayer.hpp"
#include "KMLCPPLib/Layers/SigmoidLayer.hpp"
#include "KMLCPPLib/Layers/SequenceLayer.hpp"
#include "KMLCPPLib/LossFunctions/MSELossFunction.hpp"

int main() {
    // 1. XOR Dataset
    std::vector<Eigen::VectorXd> X = {
        (Eigen::VectorXd(2) << 0.0, 0.0).finished(),
        (Eigen::VectorXd(2) << 0.0, 1.0).finished(),
        (Eigen::VectorXd(2) << 1.0, 0.0).finished(),
        (Eigen::VectorXd(2) << 1.0, 1.0).finished()
    };

    std::vector<Eigen::VectorXd> Y = {
        (Eigen::VectorXd(1) << 0.0).finished(),
        (Eigen::VectorXd(1) << 1.0).finished(),
        (Eigen::VectorXd(1) << 1.0).finished(),
        (Eigen::VectorXd(1) << 0.0).finished()
    };

    // 2. Build Model Architecture: 2 inputs -> 4 hidden -> 1 output
    auto l1 = std::make_shared<kmlcpplib::LinearLayer>(2, 4);
    auto a1 = std::make_shared<kmlcpplib::SigmoidLayer>(4);
    auto l2 = std::make_shared<kmlcpplib::LinearLayer>(4, 1);
    auto a2 = std::make_shared<kmlcpplib::SigmoidLayer>(1);

    kmlcpplib::SequenceLayer model({l1, a1, l2, a2});
    kmlcpplib::MSELossFunction loss_fn(1);

    // Randomize initial weights (since parameters default to zero)
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.5);

    auto params = model.get_params();
    for (auto &param : params) {
        for (int r = 0; r < param->value.rows(); ++r) {
            for (int c = 0; c < param->value.cols(); ++c) {
                param->value(r, c) = dist(rng);
            }
        }
    }

    // 3. Training Loop
    const int epochs = 10000;
    const double learning_rate = 0.5;

    std::cout << "Training XOR model for " << epochs << " epochs..." << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < X.size(); ++i) {
            // Zero gradients
            for (auto &param : params) {
                param->zero_grad();
            }

            // Forward pass
            Eigen::VectorXd pred = model.forward(X[i]);
            double loss = loss_fn.forward(pred, Y[i]);
            total_loss += loss;

            // Backward pass
            Eigen::VectorXd loss_grad = loss_fn.backward();
            model.backward(loss_grad);

            // Gradient Descent Update
            for (auto &param : params) {
                param->value -= learning_rate * param->gradient;
            }
        }

        if (epoch % 1000 == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << " | Total MSE Loss: " << total_loss / X.size() << std::endl;
        }
    }

    // 4. Evaluate Model Predictions
    std::cout << "\n=== XOR Model Predictions ===" << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        Eigen::VectorXd pred = model.forward(X[i]);
        std::cout << "Input: [" << X[i](0) << ", " << X[i](1) << "]"
                  << " -> Target: " << Y[i](0)
                  << " | Prediction: " << pred(0)
                  << " (" << (pred(0) >= 0.5 ? 1 : 0) << ")"
                  << std::endl;
    }

    return 0;
}
