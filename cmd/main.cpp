#include <Eigen/Dense>
#include <KMLCPPLib/Layer.hpp>
#include <iostream>

int main() {
    // --- Simple linear regression test: learn y = 2x + 1 ---
    kmlcpplib::LinearLayer linearLayer(1, 1);
    kmlcpplib::MSELossLayer lossLayer(1);

    // Sample data: (x, y) pairs for y = 2x + 1
    const int N = 4;
    Eigen::VectorXd xs[N], ys[N];
    double x_vals[] = {1.0, 2.0, 3.0, 4.0};
    double y_vals[] = {3.0, 5.0, 7.0, 9.0};
    for (int i = 0; i < N; ++i) {
        xs[i] = Eigen::VectorXd::Constant(1, x_vals[i]);
        ys[i] = Eigen::VectorXd::Constant(1, y_vals[i]);
    }

    const int epochs = 100;
    const float lr = 0.01f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (int i = 0; i < N; ++i) {
            // Forward pass through linear layer
            Eigen::VectorXd y_pred = linearLayer.forward_feed(xs[i]);

            // Compute MSE loss (forward_feed takes y_pred and y_true)
            Eigen::VectorXd loss = lossLayer.forward_feed(y_pred, ys[i]);
            total_loss += loss(0);

            // Backward pass through loss layer (returns gradient w.r.t. y_pred)
            Eigen::VectorXd grad = lossLayer.backward_prop(Eigen::MatrixXd(), lr);

            // Backward pass through linear layer
            linearLayer.backward_prop(grad, lr);

            // Update weights
            linearLayer.update();
        }

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " | Avg Loss: "
                      << total_loss / N << std::endl;
        }
    }

    // Final predictions
    std::cout << "\n--- Final Predictions (target: y = 2x + 1) ---" << std::endl;
    for (int i = 0; i < N; ++i) {
        Eigen::VectorXd y_pred = linearLayer.forward_feed(xs[i]);
        std::cout << "x=" << x_vals[i]
                  << "  predicted=" << y_pred(0)
                  << "  actual=" << y_vals[i] << std::endl;
    }

    // Test on unseen data
    std::cout << "\n--- Unseen Data ---" << std::endl;
    double test_xs[] = {0.0, 5.0, 10.0};
    for (double tx : test_xs) {
        Eigen::VectorXd input = Eigen::VectorXd::Constant(1, tx);
        Eigen::VectorXd y_pred = linearLayer.forward_feed(input);
        std::cout << "x=" << tx
                  << "  predicted=" << y_pred(0)
                  << "  expected=" << (2.0 * tx + 1.0) << std::endl;
    }

    std::cout << "\n--- Testing Softmax Layer ---" << std::endl;
    kmlcpplib::SoftmaxLayer softmaxLayer(3);
    Eigen::VectorXd test_x(3);
    test_x << 1.0, 2.0, 3.0;
    Eigen::VectorXd sm_out = softmaxLayer.forward_feed(test_x);
    std::cout << "Softmax forward output: " << sm_out.transpose() << std::endl;
    Eigen::VectorXd sm_past(3);
    sm_past << 0.1, 0.2, 0.3;
    Eigen::VectorXd sm_grad = softmaxLayer.backward_prop(sm_past, 0.01f);
    std::cout << "Softmax backward gradient: " << sm_grad.transpose() << std::endl;

    return 0;
}
