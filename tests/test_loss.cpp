#include <gtest/gtest.h>
#include <KMLCPPLib/LossFunctions/MSELossFunction.hpp>

TEST(MSELossFunctionTest, ForwardAndBackward) {
    kmlcpplib::MSELossFunction loss_layer(2);

    Eigen::VectorXd y_pred(2);
    y_pred << 2.0, 4.0;

    Eigen::VectorXd y_true(2);
    y_true << 1.0, 1.0;

    // Loss = ((2-1)^2 + (4-1)^2) / 2 = (1 + 9) / 2 = 5.0
    double loss = loss_layer.forward(y_pred, y_true);
    EXPECT_NEAR(loss, 5.0, 1e-6);

    // Grad = 2 * (y_pred - y_true) / 2 = [1.0, 3.0]
    Eigen::VectorXd grad = loss_layer.backward();
    EXPECT_EQ(grad.rows(), 2);
    EXPECT_NEAR(grad(0), 1.0, 1e-6);
    EXPECT_NEAR(grad(1), 3.0, 1e-6);
}
