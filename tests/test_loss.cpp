#include <gtest/gtest.h>
#include <cmath>
#include <KMLCPPLib/LossFunctions/MeanSquaredError.hpp>
#include <KMLCPPLib/LossFunctions/CategoricalCrossEntropy.hpp>
#include <KMLCPPLib/LossFunctions/BinaryCrossEntropy.hpp>

TEST(MeanSquaredErrorTest, ForwardAndBackward) {
    kmlcpplib::MeanSquaredError loss_layer(2);

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


TEST(BinaryCrossEntropyTest, StandardPrediction) {
    kmlcpplib::BinaryCrossEntropy loss_layer;

    // Case 1: y_pred = 0.8, y_true = 1.0
    Eigen::VectorXd y_pred(1);
    y_pred << 0.8;
    Eigen::VectorXd y_true(1);
    y_true << 1.0;

    double loss = loss_layer.forward(y_pred, y_true);
    // Expected loss = -log(0.8) ≈ 0.22314355
    EXPECT_NEAR(loss, -std::log(0.8), 1e-6);

    Eigen::VectorXd grad = loss_layer.backward();
    EXPECT_EQ(grad.rows(), 1);
    // Expected grad = (0.8 - 1.0) / (0.8 * 0.2) = -0.2 / 0.16 = -1.25
    EXPECT_NEAR(grad(0), -1.25, 1e-6);

    // Case 2: y_pred = 0.3, y_true = 0.0
    y_pred << 0.3;
    y_true << 0.0;

    loss = loss_layer.forward(y_pred, y_true);
    // Expected loss = -log(0.7) ≈ 0.35667494
    EXPECT_NEAR(loss, -std::log(0.7), 1e-6);

    grad = loss_layer.backward();
    EXPECT_EQ(grad.rows(), 1);
    // Expected grad = (0.3 - 0.0) / (0.3 * 0.7) = 0.3 / 0.21 ≈ 1.4285714
    EXPECT_NEAR(grad(0), 0.3 / 0.21, 1e-6);
}

TEST(BinaryCrossEntropyTest, ExplosionZeroPredOneTrue) {
    kmlcpplib::BinaryCrossEntropy loss_layer;

    // y_pred = 0.0 and y_true = 1.0 (Potential log(0) explosion)
    Eigen::VectorXd y_pred(1);
    y_pred << 0.0;
    Eigen::VectorXd y_true(1);
    y_true << 1.0;

    double loss = loss_layer.forward(y_pred, y_true);

    // Check that loss is finite and not NaN/Inf
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_FALSE(std::isnan(loss));

    // Clamped y_pred is eps = 1e-7, loss should be -log(1e-7) ≈ 16.11809565
    double expected_loss = -std::log(1e-7);
    EXPECT_NEAR(loss, expected_loss, 1e-5);

    Eigen::VectorXd grad = loss_layer.backward();
    EXPECT_EQ(grad.rows(), 1);
    EXPECT_TRUE(std::isfinite(grad(0)));
    EXPECT_FALSE(std::isnan(grad(0)));

    // Clamped grad should be (1e-7 - 1.0) / (1e-7 * (1.0 - 1e-7)) ≈ -1e7
    double expected_grad = (1e-7 - 1.0) / (1e-7 * (1.0 - 1e-7));
    EXPECT_NEAR(grad(0), expected_grad, 1.0);
}

TEST(BinaryCrossEntropyTest, ExplosionOnePredZeroTrue) {
    kmlcpplib::BinaryCrossEntropy loss_layer;

    // y_pred = 1.0 and y_true = 0.0 (Potential log(1 - 1) explosion)
    Eigen::VectorXd y_pred(1);
    y_pred << 1.0;
    Eigen::VectorXd y_true(1);
    y_true << 0.0;

    double loss = loss_layer.forward(y_pred, y_true);

    // Check that loss is finite and not NaN/Inf
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_FALSE(std::isnan(loss));

    // Clamped y_pred is 1.0 - 1e-7, loss should be -log(1e-7) ≈ 16.11809565
    double expected_loss = -std::log(1e-7);
    EXPECT_NEAR(loss, expected_loss, 1e-5);

    Eigen::VectorXd grad = loss_layer.backward();
    EXPECT_EQ(grad.rows(), 1);
    EXPECT_TRUE(std::isfinite(grad(0)));
    EXPECT_FALSE(std::isnan(grad(0)));

    // Clamped grad should be ((1.0 - 1e-7) - 0.0) / ((1.0 - 1e-7) * 1e-7) ≈ 1e7
    double expected_grad = (1.0 - 1e-7) / ((1.0 - 1e-7) * 1e-7);
    EXPECT_NEAR(grad(0), expected_grad, 1.0);
}

TEST(BinaryCrossEntropyTest, PerfectPredictions) {
    kmlcpplib::BinaryCrossEntropy loss_layer;

    // y_pred = 1.0, y_true = 1.0
    Eigen::VectorXd y_pred(1);
    y_pred << 1.0;
    Eigen::VectorXd y_true(1);
    y_true << 1.0;

    double loss = loss_layer.forward(y_pred, y_true);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, 0.0, 1e-5);

    Eigen::VectorXd grad = loss_layer.backward();
    EXPECT_TRUE(std::isfinite(grad(0)));
    EXPECT_NEAR(grad(0), -1.0, 1e-5);

    // y_pred = 0.0, y_true = 0.0
    y_pred << 0.0;
    y_true << 0.0;

    loss = loss_layer.forward(y_pred, y_true);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, 0.0, 1e-5);

    grad = loss_layer.backward();
    EXPECT_TRUE(std::isfinite(grad(0)));
    EXPECT_NEAR(grad(0), 1.0, 1e-5);
}

TEST(BinaryCrossEntropyTest, ExtremeOutOfBoundsPred) {
    kmlcpplib::BinaryCrossEntropy loss_layer;

    // Negative y_pred (e.g. -10.0) should be clamped to eps
    Eigen::VectorXd y_pred(1);
    y_pred << -10.0;
    Eigen::VectorXd y_true(1);
    y_true << 1.0;

    double loss = loss_layer.forward(y_pred, y_true);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, -std::log(1e-7), 1e-5);

    // y_pred > 1.0 (e.g. 10.0) should be clamped to 1.0 - eps
    y_pred << 10.0;
    y_true << 0.0;

    loss = loss_layer.forward(y_pred, y_true);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, -std::log(1e-7), 1e-5);
}

TEST(CategoricalCrossEntropyTest, StandardPrediction) {
    kmlcpplib::CategoricalCrossEntropy loss_layer(3);

    // Case 1: 3 classes, one-hot target
    Eigen::VectorXd y_pred(3);
    y_pred << 0.7, 0.2, 0.1;
    Eigen::VectorXd y_true(3);
    y_true << 1.0, 0.0, 0.0;

    double loss = loss_layer.forward(y_pred, y_true);
    // Expected loss = -log(0.7) ≈ 0.35667494
    EXPECT_NEAR(loss, -std::log(0.7), 1e-6);

    Eigen::VectorXd grad = loss_layer.backward();
    EXPECT_EQ(grad.rows(), 3);
    // Expected grad = [-1/0.7, 0, 0] = [-1.4285714, 0.0, 0.0]
    EXPECT_NEAR(grad(0), -1.0 / 0.7, 1e-6);
    EXPECT_NEAR(grad(1), 0.0, 1e-6);
    EXPECT_NEAR(grad(2), 0.0, 1e-6);

    // Case 2: 4 classes, non-one-hot soft targets
    kmlcpplib::CategoricalCrossEntropy loss_layer4(4);
    Eigen::VectorXd y_pred4(4);
    y_pred4 << 0.1, 0.5, 0.3, 0.1;
    Eigen::VectorXd y_true4(4);
    y_true4 << 0.0, 0.8, 0.2, 0.0;

    loss = loss_layer4.forward(y_pred4, y_true4);
    // Expected loss = -(0.8 * log(0.5) + 0.2 * log(0.3))
    double expected_loss = -(0.8 * std::log(0.5) + 0.2 * std::log(0.3));
    EXPECT_NEAR(loss, expected_loss, 1e-6);

    grad = loss_layer4.backward();
    EXPECT_EQ(grad.rows(), 4);
    EXPECT_NEAR(grad(0), 0.0, 1e-6);
    EXPECT_NEAR(grad(1), -0.8 / 0.5, 1e-6);
    EXPECT_NEAR(grad(2), -0.2 / 0.3, 1e-6);
    EXPECT_NEAR(grad(3), 0.0, 1e-6);
}

TEST(CategoricalCrossEntropyTest, ExplosionZeroPredOneTrue) {
    kmlcpplib::CategoricalCrossEntropy loss_layer(3);

    // y_pred = [0.0, 0.5, 0.5], y_true = [1.0, 0.0, 0.0] (log(0) explosion)
    Eigen::VectorXd y_pred(3);
    y_pred << 0.0, 0.5, 0.5;
    Eigen::VectorXd y_true(3);
    y_true << 1.0, 0.0, 0.0;

    double loss = loss_layer.forward(y_pred, y_true);

    // Check that loss is finite and not NaN/Inf
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_FALSE(std::isnan(loss));

    // Clamped y_pred(0) is eps = 1e-7, loss should be -log(1e-7) ≈ 16.11809565
    double expected_loss = -std::log(1e-7);
    EXPECT_NEAR(loss, expected_loss, 1e-5);

    Eigen::VectorXd grad = loss_layer.backward();
    EXPECT_EQ(grad.rows(), 3);
    EXPECT_TRUE(std::isfinite(grad(0)));
    EXPECT_FALSE(std::isnan(grad(0)));

    // Clamped grad(0) should be -1.0 / 1e-7 = -1e7
    double expected_grad0 = -1.0 / 1e-7;
    EXPECT_NEAR(grad(0), expected_grad0, 1.0);
    EXPECT_NEAR(grad(1), 0.0, 1e-6);
    EXPECT_NEAR(grad(2), 0.0, 1e-6);
}

TEST(CategoricalCrossEntropyTest, PerfectPredictions) {
    kmlcpplib::CategoricalCrossEntropy loss_layer(3);

    // y_pred = [0.0, 1.0, 0.0], y_true = [0.0, 1.0, 0.0]
    Eigen::VectorXd y_pred(3);
    y_pred << 0.0, 1.0, 0.0;
    Eigen::VectorXd y_true(3);
    y_true << 0.0, 1.0, 0.0;

    double loss = loss_layer.forward(y_pred, y_true);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, 0.0, 1e-5);

    Eigen::VectorXd grad = loss_layer.backward();
    EXPECT_EQ(grad.rows(), 3);
    EXPECT_TRUE(std::isfinite(grad(1)));
    EXPECT_NEAR(grad(0), 0.0, 1e-5);
    EXPECT_NEAR(grad(1), -1.0, 1e-5);
    EXPECT_NEAR(grad(2), 0.0, 1e-5);
}

TEST(CategoricalCrossEntropyTest, ExtremeOutOfBoundsPred) {
    kmlcpplib::CategoricalCrossEntropy loss_layer(3);

    // Negative y_pred (e.g. [-10.0, 0.5, 0.5]) should be clamped to eps
    Eigen::VectorXd y_pred(3);
    y_pred << -10.0, 0.5, 0.5;
    Eigen::VectorXd y_true(3);
    y_true << 1.0, 0.0, 0.0;

    double loss = loss_layer.forward(y_pred, y_true);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, -std::log(1e-7), 1e-5);

    Eigen::VectorXd grad = loss_layer.backward();
    EXPECT_EQ(grad.rows(), 3);
    EXPECT_NEAR(grad(0), -1.0 / 1e-7, 1.0);
}


