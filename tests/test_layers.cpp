#include <gtest/gtest.h>
#include <KMLCPPLib/Layers/LinearLayer.hpp>
#include <KMLCPPLib/Layers/SigmoidLayer.hpp>
#include <KMLCPPLib/Layers/SoftmaxLayer.hpp>
#include <KMLCPPLib/Layers/SequenceLayer.hpp>
#include <KMLCPPLib/Layers/ReluLayer.hpp>
#include <memory>

TEST(LinearLayerTest, ForwardAndBackward) {
    kmlcpplib::LinearLayer layer(2, 3);
    auto params = layer.get_params();
    ASSERT_EQ(params.size(), 2);

    // Set weights and bias manually
    params[0]->value << 1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0; // 2x3 matrix
    params[1]->value << 0.5, 0.5, 0.5; // 3x1 vector

    Eigen::VectorXd input(2);
    input << 1.0, 2.0;

    // Output should be W^T * x + b
    // W^T = [[1, 4], [2, 5], [3, 6]]
    // W^T * [1, 2]^T = [1*1 + 4*2, 2*1 + 5*2, 3*1 + 6*2]^T = [9, 12, 15]^T
    // + b = [9.5, 12.5, 15.5]^T
    Eigen::VectorXd output = layer.forward(input);
    EXPECT_EQ(output.rows(), 3);
    EXPECT_NEAR(output(0), 9.5, 1e-6);
    EXPECT_NEAR(output(1), 12.5, 1e-6);
    EXPECT_NEAR(output(2), 15.5, 1e-6);

    Eigen::VectorXd upstream_grad(3);
    upstream_grad << 1.0, 1.0, 1.0;
    Eigen::VectorXd dx = layer.backward(upstream_grad);
    EXPECT_EQ(dx.rows(), 2);
}

TEST(SigmoidLayerTest, ForwardAndBackward) {
    kmlcpplib::SigmoidLayer layer(2);
    Eigen::VectorXd input(2);
    input << 0.0, 2.0;

    Eigen::VectorXd output = layer.forward(input);
    EXPECT_EQ(output.rows(), 2);
    EXPECT_NEAR(output(0), 0.5, 1e-6);
    EXPECT_NEAR(output(1), 1.0 / (1.0 + std::exp(-2.0)), 1e-6);

    Eigen::VectorXd upstream_grad(2);
    upstream_grad << 1.0, 1.0;
    Eigen::VectorXd dx = layer.backward(upstream_grad);
    EXPECT_EQ(dx.rows(), 2);
}

TEST(SoftmaxLayerTest, ForwardAndBackward) {
    kmlcpplib::SoftmaxLayer layer(3);
    Eigen::VectorXd input(3);
    input << 1.0, 2.0, 3.0;

    Eigen::VectorXd output = layer.forward(input);
    EXPECT_EQ(output.rows(), 3);
    EXPECT_NEAR(output.sum(), 1.0, 1e-6);
    EXPECT_GT(output(2), output(1));
    EXPECT_GT(output(1), output(0));

    Eigen::VectorXd upstream_grad(3);
    upstream_grad << 0.1, 0.2, 0.7;
    Eigen::VectorXd dx = layer.backward(upstream_grad);
    EXPECT_EQ(dx.rows(), 3);
}

TEST(SequenceLayerTest, GetParams) {
    auto layer1 = std::make_shared<kmlcpplib::LinearLayer>(2, 3);
    auto layer2 = std::make_shared<kmlcpplib::SigmoidLayer>(3);
    auto layer3 = std::make_shared<kmlcpplib::LinearLayer>(3, 1);

    std::vector<std::shared_ptr<kmlcpplib::LayerBase>> layers = {layer1, layer2, layer3};
    kmlcpplib::SequenceLayer seq(layers);

    auto seq_params = seq.get_params();

    // LinearLayer(2,3) has 2 params (weights, bias), SigmoidLayer has 0, LinearLayer(3,1) has 2 params
    auto layer1_params = layer1->get_params();
    auto layer3_params = layer3->get_params();

    ASSERT_EQ(seq_params.size(), 4);
    EXPECT_EQ(seq_params[0], layer1_params[0]);
    EXPECT_EQ(seq_params[1], layer1_params[1]);
    EXPECT_EQ(seq_params[2], layer3_params[0]);
    EXPECT_EQ(seq_params[3], layer3_params[1]);
}

TEST(ReluLayerTest, ForwardAndBackward) {
    kmlcpplib::ReluLayer layer(3);
    Eigen::VectorXd input(3);
    input << -1.0, 0.0, 2.0;

    Eigen::VectorXd output = layer.forward(input);
    EXPECT_EQ(output.rows(), 3);
    EXPECT_NEAR(output(0), 0.0, 1e-6);
    EXPECT_NEAR(output(1), 0.0, 1e-6);
    EXPECT_NEAR(output(2), 2.0, 1e-6);

    Eigen::VectorXd upstream_grad(3);
    upstream_grad << 1.0, 2.0, 3.0;
    Eigen::VectorXd dx = layer.backward(upstream_grad);
    EXPECT_EQ(dx.rows(), 3);
    EXPECT_NEAR(dx(0), 0.0, 1e-6);
    EXPECT_NEAR(dx(1), 0.0, 1e-6);
    EXPECT_NEAR(dx(2), 3.0, 1e-6);
}

