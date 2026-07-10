#pragma once

#include <Eigen/Dense>

#include <memory>

namespace kmlcpplib {

class Layer {
   public:
    virtual ~Layer() = default;
    virtual Eigen::VectorXd forward_feed(Eigen::VectorXd, Eigen::VectorXd = Eigen::VectorXd()) = 0;
    virtual Eigen::VectorXd backward_prop(Eigen::MatrixXd, float) = 0;
    virtual void update() {};
};

class LinearLayer : public Layer {
    struct Impl;
    std::unique_ptr<Impl> pImpl;

   public:
    explicit LinearLayer(uint32_t in_nodes, uint32_t out_nodes);
    ~LinearLayer() override;

    // Move operations
    LinearLayer(LinearLayer&& other) noexcept;
    LinearLayer& operator=(LinearLayer&& other) noexcept;

    // Disable copy (unique_ptr is non-copyable)
    LinearLayer(const LinearLayer&) = delete;
    LinearLayer& operator=(const LinearLayer&) = delete;

    Eigen::VectorXd forward_feed(Eigen::VectorXd, Eigen::VectorXd = Eigen::VectorXd()) override;
    Eigen::VectorXd backward_prop(Eigen::MatrixXd past, float learning_rate) override;
    void update() override;
};

class MSELossLayer : public Layer {
    struct Impl;
    std::unique_ptr<Impl> pImpl;

   public:
    explicit MSELossLayer(uint32_t in_nodes);
    ~MSELossLayer() override;

    // Move operations
    MSELossLayer(MSELossLayer&& other) noexcept;
    MSELossLayer& operator=(MSELossLayer&& other) noexcept;

    // Disable copy (unique_ptr is non-copyable)
    MSELossLayer(const MSELossLayer&) = delete;
    MSELossLayer& operator=(const MSELossLayer&) = delete;

    Eigen::VectorXd forward_feed(Eigen::VectorXd y_pred, Eigen::VectorXd y_true =Eigen::VectorXd()) override;
    Eigen::VectorXd backward_prop(Eigen::MatrixXd past, float learning_rate) override;
    void update() override;
};

class SigmoidLayer : public Layer {
    struct Impl;
    std::unique_ptr<Impl> pImpl;

   public:
    // sigmoid function will have the same output nodes as its inputs
    explicit SigmoidLayer(uint32_t in_nodes);
    ~SigmoidLayer() override;

    // Move operations
    SigmoidLayer(SigmoidLayer&& other) noexcept;
    SigmoidLayer& operator=(SigmoidLayer&& other) noexcept;

    // Disable copy (unique_ptr is non-copyable)
    SigmoidLayer(const SigmoidLayer&) = delete;
    SigmoidLayer& operator=(const SigmoidLayer&) = delete;

    Eigen::VectorXd forward_feed(Eigen::VectorXd y_pred, Eigen::VectorXd y_true =Eigen::VectorXd()) override;
    Eigen::VectorXd backward_prop(Eigen::MatrixXd past, float learning_rate) override;
    void update() override;
};


}  // namespace kmlcpplib
