#pragma once

#include <Eigen/Dense>

#include <memory>

namespace kmlcpplib {

class ILayer {
   public:
    virtual ~ILayer() {};
    virtual Eigen::VectorXd forward(Eigen::VectorXd input) = 0;
    virtual Eigen::VectorXd backward(Eigen::VectorXd upstream_grad) = 0;
};




class SigmoidLayer : public ILayer {
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

    Eigen::VectorXd forward_feed(Eigen::VectorXd in, Eigen::VectorXd =Eigen::VectorXd()) override;
    Eigen::VectorXd backward_prop(Eigen::MatrixXd past, float learning_rate) override;
    void update() override;
};

class SoftmaxLayer : public ILayer {
    struct Impl;
    std::unique_ptr<Impl> pImpl;

   public:
    // sigmoid function will have the same output nodes as its inputs
    explicit SoftmaxLayer(uint32_t in_nodes);
    ~SoftmaxLayer() override;

    // Move operations
    SoftmaxLayer(SoftmaxLayer&& other) noexcept;
    SoftmaxLayer& operator=(SoftmaxLayer&& other) noexcept;

    // Disable copy (unique_ptr is non-copyable)
    SoftmaxLayer(const SoftmaxLayer&) = delete;
    SoftmaxLayer& operator=(const SoftmaxLayer&) = delete;

    Eigen::VectorXd forward_feed(Eigen::VectorXd y_pred, Eigen::VectorXd y_true =Eigen::VectorXd()) override;
    Eigen::VectorXd backward_prop(Eigen::MatrixXd past, float learning_rate) override;
    void update() override;

};

}  // namespace kmlcpplib
