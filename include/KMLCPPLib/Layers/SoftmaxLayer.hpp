#pragma once

#include <KMLCPPLib/Layers/LayerBase.hpp>


namespace kmlcpplib {

class SoftmaxLayer : public LayerBase {
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

    Eigen::VectorXd forward(Eigen::VectorXd input) override;
    Eigen::VectorXd backward(Eigen::VectorXd upstream_grad) override;

};

}
