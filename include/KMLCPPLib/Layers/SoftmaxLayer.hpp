#pragma once

#include <KMLCPPLib/Layers/LayerBase.hpp>


namespace kmlcpplib {

class SoftmaxLayer : public LayerBase {
    struct {
        Eigen::VectorXd forward;
    } cache;
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

    Eigen::VectorXd forward(const Eigen::VectorXd& input) override;
    Eigen::VectorXd backward(const Eigen::VectorXd& upstream_grad) override;

};

}
