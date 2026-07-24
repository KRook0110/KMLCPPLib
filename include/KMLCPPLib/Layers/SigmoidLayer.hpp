#pragma once

#include <KMLCPPLib/Layers/LayerBase.hpp>

namespace kmlcpplib {

class SigmoidLayer : public LayerBase {
    private:
        struct {
            Eigen::VectorXd forward;

        } cache;
  public:
    // sigmoid function will have the same output nodes as its inputs
    explicit SigmoidLayer(uint32_t in_nodes);
    ~SigmoidLayer() override;

    // Move operations
    SigmoidLayer(SigmoidLayer &&other) noexcept;
    SigmoidLayer &operator=(SigmoidLayer &&other) noexcept;

    // Disable copy (unique_ptr is non-copyable)
    SigmoidLayer(const SigmoidLayer &) = delete;
    SigmoidLayer &operator=(const SigmoidLayer &) = delete;

    Eigen::VectorXd forward(const Eigen::VectorXd& input) override;
    Eigen::VectorXd backward(const Eigen::VectorXd& upstream_grad) override;
};

} // namespace kmlcpplib
