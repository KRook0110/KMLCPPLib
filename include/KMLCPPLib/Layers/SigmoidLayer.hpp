#pragma once

#include <KMLCPPLib/Layers/ILayer.hpp>

namespace kmlcpplib {

class SigmoidLayer : public ILayer {
    struct Impl;
    std::unique_ptr<Impl> pImpl;

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

    Eigen::VectorXd forward(Eigen::VectorXd input) override;
    Eigen::VectorXd backward(Eigen::VectorXd upstream_grad) override;
};

} // namespace kmlcpplib
