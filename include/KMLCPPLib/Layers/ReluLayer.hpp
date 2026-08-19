#pragma once

#include "KMLCPPLib/Layers/LayerBase.hpp"

namespace kmlcpplib {
class ReluLayer : public LayerBase {
  private:
    struct {
        Eigen::VectorXd forward;
    } cache;

  public:
    explicit ReluLayer(uint32_t in_nodes) : LayerBase(in_nodes) {}
    ~ReluLayer() override = default;
    Eigen::VectorXd forward(const Eigen::VectorXd &input) override;
    Eigen::VectorXd backward(const Eigen::VectorXd &upstream_grad) override;
};
} // namespace kmlcpplib
