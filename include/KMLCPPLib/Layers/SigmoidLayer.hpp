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
    Eigen::VectorXd forward(const Eigen::VectorXd& input) override;
    Eigen::VectorXd backward(const Eigen::VectorXd& upstream_grad) override;
};

} // namespace kmlcpplib
