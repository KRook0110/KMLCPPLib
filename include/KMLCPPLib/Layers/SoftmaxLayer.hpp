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
    Eigen::VectorXd forward(const Eigen::VectorXd& input) override;
    Eigen::VectorXd backward(const Eigen::VectorXd& upstream_grad) override;

};

}
