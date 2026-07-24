#include <KMLCPPLib/Layers/SoftmaxLayer.hpp>

namespace kmlcpplib {


SoftmaxLayer::SoftmaxLayer(uint32_t in_nodes)
    : LayerBase(in_nodes) {}

SoftmaxLayer::~SoftmaxLayer() = default;

SoftmaxLayer::SoftmaxLayer(SoftmaxLayer&& other) noexcept = default;

SoftmaxLayer& SoftmaxLayer::operator=(SoftmaxLayer&& other) noexcept = default;

Eigen::VectorXd SoftmaxLayer::forward(Eigen::VectorXd x) {
    double maxCoeff = x.maxCoeff();
    Eigen::VectorXd exps = (x.array() - maxCoeff).exp();
    return cache.forward = exps / exps.sum();
}

Eigen::VectorXd SoftmaxLayer::backward(Eigen::VectorXd upstream_grad) {

    Eigen::MatrixXd res = cache.forward.asDiagonal() * (
            Eigen::MatrixXd::Identity(in_nodes, in_nodes) - cache.forward.transpose()
            );


    return res * upstream_grad;
}

}  // namespace kmlcpplib
