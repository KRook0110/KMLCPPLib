#include <KMLCPPLib/Layers/SoftmaxLayer.hpp>

namespace kmlcpplib {


SoftmaxLayer::SoftmaxLayer(uint32_t in_nodes)
    : LayerBase(in_nodes) {}

Eigen::VectorXd SoftmaxLayer::forward(const Eigen::VectorXd& x) {
    double maxCoeff = x.maxCoeff();
    Eigen::VectorXd exps = (x.array() - maxCoeff).exp();
    return cache.forward = exps / exps.sum();
}

Eigen::VectorXd SoftmaxLayer::backward(const Eigen::VectorXd& upstream_grad) {
    Eigen::MatrixXd res = Eigen::MatrixXd(cache.forward.asDiagonal()) - (cache.forward * cache.forward.transpose());
    return  res * upstream_grad;
}

}  // namespace kmlcpplib
