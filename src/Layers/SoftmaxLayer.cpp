#include <KMLCPPLib/Layers/SoftmaxLayer.hpp>

namespace kmlcpplib {

Eigen::VectorXd SoftmaxLayer::forward(const Eigen::VectorXd& x) {
    Eigen::VectorXd a = x.array() * this->inverse_temperature;
    double maxCoeff = a.maxCoeff();
    Eigen::VectorXd exps = (a.array() - maxCoeff).exp();
    return cache.forward = exps / exps.sum();
}

Eigen::VectorXd SoftmaxLayer::backward(const Eigen::VectorXd& upstream_grad) {
    Eigen::MatrixXd res = this->inverse_temperature * (Eigen::MatrixXd(cache.forward.asDiagonal()) - (cache.forward * cache.forward.transpose()));
    return  res * upstream_grad;
}

}  // namespace kmlcpplib
