#include "KMLCPPLib/Layers/ReluLayer.hpp"

namespace kmlcpplib {

Eigen::VectorXd ReluLayer::forward(const Eigen::VectorXd &input) {
    assert(input.rows() == in_nodes);
    return cache.forward = input.cwiseMax(0.0);
}

Eigen::VectorXd ReluLayer::backward(const Eigen::VectorXd &upstream_grad) {
    assert(upstream_grad.rows() == in_nodes);
    return (cache.forward.array() > 0.0).select(upstream_grad.array(), 0.0);
}

} // namespace kmlcpplib
