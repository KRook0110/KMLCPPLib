#include <KMLCPPLib/Layers/SigmoidLayer.hpp>

namespace kmlcpplib {

SigmoidLayer::SigmoidLayer(uint32_t in_nodes)
    : LayerBase(in_nodes) {}

SigmoidLayer::~SigmoidLayer() = default;

SigmoidLayer::SigmoidLayer(SigmoidLayer&& other) noexcept = default;

SigmoidLayer& SigmoidLayer::operator=(SigmoidLayer&& other) noexcept = default;

Eigen::VectorXd SigmoidLayer::forward(const Eigen::VectorXd& x) {
    assert(x.rows() == in_nodes);

    cache.forward = 1.0 / (1.0 + (-x).array().exp());

    assert(cache.forward.rows() == in_nodes);

    return cache.forward;
}

Eigen::VectorXd SigmoidLayer::backward(const Eigen::VectorXd& upstream_grad) {
    assert(cache.forward.rows() == in_nodes);

    return (cache.forward).array() *
           (1.0 - cache.forward.array()) *
           upstream_grad.array();
}
}  // namespace kmlcpplib
