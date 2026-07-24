#include <KMLCPPLib/Layers/SigmoidLayer.hpp>

namespace kmlcpplib {

struct SigmoidLayer::Impl {
    uint32_t in_nodes;

    struct {
        Eigen::VectorXd forward;
    } cache;

    Impl(uint32_t in_nodes) : in_nodes(in_nodes) {};
};

SigmoidLayer::SigmoidLayer(uint32_t in_nodes)
    : pImpl(std::make_unique<Impl>(in_nodes)) {}

SigmoidLayer::~SigmoidLayer() = default;

SigmoidLayer::SigmoidLayer(SigmoidLayer&& other) noexcept = default;

SigmoidLayer& SigmoidLayer::operator=(SigmoidLayer&& other) noexcept = default;

Eigen::VectorXd SigmoidLayer::forward_feed(Eigen::VectorXd x, Eigen::VectorXd) {
    assert(x.rows() == pImpl->in_nodes);

    pImpl->cache.forward = 1.0 / (1.0 + (-x).array().exp());

    assert(pImpl->cache.forward.rows() == pImpl->in_nodes);

    return pImpl->cache.forward;
}

Eigen::VectorXd SigmoidLayer::backward_prop(Eigen::MatrixXd past,
                                            float learning_rate) {
    assert(pImpl->cache.forward.rows() == pImpl->in_nodes);

    return (pImpl->cache.forward).array() *
           (1.0 - pImpl->cache.forward.array()) *
           past.array();
}

void SigmoidLayer::update() {}

}  // namespace kmlcpplib
