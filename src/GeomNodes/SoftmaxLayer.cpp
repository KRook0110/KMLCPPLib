#include <KMLCPPLib/Layers/SoftmaxLayer.hpp>

namespace kmlcpplib {

struct SoftmaxLayer::Impl {
    uint32_t in_nodes;


    struct {
        Eigen::VectorXd forward;
        Eigen::MatrixXd backprop;
    } cache;

    Impl(uint32_t in_nodes) : in_nodes(in_nodes) {};
};

SoftmaxLayer::SoftmaxLayer(uint32_t in_nodes)
    : pImpl(std::make_unique<Impl>(in_nodes)) {}

SoftmaxLayer::~SoftmaxLayer() = default;

SoftmaxLayer::SoftmaxLayer(SoftmaxLayer&& other) noexcept = default;

SoftmaxLayer& SoftmaxLayer::operator=(SoftmaxLayer&& other) noexcept = default;

Eigen::VectorXd SoftmaxLayer::forward_feed(Eigen::VectorXd x, Eigen::VectorXd) {
    double maxCoeff = x.maxCoeff();
    Eigen::VectorXd exps = (x.array() - maxCoeff).exp();
    return pImpl->cache.forward = exps / exps.sum();
}

Eigen::VectorXd SoftmaxLayer::backward_prop(Eigen::MatrixXd past,
                                            float learning_rate) {

    Eigen::MatrixXd res = pImpl->cache.forward.asDiagonal() * (
            Eigen::MatrixXd::Identity(pImpl->in_nodes, pImpl->in_nodes) - pImpl->cache.forward.transpose()
            );


    return pImpl->cache.backprop = res * past;
}

void SoftmaxLayer::update() {}

}  // namespace kmlcpplib
