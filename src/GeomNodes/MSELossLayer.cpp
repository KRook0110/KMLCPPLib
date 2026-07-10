#include <KMLCPPLib/Layer.hpp>

namespace kmlcpplib {

struct MSELossLayer::Impl {
    uint32_t in_nodes;

    struct {
        Eigen::VectorXd y_pred;
        Eigen::VectorXd y_true;
    } cache;

    Impl(uint32_t in_nodes) : in_nodes(in_nodes) {};
};

MSELossLayer::MSELossLayer(uint32_t in_nodes)
    : pImpl(std::make_unique<Impl>(in_nodes)) {}

MSELossLayer::~MSELossLayer() = default;

MSELossLayer::MSELossLayer(MSELossLayer&& other) noexcept = default;

MSELossLayer& MSELossLayer::operator=(MSELossLayer&& other) noexcept = default;

Eigen::VectorXd MSELossLayer::forward_feed(Eigen::VectorXd y_pred,
                                           Eigen::VectorXd y_true) {
    assert(y_pred.rows() == pImpl->in_nodes);
    pImpl->cache.y_pred = y_pred;
    pImpl->cache.y_true = y_true;
    Eigen::VectorXd del_y = y_true - y_pred;
    double scalar_loss = del_y.squaredNorm() / pImpl->in_nodes;

    return Eigen::VectorXd::Constant(1, scalar_loss);
}

Eigen::VectorXd MSELossLayer::backward_prop(Eigen::MatrixXd past,
                                            float learning_rate) {
    Eigen::VectorXd del_y = pImpl->cache.y_pred - pImpl->cache.y_true;
    return 2.0 * (del_y) / pImpl->in_nodes;
}

void MSELossLayer::update() {}

}  // namespace kmlcpplib
