
#include <KMLCPPLib/Layers/LinearLayer.hpp>

namespace kmlcpplib {

struct LinearLayer::Impl {
    Eigen::MatrixXd w;
    Eigen::VectorXd b;
    uint32_t in_nodes;
    uint32_t out_nodes;

    struct Cache {
        Eigen::VectorXd x;
        Eigen::MatrixXd w;
        Eigen::VectorXd b;

        Eigen::MatrixXd w_backprop;
        Eigen::VectorXd b_backprop;

        // result
        Eigen::VectorXd forward;

        Cache(uint32_t in_nodes, uint32_t out_nodes)
            : x(in_nodes), w(in_nodes, out_nodes), b(out_nodes),
              forward(out_nodes), w_backprop(in_nodes, out_nodes),
              b_backprop(out_nodes) {}

    } cache;

    Impl(uint32_t in_nodes, uint32_t out_nodes)
        : w(in_nodes, out_nodes), b(out_nodes), cache(in_nodes, out_nodes),
          in_nodes(in_nodes), out_nodes(out_nodes) {};
};

LinearLayer::LinearLayer(uint32_t in_nodes, uint32_t out_nodes)
    : pImpl(std::make_unique<Impl>(in_nodes, out_nodes)) {}

LinearLayer::~LinearLayer() = default;

LinearLayer::LinearLayer(LinearLayer &&other) noexcept = default;

LinearLayer &LinearLayer::operator=(LinearLayer &&other) noexcept = default;

Eigen::VectorXd LinearLayer::forward_feed(Eigen::VectorXd x, Eigen::VectorXd) {

    assert(x.rows() == pImpl->in_nodes);

    pImpl->cache.x = x;
    return pImpl->cache.forward = pImpl->w.transpose() * x + pImpl->b;
}

Eigen::VectorXd LinearLayer::backward_prop(Eigen::MatrixXd past,
                                           float learning_rate) {
    assert(past.rows() == pImpl->out_nodes);
    assert(past.cols() == 1);

    pImpl->cache.w_backprop =
        pImpl->w - learning_rate * (pImpl->cache.x * past.transpose());
    pImpl->cache.b_backprop = pImpl->b - learning_rate * past;
    return pImpl->w * past;
}

void LinearLayer::update() {
    pImpl->w = pImpl->cache.w_backprop;
    pImpl->b = pImpl->cache.b_backprop;
}

} // namespace kmlcpplib
