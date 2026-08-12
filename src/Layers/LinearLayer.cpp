#include <KMLCPPLib/Layers/LinearLayer.hpp>

namespace kmlcpplib {

LinearLayer::LinearLayer(uint32_t in_nodes, uint32_t out_nodes)
    : LayerBase(in_nodes, out_nodes),
      w(std::make_shared<Parameter>(in_nodes, out_nodes)),
      b(std::make_shared<Parameter>(out_nodes, 1)) {}

Eigen::VectorXd LinearLayer::forward(const Eigen::VectorXd& x) {

    assert(x.rows() == in_nodes);

    cache.x = x;
    return w->value.transpose() * x + b->value;
}

Eigen::VectorXd LinearLayer::backward(const Eigen::VectorXd& upstream_grad) {
    assert(upstream_grad.rows() == out_nodes);
    assert(upstream_grad.cols() == 1);

    w->gradient = cache.x * upstream_grad.transpose();
    b->gradient = upstream_grad;
    return w->value * upstream_grad;
}

} // namespace kmlcpplib
