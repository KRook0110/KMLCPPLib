#pragma once

#include <KMLCPPLib/Layers/ILayer.hpp>

namespace kmlcpplib {

class LinearLayer : public ILayer {
    struct Impl;
    std::unique_ptr<Impl> pImpl;

  public:
    explicit LinearLayer(uint32_t in_nodes, uint32_t out_nodes);
    ~LinearLayer() override;

    // Move operations
    LinearLayer(LinearLayer &&other) noexcept;
    LinearLayer &operator=(LinearLayer &&other) noexcept;

    // Disable copy (unique_ptr is non-copyable)
    LinearLayer(const LinearLayer &) = delete;
    LinearLayer &operator=(const LinearLayer &) = delete;

    Eigen::VectorXd forward(Eigen::VectorXd) override;
    Eigen::VectorXd backward(Eigen::VectorXd upstream_grad) override;
};

} // namespace kmlcpplib
