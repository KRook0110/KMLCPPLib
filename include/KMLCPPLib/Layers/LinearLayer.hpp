#pragma once

#include <KMLCPPLib/Layers/ILayer.hpp>
#include <KMLCPPLib/Parameter.hpp>

namespace kmlcpplib {

class LinearLayer : public ILayer {
    uint32_t in_nodes;
    uint32_t out_nodes;
    std::shared_ptr<Parameter> w;
    std::shared_ptr<Parameter> b;

    struct {
        Eigen::VectorXd x;
    } cache;

  public:
    explicit LinearLayer(uint32_t in_nodes, uint32_t out_nodes);
    ~LinearLayer() override;

    // Move operations
    LinearLayer(LinearLayer &&other) noexcept;
    LinearLayer &operator=(LinearLayer &&other) noexcept;

    // Disable copy (unique_ptr is non-copyable)
    LinearLayer(const LinearLayer &) = delete;
    LinearLayer &operator=(const LinearLayer &) = delete;

    Eigen::VectorXd forward(Eigen::VectorXd input) override;
    Eigen::VectorXd backward(Eigen::VectorXd upstream_grad) override;

    [[nodiscard]] virtual std::vector<std::shared_ptr<Parameter>>
    get_params() const override {
        return {w, b};
    }
};

} // namespace kmlcpplib
