#pragma once

#include <KMLCPPLib/Layers/LayerBase.hpp>
#include <KMLCPPLib/Parameter.hpp>

namespace kmlcpplib {

class LinearLayer : public LayerBase {
    std::shared_ptr<Parameter> w;
    std::shared_ptr<Parameter> b;

    struct {
        Eigen::VectorXd x;
    } cache;

  public:
    explicit LinearLayer(uint32_t in_nodes, uint32_t out_nodes);
    Eigen::VectorXd forward(const Eigen::VectorXd& input) override;
    Eigen::VectorXd backward(const Eigen::VectorXd& upstream_grad) override;

    [[nodiscard]] virtual std::vector<std::shared_ptr<Parameter>>
    get_params() const override {
        return {w, b};
    }
};

} // namespace kmlcpplib
