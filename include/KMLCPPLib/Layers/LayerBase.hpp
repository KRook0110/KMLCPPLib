#pragma once

#include "KMLCPPLib/Parameter.hpp"
#include <Eigen/Dense>

#include <memory>

namespace kmlcpplib {
class LayerBase {
  protected:
    uint32_t in_nodes, out_nodes;

  public:
    explicit LayerBase(uint32_t in_nodes)
        : in_nodes(in_nodes), out_nodes(in_nodes) {}
    explicit LayerBase(uint32_t in_nodes, uint32_t out_nodes)
        : in_nodes(in_nodes), out_nodes(out_nodes) {}
    virtual ~LayerBase() {};
    virtual Eigen::VectorXd forward(const Eigen::VectorXd& input) = 0;
    virtual Eigen::VectorXd backward(const Eigen::VectorXd& upstream_grad) = 0;

    [[nodiscard]] virtual std::vector<std::shared_ptr<Parameter>>
    get_params() const {
        return {};
    };
};

} // namespace kmlcpplib
