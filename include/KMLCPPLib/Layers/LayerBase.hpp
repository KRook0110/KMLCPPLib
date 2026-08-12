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
    virtual ~LayerBase() = default;

    // Disable copy operations to prevent slicing
    LayerBase(const LayerBase&) = delete;
    LayerBase& operator=(const LayerBase&) = delete;

    // Disable move operations (polymorphic copy/move is generally unsafe)
    LayerBase(LayerBase&&) = delete;
    LayerBase& operator=(LayerBase&&) = delete;
    virtual Eigen::VectorXd forward(const Eigen::VectorXd& input) = 0;
    virtual Eigen::VectorXd backward(const Eigen::VectorXd& upstream_grad) = 0;

    [[nodiscard]] virtual std::vector<std::shared_ptr<Parameter>>
    get_params() const {
        return {};
    };

    uint32_t get_in_nodes() {
        return in_nodes;
    }

    uint32_t get_out_nodes() {
        return out_nodes;
    }
};

} // namespace kmlcpplib
