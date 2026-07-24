#pragma once

#include "KMLCPPLib/Parameter.hpp"
#include <Eigen/Dense>

#include <memory>

namespace kmlcpplib {
class ILayer {
  protected:
  public:
    virtual ~ILayer() {};
    virtual Eigen::VectorXd forward(Eigen::VectorXd input) = 0;
    virtual Eigen::VectorXd backward(Eigen::VectorXd upstream_grad) = 0;

    [[nodiscard]] virtual std::vector<std::shared_ptr<Parameter>>
    get_params() const {
        return {};
    };
};

} // namespace kmlcpplib
