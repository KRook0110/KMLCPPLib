#pragma once

#include <Eigen/Dense>

#include <memory>

namespace kmlcpplib {

class ILayer {
   public:
    virtual ~ILayer() {};
    virtual Eigen::VectorXd forward(Eigen::VectorXd input) = 0;
    virtual Eigen::VectorXd backward(Eigen::VectorXd upstream_grad) = 0;
};



}  // namespace kmlcpplib
