
#pragma once

#include "KMLCPPLib/LossFunctions/LossFunctionBase.hpp"
#include <Eigen/Dense>

namespace kmlcpplib {
class MSELossLayer : public LossFunctionBase {
    struct {
        Eigen::VectorXd y_pred;
        Eigen::VectorXd y_true;
    } cache;
  public:
    using LossFunctionBase::LossFunctionBase;

    Eigen::VectorXd forward(const Eigen::VectorXd &y_pred,
                            const Eigen::VectorXd &y_true) override;
    Eigen::VectorXd backward() override;
};
} // namespace kmlcpplib
