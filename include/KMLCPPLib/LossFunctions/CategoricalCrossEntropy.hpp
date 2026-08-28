#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

#include "KMLCPPLib/LossFunctions/LossFunctionBase.hpp"

namespace kmlcpplib {
class CategoricalCrossEntropy : public LossFunctionBase {

    static constexpr double eps = 1e-7;

    struct {
        Eigen::VectorXd y_pred;
        Eigen::VectorXd y_true;

    } cache;

  public:
    using LossFunctionBase::LossFunctionBase;

    double forward(const Eigen::VectorXd &y_pred,
                   const Eigen::VectorXd &y_true) override;
    Eigen::VectorXd backward() override;
};

} // namespace kmlcpplib
