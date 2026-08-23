#pragma once

#include "KMLCPPLib/LossFunctions/LossFunctionBase.hpp"
#include <Eigen/Dense>

namespace kmlcpplib {
class BinaryCrossEntropy : public LossFunctionBase {
    // clamping epsilon to prevent explosion
    static constexpr double eps = 1e-7;

    struct {
        double y_pred;
        double y_true;
    }cache;

    public:
    // binary cross entropy can only have one output
    BinaryCrossEntropy() : LossFunctionBase(1) { }

    double forward(const Eigen::VectorXd &y_pred, const Eigen::VectorXd &y_true) override;
    Eigen::VectorXd backward() override;
};
} // namespace kmlcpplib
