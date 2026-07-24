#pragma once

#include <Eigen/Dense>


namespace kmlcpplib {

class Parameter {
  public:
    Eigen::MatrixXd value;
    Eigen::MatrixXd gradient;

    Parameter() = default;
    Parameter(Eigen::Index rows, Eigen::Index cols);

    [[nodiscard]] Eigen::Index rows() const { return value.rows(); }
    [[nodiscard]] Eigen::Index cols() const { return value.cols(); }
    void zero_grad() { gradient.setZero(); }
};

} // namespace kmlcpplib

