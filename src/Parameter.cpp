#include <KMLCPPLib/Parameter.hpp>

namespace kmlcpplib {

Parameter::Parameter(Eigen::Index rows, Eigen::Index cols)
    : value(Eigen::MatrixXd::Zero(rows, cols)),
      gradient(Eigen::MatrixXd::Zero(rows, cols)) {}

} // namespace kmlcpplib

