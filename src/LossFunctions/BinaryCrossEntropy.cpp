#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

#include "KMLCPPLib/LossFunctions/BinaryCrossEntropy.hpp"

namespace kmlcpplib {

double BinaryCrossEntropy::forward(const Eigen::VectorXd &y_pred,
                                   const Eigen::VectorXd &y_true) {
    // in_nodes will always be one
    assert(y_pred.rows() == in_nodes);


    double yp = std::clamp(y_pred(0), eps,  1.0 - eps) ;
    double yt = y_true(0);
 cache.y_pred = yp;
 cache.y_true = yt;

    return -(yt * std::log(yp) + (1.0 - yt) * std::log( 1.0 - yp));
}

Eigen::VectorXd BinaryCrossEntropy::backward() {
    double yp = cache.y_pred;
    double yt = cache.y_true;

    return Eigen::Vector<double, 1>{ (yp - yt) / (yp * (1.0 - yp)) };
}

} // namespace kmlcpplib
