

#include <Eigen/Dense>
#include "KMLCPPLib/LossFunctions/CategoricalCrossEntropy.hpp"


namespace kmlcpplib {
double CategoricalCrossEntropy::forward(const Eigen::VectorXd &y_pred,
                                   const Eigen::VectorXd &y_true) {
    assert(y_pred.rows() == in_nodes);
    assert(y_true.rows() == in_nodes);

    cache.y_pred = y_pred.cwiseMax(eps);
    cache.y_true = y_true;
    return -(cache.y_pred.array().log() * cache.y_true.array()).sum();
}

Eigen::VectorXd CategoricalCrossEntropy::backward() {
    return -cache.y_true.cwiseQuotient(cache.y_pred);
}
}
