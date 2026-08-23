#include <KMLCPPLib/LossFunctions/MeanSquaredError.hpp>

namespace kmlcpplib {


double MeanSquaredError::forward(const Eigen::VectorXd& y_pred,
                                           const Eigen::VectorXd& y_true) {
    assert(y_pred.rows() == in_nodes);
    cache.y_pred = y_pred;
    cache.y_true = y_true;
    Eigen::VectorXd del_y = y_true - y_pred;
    double scalar_loss = del_y.squaredNorm() / in_nodes;

    return  scalar_loss;
}

Eigen::VectorXd MeanSquaredError::backward() {
    Eigen::VectorXd del_y = cache.y_pred - cache.y_true;
    return 2.0 * (del_y) / in_nodes;
}

}  // namespace kmlcpplib
