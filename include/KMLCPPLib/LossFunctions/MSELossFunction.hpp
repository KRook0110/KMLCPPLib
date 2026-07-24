
#pragma once

#include "KMLCPPLib/LossFunctions/ILossFunction.hpp"
#include <Eigen/Dense>

namespace kmlcpplib {
class MSELossLayer : public ILossFunction {

    struct Cache {
        Eigen::VectorXd y_pred;
        Eigen::VectorXd y_true;
    } m_cache;

    uint32_t m_in_nodes;

  public:
    explicit MSELossLayer(uint32_t in_nodes);

    Eigen::VectorXd forward(const Eigen::VectorXd& y_pred,const Eigen::VectorXd& y_true) override;
    Eigen::VectorXd backward() override;
};
} // namespace kmlcpplib
