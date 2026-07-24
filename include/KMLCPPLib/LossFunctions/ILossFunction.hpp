#pragma once

#include <Eigen/Dense>

namespace kmlcpplib {
    class ILossFunction {
        public:
            virtual ~ILossFunction() {};
            virtual Eigen::VectorXd forward(const Eigen::VectorXd& y_pred, const Eigen::VectorXd& y_true);
            virtual Eigen::VectorXd backward();
    };
}

