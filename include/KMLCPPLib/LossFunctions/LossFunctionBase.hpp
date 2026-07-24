#pragma once

#include <Eigen/Dense>

namespace kmlcpplib {
    class LossFunctionBase {
        public:
            virtual ~LossFunctionBase() {};
            virtual Eigen::VectorXd forward(const Eigen::VectorXd& y_pred, const Eigen::VectorXd& y_true);
            virtual Eigen::VectorXd backward();
    };
}

