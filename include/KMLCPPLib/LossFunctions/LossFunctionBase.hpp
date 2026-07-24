#pragma once

#include <Eigen/Dense>

namespace kmlcpplib {
    class LossFunctionBase {
        protected:
            uint32_t in_nodes;
        public:
            explicit LossFunctionBase(uint32_t result_dimensions) : in_nodes(result_dimensions) {}
            virtual ~LossFunctionBase() {};
            virtual Eigen::VectorXd forward(const Eigen::VectorXd& y_pred, const Eigen::VectorXd& y_true);
            virtual Eigen::VectorXd backward();
    };
}

