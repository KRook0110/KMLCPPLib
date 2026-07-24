#pragma once

#include "KMLCPPLib/Params.hpp"
#include <Eigen/Dense>

namespace kmlcpplib {
    class ILossFunction {
        public:
            virtual ~ILossFunction() {};
            virtual Eigen::VectorXd forward(const Eigen::VectorXd& y_pred, const Eigen::VectorXd& y_true);
            virtual Eigen::VectorXd backward();
            virtual std::vector<std::shared_ptr<Parameter>> get_params();
    };
}

