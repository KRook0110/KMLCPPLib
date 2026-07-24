#pragma once

#include <Eigen/Dense>

namespace kmlcpplib {

class Parameter {
    public:
        Eigen::MatrixXd value;
        Eigen::MatrixXd gradient;
};


}
