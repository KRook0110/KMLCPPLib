#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>
#include <random>

#include "KMLCPPLib/Layers/LinearLayer.hpp"
#include "KMLCPPLib/Layers/SigmoidLayer.hpp"
#include "KMLCPPLib/Layers/SequenceLayer.hpp"
#include "KMLCPPLib/LossFunctions/MSELossFunction.hpp"
#include "KMLCPPLib/WeightInitializors/RandomizedWeight.hpp"

int main() {
    Eigen::VectorXd x(5);

    x.setOnes();
    double a = 2.4;

    std::cout << x * a << std::endl;
    std::cout << "---\n\n";
    std::cout << a * x << std::endl;
}
