#include <Eigen/Dense>
#include <iostream>

int main() {

    Eigen::VectorXd x(5);

    x.setOnes();
    double a = 2.4;

    std::cout << x * a << std::endl;
    std::cout << "---\n\n";
    std::cout << a * x << std::endl;
}
