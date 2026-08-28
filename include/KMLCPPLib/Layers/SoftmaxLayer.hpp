#pragma once

#include <KMLCPPLib/Layers/LayerBase.hpp>


namespace kmlcpplib {

class SoftmaxLayer : public LayerBase {
    double inverse_temperature;
    struct {
        Eigen::VectorXd forward;
    } cache;
   public:
    // sigmoid function will have the same output nodes as its inputs
    explicit SoftmaxLayer(uint32_t in_nodes) : LayerBase(in_nodes), inverse_temperature(1.0){}
    explicit SoftmaxLayer(uint32_t in_nodes, double inverse_temperature) : LayerBase(in_nodes), inverse_temperature(inverse_temperature) {}

    Eigen::VectorXd forward(const Eigen::VectorXd& input) override;
    Eigen::VectorXd backward(const Eigen::VectorXd& upstream_grad) override;

    void setInverseTemperature(double inverseTemperature) { this-> inverse_temperature = inverseTemperature; };
    void setTemperature(double temperature) { inverse_temperature = 1.0 / temperature; };

    double getTemperature() { return 1.0 / inverse_temperature; };
    double getInverseTemperature() { return inverse_temperature; };
};

}
