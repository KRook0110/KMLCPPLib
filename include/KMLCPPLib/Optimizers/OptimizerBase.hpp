#pragma once

#include "KMLCPPLib/Layers/LayerBase.hpp"

namespace kmlcpplib {
class OptimizerBase {
  protected:
    double learning_rate;
    std::vector<std::shared_ptr<Parameter>> params;

  public:
    virtual ~OptimizerBase() {}

    virtual void step() = 0;

    virtual void set_learning_rate(double learning_rate) {
        this->learning_rate = learning_rate;
    }

    virtual double get_learning_rate() {
        return learning_rate;
    }

    virtual std::vector<std::shared_ptr<Parameter>> get_params() {
        return params;
    }

    virtual void set_params(std::vector<std::shared_ptr<Parameter>> new_params) {
        params = new_params;
    }
};
} // namespace kmlcpplib
