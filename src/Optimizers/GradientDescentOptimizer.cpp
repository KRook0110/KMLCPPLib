#include <memory>

#include "KMLCPPLib/Optimizers/GradientDescentOptimizer.hpp"
#include "KMLCPPLib/Parameter.hpp"

namespace kmlcpplib {
    void GradientDescentOptimizer::step() {
        for(std::shared_ptr<Parameter> &param: params) {
            param->value -= param->gradient * learning_rate;
        }
    }
}


