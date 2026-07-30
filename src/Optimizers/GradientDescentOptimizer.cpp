#include "KMLCPPLib/Optimizers/GradientDescentOptimizer.hpp"
#include "KMLCPPLib/Parameter.hpp"

namespace kmlcpplib {
    void GradientDescentOptimizer::step() {
        for(auto &param: params) {
            param->value -= param->gradient * learning_rate;
        }
    }
}


