#include <algorithm>
#include <vector>

#include "KMLCPPLib/Parameter.hpp"
#include "KMLCPPLib/WeightInitializers/RandomizedWeight.hpp"

namespace kmlcpplib {

void randomize_weights(
    const std::vector<std::shared_ptr<kmlcpplib::Parameter>> &weights) {
    for (auto weight : weights) {
        randomize_weights(weight);
    }
}

void randomize_weights(std::shared_ptr<kmlcpplib::Parameter> &weight) {
    randomize_weights(*weight.get());
}

void randomize_weights(kmlcpplib::Parameter &weight) {
    weight.value = weight.value.setRandom();
}

} // namespace kmlcpplib
