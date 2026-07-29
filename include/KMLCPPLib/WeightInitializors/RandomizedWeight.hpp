#pragma once

#include "KMLCPPLib/Parameter.hpp"

#include <vector>

namespace kmlcpplib {
void randomize_weights(
    const std::vector<std::shared_ptr<kmlcpplib::Parameter>> &weights);
void randomize_weights(std::shared_ptr<kmlcpplib::Parameter> &weights);
void randomize_weights(kmlcpplib::Parameter &weight);
} // namespace kmlcpplib
