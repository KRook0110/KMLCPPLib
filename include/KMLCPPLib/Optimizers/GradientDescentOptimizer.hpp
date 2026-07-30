#pragma once

#include "KMLCPPLib/Optimizers/OptimizerBase.hpp"

namespace kmlcpplib {
class GradientDescentOptimizer : OptimizerBase {
  private:

  public:
    GradientDescentOptimizer() = default;
    virtual void step() override;
};
} // namespace kmlcpplib
