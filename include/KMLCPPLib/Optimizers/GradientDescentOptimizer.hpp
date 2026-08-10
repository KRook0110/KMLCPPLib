#pragma once

#include "KMLCPPLib/Optimizers/OptimizerBase.hpp"

namespace kmlcpplib {
class GradientDescentOptimizer : public OptimizerBase {
  private:

  public:
    GradientDescentOptimizer() = default;
    virtual void step() override;
};
} // namespace kmlcpplib
