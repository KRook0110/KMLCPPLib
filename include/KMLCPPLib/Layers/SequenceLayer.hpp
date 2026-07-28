#pragma once

#include <KMLCPPLib/Layers/LayerBase.hpp>

namespace kmlcpplib {

class SequenceLayer : public LayerBase {
  private:
    std::vector<std::shared_ptr<LayerBase>> stored_layers;

  public:
    // sigmoid function will have the same output nodes as its inputs
    explicit SequenceLayer(
        const std::vector<std::shared_ptr<LayerBase>> &stored_layers);
    ~SequenceLayer() override = default;

    // Move operations
    SequenceLayer(SequenceLayer &&other) noexcept = default;
    SequenceLayer &operator=(SequenceLayer &&other) noexcept = default;

    // Disable copy (unique_ptr is non-copyable)
    SequenceLayer(const SequenceLayer &) = delete;
    SequenceLayer &operator=(const SequenceLayer &) = delete;

    Eigen::VectorXd forward(const Eigen::VectorXd &input) override;
    Eigen::VectorXd backward(const Eigen::VectorXd &upstream_grad) override;

    [[nodiscard]] virtual std::vector<std::shared_ptr<Parameter>> get_params() const override;
};

} // namespace kmlcpplib
