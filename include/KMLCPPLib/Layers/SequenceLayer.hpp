#pragma once

#include <KMLCPPLib/Layers/LayerBase.hpp>

namespace kmlcpplib {

class SequenceLayer : public LayerBase {
  private:
    std::vector<std::shared_ptr<LayerBase>> stored_layers;

  public:
    explicit SequenceLayer(
        const std::vector<std::shared_ptr<LayerBase>> &stored_layers);
    ~SequenceLayer() override = default;

    SequenceLayer(SequenceLayer &&other) noexcept = default;
    SequenceLayer &operator=(SequenceLayer &&other) noexcept = default;

    SequenceLayer(const SequenceLayer &) = delete;
    SequenceLayer &operator=(const SequenceLayer &) = delete;

    Eigen::VectorXd forward(const Eigen::VectorXd &input) override;
    Eigen::VectorXd backward(const Eigen::VectorXd &upstream_grad) override;

    [[nodiscard]] virtual std::vector<std::shared_ptr<Parameter>> get_params() const override;
};

} // namespace kmlcpplib
