#include <KMLCPPLib/Layers/SequenceLayer.hpp>
#include <vector>

namespace kmlcpplib {

SequenceLayer::SequenceLayer(
    const std::vector<std::shared_ptr<LayerBase>> &in_layers)
    : LayerBase(in_layers.front()->get_in_nodes(),
                in_layers.back()->get_out_nodes()),
      stored_layers(in_layers) {
          for(int i =1;i < in_layers.size();i++) {
              assert(in_layers[i - 1]->get_out_nodes() == in_layers[i]->get_in_nodes() && "Layer node mismatch");
          }
      }

Eigen::VectorXd SequenceLayer::forward(const Eigen::VectorXd &x) {
    Eigen::VectorXd latest_output = x;
    for (const auto &layer : stored_layers) {
        latest_output = layer->forward(latest_output);
    }
    return latest_output;
}

Eigen::VectorXd SequenceLayer::backward(const Eigen::VectorXd &upstream_grad) {

    Eigen::VectorXd latest_upstream_grad = upstream_grad;

    for( auto it = stored_layers.rbegin();it != stored_layers.rend() ; ++it) {
        latest_upstream_grad = (*it)->backward(latest_upstream_grad);
    }
    return latest_upstream_grad;
}

[[nodiscard]] std::vector<std::shared_ptr<Parameter>>
SequenceLayer::get_params() const  {
    std::vector<std::shared_ptr<Parameter>> combined_list;
    for(const auto &layer: stored_layers) {
        auto layer_params = layer->get_params();
        combined_list.insert(combined_list.end(), layer_params.begin(), layer_params.end());
    }
    return combined_list;
}

} // namespace kmlcpplib
