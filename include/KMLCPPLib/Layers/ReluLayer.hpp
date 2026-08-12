#pragma once

#include "KMLCPPLib/Layers/LayerBase.hpp"

namespace kmlcpplib {
class ReluLayer : public LayerBase {
    private:
        struct {
        } cache;

    public:
        explicit ReluLayer(uint32_t in_nodes) : LayerBase(in_nodes) {}
        ~ReluLayer() override = default;


};
}

