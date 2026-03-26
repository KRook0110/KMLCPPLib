#pragma once

namespace kmlcpplib {

class MathGeomNode {

public:
    virtual ~MathGeomNode();
    virtual void forward_feed();
    virtual void derivative();
    virtual void backward_prop();

};
}
