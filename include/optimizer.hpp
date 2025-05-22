#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "layer.hpp"

class Optimizer {
public:
    virtual void step(Layer* layer, int t) = 0;
    virtual ~Optimizer() = default;
};

#endif
