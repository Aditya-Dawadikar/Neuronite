#ifndef ADAM_OPTIMIZER_HPP
#define ADAM_OPTIMIZER_HPP

#include "optimizer.hpp"
#include <unordered_map>

class AdamOptimizer:public Optimizer{
    private:
        double lr;
        double beta1;
        double beta2;
        double epsilon;

        // t-th moment estimates for each layer pointer
        std::unordered_map<Layer*, Matrix> m_weights;
        std::unordered_map<Layer*, Matrix> v_weights;
        std::unordered_map<Layer*, Matrix> m_bias;
        std::unordered_map<Layer*, Matrix> v_bias;

    public:
        AdamOptimizer(double lr=0.001, double beta1=0.9, double beta2=0.999, double epsilon=1e-8);

        void step(Layer* layer, int t) override;
};

#endif