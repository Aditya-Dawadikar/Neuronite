#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include "matrix.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

class Model{
    private:
        std::vector<Layer*> layers;
    
    public:
        void add(Layer* layer);
        Matrix forward(const Matrix& loss_grad);
        Matrix backward(const Matrix& loss_grad);
        void update(double learning_rate);
        static double compute_accuracy(const Matrix& prediction,
                                const Matrix& target);
        void train(const Matrix& input,
                    const Matrix& target,
                    Loss& loss_fn,
                    Optimizer& optimizer,
                    int epochs,
                    int patience = 10
                );
        void summarize(int input_dim);
};

#endif