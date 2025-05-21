#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include <matrix.hpp>

class DenseLayer{
    private:
        Matrix weights;
        Matrix bias;

        Matrix input_cache;
        Matrix d_weights;
        Matrix d_bias;
    
    public:
        DenseLayer(int input_dim, int output_dim);

        Matrix forward(const Matrix& input);
        Matrix backward(const Matrix& grad_output);

        void update(double learning_rate);

        Matrix get_weights() const;
        Matrix get_bias() const; 
        Matrix get_d_weights() const;
        Matrix get_d_bias() const;
};

#endif