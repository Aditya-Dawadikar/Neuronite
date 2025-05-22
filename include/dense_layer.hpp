#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include <matrix.hpp>
#include "layer.hpp"

class DenseLayer: public Layer{
    private:

        Matrix input_cache;
        Matrix d_weights;
        Matrix d_bias;

        std::pair<int,int> input_shape;
        std::pair<int,int> output_shape;
    
    public:
        Matrix weights;
        Matrix bias;

        DenseLayer(int input_dim, int output_dim);

        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;
        void update(double learning_rate) override;
        std::string get_name() const override;
        std::pair<int,int> get_input_shape() const override;
        std::pair<int,int> get_output_shape() const override;
        int param_count() const override;
        void apply_adam_update(const Matrix& new_weights, const Matrix& new_bias);

        Matrix get_weights() const;
        Matrix get_bias() const; 
        Matrix get_d_weights() const;
        Matrix get_d_bias() const;
};

#endif