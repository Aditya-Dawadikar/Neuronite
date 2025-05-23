#ifndef BATCH_NORM_HPP
#define BATCH_NORM_HPP

#include "matrix.hpp"
#include "layer.hpp"

class BatchNorm: public Layer{
    private:
        Matrix gamma, beta;
        Matrix mean, variance;
        Matrix input_cache;
        double epsilon = 1e-5;

        Matrix d_gamma, d_beta;
        
        Matrix standard_deviation_cache;

        std::pair<int,int> input_shape;
    
    public:
        Matrix x_hat;

        BatchNorm(int input_dim, int output_dim);

        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_out) override;
        void update(double learning_rate) override;

        std::string get_name() const override;
        std::pair<int,int> get_input_shape() const override;
        std::pair<int,int> get_output_shape() const override;
        int param_count() const override;

        Matrix compute_mean(const Matrix& input);
        Matrix compute_variance(const Matrix& input);

};

#endif