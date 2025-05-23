#include "batch_norm.hpp"
#include "matrix.hpp"
#include <cmath>

BatchNorm:: BatchNorm(int input_dim, int output_dim)
    : input_shape({input_dim, output_dim}){
        gamma = Matrix(1, output_dim, 1.0);
        beta = Matrix(1, output_dim, 0.0);

        d_gamma = Matrix(1, output_dim, 0.0);
        d_beta = Matrix(1, output_dim, 0.0);

        x_hat = Matrix(input_dim, output_dim);

    }

Matrix BatchNorm::forward(const Matrix& input) {
    input_cache = input;  // cache input for backward use

    int m = input.rows;   // batch size
    int n = input.cols;   // number of features

    // Step 1: Compute mean for each feature (column-wise)
    // μ_j = (1/m) ∑_i x_ij
    mean = BatchNorm::compute_mean(input);  // shape: (1 × n)

    // Step 2: Center the input by subtracting the mean
    // x_centered_ij = x_ij - μ_j
    Matrix centered(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            centered.data[i][j] = input.data[i][j] - mean.data[0][j];

    // Step 3: Compute variance for each feature (on centered data)
    // σ²_j = (1/m) ∑_i (x_ij - μ_j)^2
    variance = BatchNorm::compute_variance(centered);  // shape: (1 × n)

    // Step 4: Compute standard deviation with epsilon for numerical stability
    // σ_j = sqrt(σ²_j + ε)
    Matrix standard_deviation(1, n);
    for (int j = 0; j < n; ++j)
        standard_deviation.data[0][j] = std::sqrt(variance.data[0][j] + epsilon);

    standard_deviation_cache = standard_deviation;  // cache for backward()

    // Step 5: Normalize the input
    // x̂_ij = (x_ij - μ_j) / σ_j
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            x_hat.data[i][j] = centered.data[i][j] / standard_deviation.data[0][j];

    // Step 6: Scale and shift
    // y_ij = γ_j * x̂_ij + β_j
    Matrix output(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            output.data[i][j] = gamma.data[0][j] * x_hat.data[i][j] + beta.data[0][j];

    return output;
}

Matrix BatchNorm::compute_mean(const Matrix&input) {
    Matrix mean = input.col_sum();
    for(int j=0;j<mean.cols;++j){
        mean.data[0][j] /= input.rows;
    }
    return mean;
}

Matrix BatchNorm::compute_variance(const Matrix&input) {
    Matrix variance = Matrix(1, input.cols);

    for(int i=0;i<input.rows;++i){
        for(int j=0;j<input.cols;++j){
            variance.data[0][j] += input.data[i][j]*input.data[i][j];
        }
    }

    for(int j=0;j<variance.cols;++j){
        variance.data[0][j] /= input.rows;
    }

    return variance;
}

Matrix BatchNorm::backward(const Matrix& grad_out) {
    int m = grad_out.rows;
    int n = grad_out.cols;

    // dy * gamma — element-wise scaling
    // ∂L/∂y * γ : broadcast γ across batch
    Matrix dy_gamma(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            dy_gamma.data[i][j] = grad_out.data[i][j] * gamma.data[0][j];

    // ∑(dy * gamma) — sum over the batch (along rows)
    Matrix sum_dy_gamma = dy_gamma.col_sum(); // shape (1 x n)

    // (dy * gamma) * x̂ — element-wise product
    // Used in ∑(∂L/∂y * γ * x̂) term
    Matrix dy_gamma_xhat(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            dy_gamma_xhat.data[i][j] = dy_gamma.data[i][j] * x_hat.data[i][j];

    // ∑((dy * gamma) * x_hat)
    Matrix sum_dy_gamma_xhat = dy_gamma_xhat.col_sum(); // shape (1 x n)

    // Final gradient input calculation using canonical batchnorm derivative
    // ∂L/∂x = (1 / mσ) * [ m·(dy·γ) - ∑(dy·γ) - x̂·∑((dy·γ)·x̂) ]
    Matrix grad_input(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sigma = standard_deviation_cache.data[0][j]; // σ = sqrt(var + ε)
            double term1 = dy_gamma.data[i][j] * m;             // m·(dy·γ)
            double term2 = sum_dy_gamma.data[0][j];             // ∑(dy·γ)
            double term3 = x_hat.data[i][j] * sum_dy_gamma_xhat.data[0][j]; // x̂·∑((dy·γ)·x̂)
            grad_input.data[i][j] = (1.0 / (m * sigma)) * (term1 - term2 - term3);
        }
    }

    // ∂L/∂γ = ∑(∂L/∂y · x̂)
    d_gamma = (grad_out * x_hat).col_sum();

    // ∂L/∂β = ∑(∂L/∂y)
    d_beta = grad_out.col_sum();

    return grad_input;
}

void BatchNorm::update(double learning_rate){
    // Gradient descent update:
    // γ ← γ - η * ∂L/∂γ
    // β ← β - η * ∂L/∂β
    for (int j = 0; j < gamma.cols; ++j) {
        gamma.data[0][j] -= learning_rate * d_gamma.data[0][j];
        beta.data[0][j]  -= learning_rate * d_beta.data[0][j];
    }
}

std::string BatchNorm::get_name() const{
    return "BatchNormalization";
}

std::pair<int,int> BatchNorm::get_input_shape() const {
    return input_shape;
}

std::pair<int,int> BatchNorm::get_output_shape() const {
    return input_shape;
}

int BatchNorm::param_count() const{
    return gamma.rows * gamma.cols + beta.rows * beta.cols;
}
