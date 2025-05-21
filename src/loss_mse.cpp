#include "matrix.hpp"
#include "loss_mse.hpp"
#include <cmath>

double LossMSE::forward(const Matrix& prediction, const Matrix& target){

    if(prediction.rows!=target.rows || prediction.cols!=target.cols){
        throw std::invalid_argument("LossMSE::forward: Shape mismatch");
    }

    prediction_cache = prediction;
    target_cache = target;

    double loss = 0.0;

    for(int i=0;i<prediction.rows;++i){
        for(int j=0;j<target.cols;++j){
            loss += std::pow(prediction.data[i][j] - target.data[i][j],2);
        }
    }

    int total_elements = prediction.rows * prediction.cols;

    loss /= total_elements;

    return loss;
}

Matrix LossMSE::backward(){
    Matrix grad_input = Matrix(prediction_cache.rows, prediction_cache.cols);

    int total_elements = prediction_cache.rows*prediction_cache.cols;

    for(int i=0;i<grad_input.rows;++i){
        for(int j=0;j<grad_input.cols;++j){
            grad_input.data[i][j] = (2.0/total_elements)*(prediction_cache.data[i][j]-target_cache.data[i][j]);
        }
    }

    return grad_input;
}