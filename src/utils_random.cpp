#include "utils_random.hpp"
#include "random"

static std::mt19937 rng(std::random_device{}());

void set_random_seed(unsigned int seed){
    rng.seed(seed);
}

void initialize_random(Matrix &mat, double min, double max){
    std::uniform_real_distribution<double> dist(min, max);

    for(int i=0;i< mat.rows; ++i){
        for(int j=0;j<mat.cols;++j){
            mat.data[i][j] = dist(rng);
        }
    }
}