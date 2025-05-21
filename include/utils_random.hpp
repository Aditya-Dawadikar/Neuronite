#ifndef UTILS_RANDOM_HPP
#define UTILS_RANDOM_HPP

#include "matrix.hpp"

void set_random_seed(unsigned int seed);
void initialize_random(Matrix& mat, double min=-1.0, double max = 1.0);

#endif