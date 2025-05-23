#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>

class Matrix {

    public:
        int rows, cols;
        std::vector<std::vector<double>> data;

        Matrix();
        Matrix(int rows, int cols);
        Matrix(int rows, int cols, double init_val);
        Matrix(const std::vector<std::vector<double>>& values);

        static Matrix dot(const Matrix& A, const Matrix& B);
        Matrix transpose() const;
        Matrix col_sum() const;

        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator*(double scalar) const;

        void print() const;
};

#endif