//
// Created by jewoo on 2025-02-12.
//

#pragma once
#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>

namespace MD::Common {
    class Matrix {
    public :
        Matrix();

        Matrix(int height, int width);

        Matrix(std::vector<std::vector<double> > const &array);

        Matrix add(Matrix const &m) const;

        Matrix subtract(Matrix const &m) const;

        Matrix multiply(Matrix const &m) const;

        Matrix dot(Matrix const &m) const;

        Matrix transpose() const;

        Matrix multiply(double const &value) const;

        Matrix applyFunction(double (*function)(double)) const;

        int getWidth() const;

        int getHeight() const;

        double get(int i, int j) const;

        double sum() const;

        void print(std::ostream &flux) const;

    private:
        std::vector<std::vector<double> > array_;
        int height_ = 0;
        int width_ = 0;
    };

    static std::ostream &operator<<(std::ostream &flux, Matrix const &matrix) {
        matrix.print(flux);
        return flux;
    }
}
