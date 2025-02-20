#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>


namespace ppo_cpp {
    using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    class Median {
        static std::vector<float> median(const Mat &input,
                                         float epsilon = 1e-4) {
            Mat y{Mat::Zero(1, input.cols())};
            Mat new_y{Mat::Zero(1, input.cols())};


            float error;

            do {
                y = new_y;
                Mat stack_y{Mat::Ones(input.rows(), 1) * y};
                Mat input_sub_y{input - stack_y};


                Mat input_sub_y_sq{input_sub_y.cwiseProduct(input_sub_y)};
                Mat distances{input_sub_y_sq.rowwise().sum().cwiseSqrt()};
                Mat inverted_distance{Mat::Zero(input.rows(), 1)};

                for (int i = 0; i < input.rows(); ++i) {
                    if (distances(i, 0) != 0.f) {
                        inverted_distance(i, 0) = 1.0f / distances(i, 0);
                    }
                }

                Mat input_scaled{input.cwiseProduct(inverted_distance * Mat::Ones(1, input.cols()))};
                new_y = input_scaled.colwise().sum() / inverted_distance.rows();

                error = std::sqrt((new_y - y).cwiseProduct(new_y - y).sum());
            } while (error > epsilon);

            std::vector<float> result(new_y.size());

            Eigen::Map<Mat>(result.data(), new_y.rows(), new_y.cols()) = new_y;
            return std::move(result);
        }
    };
}
