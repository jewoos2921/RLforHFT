//
// Created by jewoo on 2025-02-15.
//

#include "tile_coding.h"
#include <fstream>

namespace rlagent {
    TileCoding::TileCoding(int number_of_actions, int dimensions_of_statespace, double step_size, int tilings,
                           const Eigen::Ref<const VectorXI> &displacement, const Eigen::Ref<const VectorXI> &segments,
                           const Eigen::Ref<const VectorXF> &min_values, const Eigen::Ref<const VectorXF> &max_values,
                           const Eigen::Ref<const VectorXF> &action_kernel, double init_min_value,
                           double init_max_value): Approximator(number_of_actions, dimensions_of_statespace) {
        auto size_statespace = max_values - min_values;
        auto segment_size = size_statespace.array() / segments.cast<float>().array();

        auto tile_size = segment_size.array() / float(tilings);

        for (int i = 0; i < tilings; i++) {
            auto offset = tile_size.array() * displacement.cast<float>().array() * float(i);
            auto layer_min_values = min_values.array() - offset.array();
            auto layer_segments = segments.cast<float>().array() + (offset.array() / segment_size.array()).ceil();
            auto layer_max_values = layer_min_values.array() + segment_size.array() * layer_segments.array();

            _layers.push_back(StateAggregation(number_of_actions,
                                               dimensions_of_statespace,
                                               step_size / tilings,
                                               layer_segments.cast<int>(),
                                               layer_min_values,
                                               layer_max_values,
                                               action_kernel,
                                               init_min_value / tilings,
                                               init_max_value / tilings
            ));
        }
    }

    void TileCoding::save(std::string filename) {
        std::ofstream outfile(filename, std::ios_base::binary);
        if (outfile.is_open()) {
            for (auto &layer: _layers) {
                auto *data = layer.getValues().data();
                auto data_size = layer.getValues().size();
                outfile.write(reinterpret_cast<const char *>(data),
                              static_cast<int64_t>(data_size * sizeof(data[0])));
            }
            outfile.close();
        }
    }

    void TileCoding::load(std::string filename) {
        std::ifstream infile(filename, std::ios_base::binary);
        if (infile.good()) {
            for (auto &layer: _layers) {
                auto *data = layer.getValues().data();
                auto data_size = layer.getValues().size();
                infile.read(reinterpret_cast<char *>(data),
                            static_cast<int64_t>(data_size * sizeof(data[0])));
            }
            infile.close();
        }
    }

    VectorXD TileCoding::predict(const Eigen::Ref<const VectorXD> &states,
                                 const Eigen::Ref<const VectorXI> &actions) {
        VectorXD prediction = VectorXD::Zero(actions.size());
        for (auto &layer: _layers) {
            VectorXD layer_prediction = layer.predict(states, actions) / _layers.size();
            prediction = prediction + layer_prediction;
        }
        return prediction;
    }

    double TileCoding::update(const Eigen::Ref<const VectorXD> &state, int action, double target) {
        double td_error{0.0};

        for (auto &layer: _layers) {
            double layer_error = layer.update(state, action, target) / _layers.size();
            td_error = td_error + layer_error;
        }
        return td_error;
    }
}
