//
// Created by jewoo on 2025-02-13.
//

#include "state_aggregation.h"
#include <fstream>

namespace rlagent {
    StateAggregation::StateAggregation(int number_of_actions, int dimensions_of_statespace, double step_size,
                                       const Eigen::Ref<const VectorXI> &segments,
                                       const Eigen::Ref<const VectorXF> &min_values,
                                       const Eigen::Ref<const VectorXF> &max_values,
                                       const Eigen::Ref<const VectorXF> &action_kernel,
                                       double init_min_value, double init_max_value): Approximator(
            number_of_actions, dimensions_of_statespace),
        _step_size(step_size),
        _action_kernel(action_kernel),
        _segments(segments),
        _min_values(min_values),
        _max_values(max_values) {
        auto size_statespace = max_values - min_values;

        _segment_size = size_statespace.array() / segments.cast<float>().array();

        _values = (VectorXF::Random(
                       segments.prod() * number_of_actions).array() + 1.0) / 2.0
                  * (init_max_value - init_min_value) + init_min_value;
    }

    Eigen::Ref<VectorXF> StateAggregation::getValues() {
        return _values;
    }

    void StateAggregation::save(std::string filename) {
        std::ofstream outfile(filename, std::ios_base::binary);
        if (outfile.is_open()) {
            outfile.write(
                reinterpret_cast<const char *>(_values.data()),
                static_cast<int64_t>(_values.size() * sizeof(_values.data()[0])));
            outfile.close();
        }
    }

    void StateAggregation::load(std::string filename) {
        std::ifstream infile(filename, std::ios_base::binary);
        if (infile.good()) {
            infile.read(
                reinterpret_cast<char *>(_values.data()),
                static_cast<int64_t>(_values.size() * sizeof(_values.data()[0])));
            infile.close();
        }
    }

    VectorXI StateAggregation::get_indices(VectorXD state) {
        VectorXI indices_out = VectorXI::Zero(_number_of_actions);
        VectorXF state_shifted = state.cast<float>() - _min_values;
        VectorXI indices = (state_shifted.array() / _segment_size.array()).cast<int>();

        indices = indices.array().min(_segments.array() - 1);
        unsigned int index = indices[0];
        for (int i = 1; i < indices.size(); i++) {
            index *= _segments[i];
            index += indices[i];
        }

        for (int i = 0; i < _number_of_actions; i++) {
            indices_out.coeffRef(i) = index + i * _segments.prod();
        }
        return indices_out;
    }

    double StateAggregation::predict_implementation(Eigen::Ref<const VectorXF> state, int action) {
        VectorXI indices = get_indices(state);
        double prediction = _action_kernel[0] * _values[indices[action]];
        for (int i = 1; i < _action_kernel.size(); i++) {
            int actionP = std::min(action + i, _number_of_actions - 1);
            int action_n = std::max(action - i, 0);
            prediction += _action_kernel[i] * _values[indices[actionP]] +
                    _action_kernel[i] * _values[indices[action_n]];
        }
        return prediction;
    }

    double StateAggregation::update_implementation(const Eigen::Ref<const VectorXD> state, int action, double target) {
        VectorXI indices = get_indices(state);

        double prediction = _action_kernel[0] * _values[indices[action]];
        for (int i = 1; i < _action_kernel.size(); i++) {
            int actionP = std::min(action + i, _number_of_actions - 1);
            int action_n = std::max(action - i, 0);
            prediction += _action_kernel[i] * _values[indices[actionP]] +
                    _action_kernel[i] * _values[indices[action_n]];
        }
        double prediction_error = target - prediction;
        _values[indices[action]] += _step_size * prediction_error * _action_kernel[0];
        for (int i = 0; i < _action_kernel.size(); i++) {
            int actionP = std::min(action + i, _number_of_actions - 1);
            int action_n = std::max(action - i, 0);
            _values[indices[actionP]] += _step_size * prediction_error * _action_kernel[i];
            _values[indices[action_n]] += _step_size * prediction_error * _action_kernel[i];
        }
        return prediction_error;
    }
}
