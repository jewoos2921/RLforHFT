//
// Created by jewoo on 2025-02-13.
//

#include "dense.h"
#include "network_serializer.h"

namespace nnetcpp {
    float Dense::momentum = 0.1f;

    template<typename Derived>
    void _serialize(NetworkSerializer &serializer,
                    const Eigen::PlainObjectBase<Derived> &value) {
        unsigned int count = value.rows() * value.cols();
        const float *data = value.data();

        for (unsigned int i = 0; i < count; ++i) {
            serializer.writeWeight(data[i]);
        }
    }

    template<typename Derived>
    void _deserialize(NetworkSerializer &serializer,
                      Eigen::PlainObjectBase<Derived> &value) {
        unsigned int count = value.rows() * value.cols();
        float *data = value.data();

        for (unsigned int i = 0; i < count; ++i) {
            data[i] = serializer.readWeight();
        }
    }

    Dense::Dense(unsigned int outputs, Float learning_rate, Float decay,
                 bool bias_initialized_at_one) : _input(nullptr), _learning_rate(learning_rate),
                                                 _decay(decay), _bias_initialized_at_one(bias_initialized_at_one) {
        _output.error.resize(outputs);
        _output.value.resize(outputs);
    }

    void Dense::serialize(NetworkSerializer &serializer) {
        _serialize(serializer, _weights);
        _serialize(serializer, _avg_d_weights);
        _serialize(serializer, _bias);
        _serialize(serializer, _avg_d_bias);
    }

    void Dense::deserialize(NetworkSerializer &serializer) {
        _deserialize(serializer, _weights);
        _deserialize(serializer, _avg_d_weights);
        _deserialize(serializer, _bias);
        _deserialize(serializer, _avg_d_bias);
    }

    void Dense::setInput(Port *input) {
        _input = input;

        unsigned int inputs = _input->value.rows();
        unsigned int outputs = _output.value.rows();

        _weights = Matrix::Random(outputs, inputs) * 0.1f;
        _d_weights = Matrix::Zero(outputs, inputs);
        _avg_d_weights = Matrix::Zero(outputs, inputs);
        _d_bias = Vector::Zero(outputs);
        _avg_d_bias = Vector::Zero(outputs);

        if (_bias_initialized_at_one) {
            _bias = Vector::Ones(outputs);
        } else {
            _bias = Vector::Random(outputs) * 0.1f;
        }

        clearError();
    }

    AbstractNode::Port *Dense::output() {
        return &_output;
    }

    void Dense::forward() {
        _output.value.noalias() = _weights * _input->value;
        _output.value += _bias;
    }

    void Dense::backward() {
        _input->error.noalias() += _weights.transpose() * _output.error;
        _d_weights.noalias() += _output.error * _input->value.transpose();
        _d_bias.noalias() += _output.error;
    }

    void Dense::update() {
        float normalization_factor = 1.0f / float(_max_timestep + 1);

        _d_weights *= normalization_factor;
        _d_bias *= normalization_factor;

        _avg_d_weights = _decay * _avg_d_weights + (1.0f - _decay) * _d_weights.array().square().matrix();
        _avg_d_bias = _decay * _avg_d_bias + (1.0f - _decay) * _d_bias.array().square().matrix();

        _weights.noalias() -= (_learning_rate * _d_weights).cwiseQuotient(
            (_avg_d_weights.array().sqrt() + 1e-3).matrix());
        _bias.noalias() -= (_learning_rate * _d_bias).cwiseQuotient(
            (_avg_d_bias.array().sqrt() + 1e-3).matrix());
    }

    void Dense::clearError() {
        _output.error.setZero();
        _output.value.setZero();

        _d_weights *= momentum;
        _d_bias *= momentum;
    }

    void Dense::setCurrentTimestep(unsigned int timestep) {
        (void) timestep;

        _output.error.setZero();
        _output.value.setZero();

        _max_timestep = std::max(_max_timestep, timestep);
    }

    void Dense::reset() {
        _max_timestep = 0;
    }
}
