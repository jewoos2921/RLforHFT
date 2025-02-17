//
// Created by jewoo on 2025-02-13.
//
#include "network.h"
#include <cassert>
#include <algorithm>

namespace nnetcpp {
    Network::Network(unsigned int inputs) {
        _input_port.error = Vector::Zero(inputs);
        _input_port.value = Vector::Zero(inputs);
    }

    AbstractNode::Port *Network::inputPort() {
        return &_input_port;
    }

    AbstractNode::Port *Network::output() {
        return _nodes.back()->output();
    }

    void Network::reset() {
        AbstractRecurrentNetworkNode::reset();
        setCurrentTimestep(0);
    }

    void Network::clearError() {
        AbstractRecurrentNetworkNode::clearError();
        _input_port.error.setZero();
    }

    void Network::update() {
        AbstractRecurrentNetworkNode::update();
        clearError();
    }

    void Network::train(const Eigen::MatrixXf &inputs, const Eigen::MatrixXf &outputs, unsigned int batch_size,
                        unsigned int epochs) {
        train(inputs, outputs, nullptr, batch_size, epochs);
    }

    void Network::trainSequence(const Eigen::MatrixXf &inputs, const Eigen::MatrixXf &outputs, unsigned int epochs) {
        trainSequence(inputs, outputs, nullptr, epochs);
    }

    void Network::train(const Eigen::MatrixXf &inputs, const Eigen::MatrixXf &outputs,
                        const Eigen::MatrixXf &weights,
                        unsigned int batch_size, unsigned int epochs) {
        train(inputs, outputs, &weights, batch_size, epochs);
    }

    void Network::trainSequence(const Eigen::MatrixXf &inputs, const Eigen::MatrixXf &outputs,
                                const Eigen::MatrixXf &weights, unsigned int epochs) {
        trainSequence(inputs, outputs, &weights, epochs);
    }

    void Network::train(const Eigen::MatrixXf &inputs, const Eigen::MatrixXf &outputs, const Eigen::MatrixXf *weights,
                        unsigned int batch_size, unsigned int epochs) {
        std::vector<int> indexes(inputs.cols());
        for (int i = 0; i < inputs.cols(); ++i) {
            indexes[i] = i;
        }

        reset();
        for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
            std::random_shuffle(indexes.begin(), indexes.end());

            unsigned int batch_remaining = batch_size;

            for (int index: indexes) {
                predict(inputs.col(index), nullptr);

                if (weights == nullptr) {
                    setExpectedOutput(outputs.col(index));
                } else {
                    setExpectedOutput(outputs.col(index), weights->col(index));
                }
                if (--batch_remaining == 0) {
                    batch_remaining = batch_size;
                    update();
                }
            }
        }
    }

    void Network::trainSequence(const Eigen::MatrixXf &inputs, const Eigen::MatrixXf &outputs,
                                const Eigen::MatrixXf *weights, unsigned int epochs) {
        Eigen::MatrixXf errors(outputs.rows(), outputs.cols());

        reset();

        for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
            for (int t = 0; t < outputs.cols(); ++t) {
                setCurrentTimestep(t);
                predict(inputs.col(t), nullptr);

                if (weights == nullptr) {
                    errors.col(t) = outputs.col(t) - output()->value;
                } else {
                    errors.col(t) = (outputs.col(t) - output()->value).cwiseProduct(weights->col(t));
                }
            }
            for (int t = outputs.cols() - 1; t >= 0; --t) {
                setCurrentTimestep(t);
                predict(inputs.col(t));
                setError(errors.col(t));
            }
            update();
            reset();
        }
    }
}
