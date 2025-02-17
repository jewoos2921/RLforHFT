//
// Created by jewoo on 2025-02-13.
//

#pragma once

#include "abstract_recurrent_network_node.h"

namespace nnetcpp {
    class Network : public AbstractRecurrentNetworkNode {
    public:
        Network(unsigned int inputs);

        Port *inputPort();

        Port *output() override;

        template<typename Derived>
        Vector predict(const Eigen::MatrixBase<Derived> &input);

        void reset() override;

        template<typename Derived>
        Float setExpectedOutput(const Eigen::MatrixBase<Derived> &output);

        template<typename DerivedA, typename DerivedB>
        Float setExpectedOutput(const Eigen::MatrixBase<DerivedA> &output,
                                const Eigen::MatrixBase<DerivedB> &weights);

        template<typename Derived>
        Float setError(const Eigen::MatrixBase<Derived> &error);


        void clearError() override;

        void update() override;

        template<typename DerivedA, typename DerivedB>
        Float trainSample(const Eigen::MatrixBase<DerivedA> &input,
                          const Eigen::MatrixBase<DerivedB> &output);

        template<typename DerivedA, typename DerivedB, typename DerivedC>
        Float trainSample(const Eigen::MatrixBase<DerivedA> &input,
                          const Eigen::MatrixBase<DerivedB> &output,
                          const Eigen::MatrixBase<DerivedC> &weights);

        void train(const Eigen::MatrixXf &inputs,
                   const Eigen::MatrixXf &outputs,
                   unsigned int batch_size,
                   unsigned int epochs);

        void train(const Eigen::MatrixXf &inputs,
                   const Eigen::MatrixXf &outputs,
                   const Eigen::MatrixXf &weights,
                   unsigned int batch_size,
                   unsigned int epochs);

        void trainSequence(const Eigen::MatrixXf &inputs,
                           const Eigen::MatrixXf &outputs,
                           unsigned int epochs);

        void trainSequence(const Eigen::MatrixXf &inputs,
                           const Eigen::MatrixXf &outputs,
                           const Eigen::MatrixXf &weights,
                           unsigned int epochs);

    private:
        template<typename Derived>
        void predict(const Eigen::MatrixBase<Derived> &input,
                     Vector *rs);

        template<typename DerivedA, typename DerivedB>
        Float setExpectedOutput(const Eigen::MatrixBase<DerivedA> &output,
                                const Eigen::MatrixBase<DerivedB> *weights);

        template<typename DerivedA, typename DerivedB, typename DerivedC>
        Float trainSample(const Eigen::MatrixBase<DerivedA> &input,
                          const Eigen::MatrixBase<DerivedB> &output,
                          const Eigen::MatrixBase<DerivedC> *weights);

        void train(const Eigen::MatrixXf &inputs,
                   const Eigen::MatrixXf &outputs,
                   const Eigen::MatrixXf *weights,
                   unsigned int batch_size,
                   unsigned int epochs);

        void trainSequence(const Eigen::MatrixXf &inputs,
                           const Eigen::MatrixXf &outputs,
                           const Eigen::MatrixXf *weights,
                           unsigned int epochs);

        Port _input_port;
    };

    template<typename Derived>
    Vector Network::predict(const Eigen::MatrixBase<Derived> &input) {
        Vector rs;
        predict(input, &rs);
        return rs;
    }


    template<typename Derived>
    void Network::predict(const Eigen::MatrixBase<Derived> &input, Vector *rs) {
        assert(input.cols() == _input_port .value.rows());

        _input_port.value = input;

        for (AbstractNode *node: _nodes) {
            node->forward();
        }
        if (rs) {
            *rs = output()->value;
        }
    }

    template<typename Derived>
    Float Network::setExpectedOutput(const Eigen::MatrixBase<Derived> &output) {
        return setExpectedOutput(output, (Eigen::MatrixXf *) 0);
    }

    template<typename DerivedA, typename DerivedB>
    Float Network::setExpectedOutput(const Eigen::MatrixBase<DerivedA> &output,
                                     const Eigen::MatrixBase<DerivedB> &weights) {
        return setExpectedOutput(output, &weights);
    }

    template<typename Derived>
    Float Network::setError(const Eigen::MatrixBase<Derived> &error) {
        assert(error.rows() == output()->error.rows());

        output()->error = error;

        backward();

        return error.array().square().mean();
    }

    template<typename DerivedA, typename DerivedB>
    Float Network::trainSample(const Eigen::MatrixBase<DerivedA> &input, const Eigen::MatrixBase<DerivedB> &output) {
        return trainSample(input, output, (Eigen::MatrixXf *) 0);
    }

    template<typename DerivedA, typename DerivedB, typename DerivedC>
    Float Network::trainSample(const Eigen::MatrixBase<DerivedA> &input, const Eigen::MatrixBase<DerivedB> &output,
                               const Eigen::MatrixBase<DerivedC> &weights) {
        return trainSample(input, output, &weights);
    }

    template<typename DerivedA, typename DerivedB>
    Float Network::setExpectedOutput(const Eigen::MatrixBase<DerivedA> &output,
                                     const Eigen::MatrixBase<DerivedB> *weights) {
        if (weights == nullptr) {
            return setError(output - this->output()->value);
        } else {
            return setError((output - this->output()->value).cwiseProduct(*weights));
        }
    }

    template<typename DerivedA, typename DerivedB, typename DerivedC>
    Float Network::trainSample(const Eigen::MatrixBase<DerivedA> &input, const Eigen::MatrixBase<DerivedB> &output,
                               const Eigen::MatrixBase<DerivedC> *weights) {
        Float error;

        predict(input, nullptr);
        error = setExpectedOutput(output, weights);
        update();
        return error;
    }
}
