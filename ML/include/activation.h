//
// Created by jewoo on 2025-02-12.
//

#pragma once
#include "abstract_node.h"


namespace nnetcpp {
    template<typename F, typename DF>
    class Activation : public AbstractNode {
    public:
        void setInput(Port *input) {
            unsigned int inputs = input->value.rows();

            _output->value = Vector::Zero(inputs);
            _output->error = Vector::Zero(inputs);
            _input = input;
        }

        Port *output() override {
            return &_output;
        }

        void forward() override {
            _output.value.noalias() = _input->value.unaryExpr<F>();
        }

        void backward() override {
            _input->error.noalias() += _output.error.cwiseProduct(_output.value.unaryExpr<DF>());
        }

        void update() override {
        }

        void clearError() override {
            _output.error.setZero();
            _output.value.setZero();
        }

    private:
        Port *_input;
        Port _output;
    };

    namespace nnetcppinternals {
        inline Float _exp(Float x) {
            if (x < -30.0f) return std::exp(-30.0f);
            if (x > 30.0f) return std::exp(30.0f);
            return std::exp(x);
        }

        struct Tanh {
            Float operator()(Float x) const {
                return 2.0f / (1.0f + _exp(-2.0f * x)) - 1.0f;
            }
        };

        struct dTanh {
            Float operator()(Float x) const {
                return 1.0f - x * x;
            }
        };

        struct Sigmoid {
            Float operator()(Float x) const {
                return 1.0f / (1.0f + _exp(-x));
            }
        };

        struct dSigmoid {
            Float operator()(Float x) const {
                return x * (1.0f - x);
            }
        };

        struct OneMinus {
            Float operator()(Float x) const {
                return 1.0f - x;
            }
        };

        struct dOneMinus {
            Float operator()(Float x) const {
                (void) x;
                return -1.0f;
            }
        };

        struct Linear {
            Float operator()(Float x) const {
                return x;
            }
        };

        struct dLinear {
            Float operator()(Float x) const {
                (void) x;
                return 1.0f;
            }
        };
    }

    typedef Activation<nnetcppinternals::Tanh, nnetcppinternals::dTanh> TanhActivation;
    typedef Activation<nnetcppinternals::Sigmoid, nnetcppinternals::dSigmoid> SigmoidActivation;
    typedef Activation<nnetcppinternals::OneMinus, nnetcppinternals::dOneMinus> OneMinusActivation;
    typedef Activation<nnetcppinternals::Linear, nnetcppinternals::dLinear> LinearActivation;
}
