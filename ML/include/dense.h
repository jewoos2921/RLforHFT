//
// Created by jewoo on 2025-02-13.
//


#pragma once
#include "abstract_node.h"

namespace nnetcpp {
    class Dense : public AbstractNode {
    public:
        static float momentum;

        Dense(unsigned int outputs,
              Float learning_rate,
              Float decay = 0.9f,
              bool bias_initialized_at_one = false);

        void setInput(Port *input);

        void serialize(NetworkSerializer &serializer) override;

        void deserialize(NetworkSerializer &serializer) override;

        Port *output() override;

        void forward() override;

        void backward() override;

        void update() override;

        void clearError() override;

        void reset() override;

        void setCurrentTimestep(unsigned int timestep) override;

    private:
        Port *_input;
        Float _learning_rate;
        Float _decay;
        bool _bias_initialized_at_one;

        Port _output;

        Matrix _weights;
        Matrix _d_weights;
        Matrix _avg_d_weights;
        Matrix _bias;
        Matrix _d_bias;
        Matrix _avg_d_bias;

        unsigned int _max_timestep;
    };
}
