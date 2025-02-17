//
// Created by jewoo on 2025-02-15.
//

#pragma once
#include "abstract_recurrent_network_node.h"
#include "activation.h"

namespace nnetcpp {
    class MergeSum;

    class GRU : public AbstractRecurrentNetworkNode {
    public:
        GRU(unsigned int size, Float learning_rate, Float decay = 0.9f);

        void addInput(Port *input);

        void addZ(Port *z);

        void addR(Port *r);

        Port *output() override;

        void setCurrentTimestep(unsigned int timestep) override;

    private:
        MergeSum *_inputs;
        MergeSum *_updates;
        MergeSum *_resets;
        LinearActivation *_real_output;
        LinearActivation *_recurrent_output;
    };
}
