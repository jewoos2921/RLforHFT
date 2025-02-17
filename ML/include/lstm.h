//
// Created by jewoo on 2025-02-14.
//

#pragma once
#include <abstract_recurrent_network_node.h>

namespace nnetcpp {
    class MergeSum;
    class MergeProduct;


    class LSTM : public AbstractRecurrentNetworkNode {
    public:
        LSTM(unsigned int size, Float learning_rate, Float decay = 0.9f);

        void addInput(Port *input);

        void addIngate(Port *in);

        void addOutgate(Port *out);

        void addForgetgate(Port *forget);


        Port *output() override;

    private:
        MergeSum *_inputs;
        MergeSum *_ingates;
        MergeSum *_outtgates;
        MergeSum *_forgetgates;
        MergeSum *_cells;
        MergeProduct *_output;
    };
}
