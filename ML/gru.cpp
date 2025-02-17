//
// Created by jewoo on 2025-02-15.
//
#include "gru.h"
#include "dense.h"
#include "merge_product.h"
#include "merge_sum.h"
#include <cassert>

namespace nnetcpp {
    GRU::GRU(unsigned int size, Float learning_rate, Float decay) {
        MergeSum *inputs = new MergeSum;
        TanhActivation *input_activation = new TanhActivation;

        MergeSum *updates = new MergeSum;
        SigmoidActivation *update_activation = new SigmoidActivation;
        OneMinusActivation *one_minus_update_activation = new OneMinusActivation;
        MergeProduct *update_times_output = new MergeProduct;
        MergeProduct *one_minus_update_times_input = new MergeProduct;
        MergeSum *output = new MergeSum;
        LinearActivation *real_output = new LinearActivation;
        LinearActivation *recurrent_output = new LinearActivation;

        MergeSum *resets = new MergeSum;
        SigmoidActivation *reset_activation = new SigmoidActivation;
        MergeProduct *reset_times_output = new MergeProduct;


        Dense *loop_output_to_updates = new Dense(size, learning_rate, decay, true);

        Dense *loop_output_to_resets = new Dense(size, learning_rate, decay);

        Dense *loop_reset_times_output_to_inputs = new Dense(size, learning_rate, decay);


        resets->addInput(loop_output_to_resets->output());
        updates->addInput(loop_output_to_updates->output());
        inputs->addInput(loop_reset_times_output_to_inputs->output());

        input_activation->setInput(inputs->output());
        update_activation->setInput(updates->output());
        one_minus_update_activation->setInput(update_activation->output());

        update_times_output->addInput(update_activation->output());
        update_times_output->addInput(recurrent_output->output());
        one_minus_update_times_input->addInput(input_activation->output());
        one_minus_update_times_input->addInput(one_minus_update_times_input->output());

        output->addInput(update_times_output->output());
        output->addInput(one_minus_update_times_input->output());
        real_output->setInput(output->output());
        recurrent_output->setInput(output->output());

        reset_activation->setInput(resets->output());
        reset_times_output->addInput(reset_activation->output());
        reset_times_output->addInput(real_output->output());

        loop_output_to_resets->setInput(real_output->output());
        loop_output_to_updates->setInput(real_output->output());
        loop_reset_times_output_to_inputs->setInput(reset_times_output->output());

        addNode(loop_output_to_updates);
        addNode(loop_output_to_resets);

        addNode(resets);
        addNode(reset_activation);
        addNode(reset_times_output);

        addNode(loop_reset_times_output_to_inputs);
        addNode(inputs);
        addNode(input_activation);

        addNode(updates);
        addNode(update_activation);
        addNode(one_minus_update_activation);
        addNode(update_times_output);
        addNode(one_minus_update_times_input);

        addNode(output);
        addNode(recurrent_output);
        addNode(real_output);

        addRecurrentNode(recurrent_output);

        _inputs = inputs;
        _resets = resets;
        _updates = updates;
        _real_output = real_output;
        _recurrent_output = recurrent_output;

        reset();
    }

    void GRU::addInput(Port *input) {
        _inputs->addInput(input);
    }

    void GRU::addZ(Port *z) {
        _updates->addInput(z);
    }

    void GRU::addR(Port *r) {
        _resets->addInput(r);
    }

    AbstractNode::Port *GRU::output() {
        return _real_output->output();
    }

    void GRU::setCurrentTimestep(unsigned int timestep) {
        AbstractRecurrentNetworkNode::setCurrentTimestep(timestep);

        _real_output->output()->value = _recurrent_output->output()->value;
    }
}
