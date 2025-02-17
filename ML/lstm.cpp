//
// Created by jewoo on 2025-02-14.
//
#include "lstm.h"
#include "activation.h"
#include "dense.h"
#include "merge_product.h"
#include "merge_sum.h"

namespace nnetcpp {
    LSTM::LSTM(unsigned int size, Float learning_rate, Float decay) {
        MergeSum *inputs = new MergeSum;
        TanhActivation *input_activation = new TanhActivation;

        MergeSum *input_gate = new MergeSum;
        SigmoidActivation *input_gate_activation = new SigmoidActivation;

        MergeSum *forget_gate = new MergeSum;
        SigmoidActivation *forget_gate_activation = new SigmoidActivation;

        MergeSum *output_gate = new MergeSum;
        SigmoidActivation *output_gate_activation = new SigmoidActivation;

        MergeProduct *input_times_input_gate = new MergeProduct;
        MergeProduct *cells_times_forget_gate = new MergeProduct;
        MergeSum *cells = new MergeSum;

        LinearActivation *cells_recurrent = new LinearActivation;
        TanhActivation *cells_activation = new TanhActivation;
        MergeProduct *cells_times_output_gate = new MergeProduct;

        Dense *loop_output_to_output_gate = new Dense(size, learning_rate, decay);
        Dense *loop_output_to_input_gate = new Dense(size, learning_rate, decay);
        Dense *loop_output_to_forget_gate = new Dense(size, learning_rate, decay, true);
        Dense *loop_output_to_input = new Dense(size, learning_rate, decay);

        inputs->addInput(loop_output_to_input->output());
        input_gate->addInput(loop_output_to_input_gate->output());
        forget_gate->addInput(loop_output_to_forget_gate->output());
        output_gate->addInput(loop_output_to_output_gate->output());

        input_activation->setInput(inputs->output());
        input_gate_activation->setInput(input_gate->output());
        forget_gate_activation->setInput(forget_gate->output());
        output_gate_activation->setInput(output_gate->output());

        input_times_input_gate->addInput(input_gate_activation->output());
        input_times_input_gate->addInput(input_activation->output());

        cells_times_forget_gate->addInput(input_gate_activation->output());
        cells_times_forget_gate->addInput(cells_recurrent->output());

        cells->addInput(input_times_input_gate->output());
        cells->addInput(cells_times_forget_gate->output());
        cells_recurrent->setInput(cells->output());
        cells_activation->setInput(cells->output());

        cells_times_output_gate->addInput(output_gate_activation->output());
        cells_times_output_gate->addInput(cells_activation->output());

        loop_output_to_forget_gate->setInput(cells_recurrent->output());
        loop_output_to_input_gate->setInput(cells_recurrent->output());
        loop_output_to_output_gate->setInput(cells_recurrent->output());
        loop_output_to_input->setInput(cells_recurrent->output());

        addNode(loop_output_to_forget_gate);
        addNode(loop_output_to_input);
        addNode(loop_output_to_input_gate);
        addNode(loop_output_to_output_gate);

        addNode(inputs);
        addNode(input_activation);
        addNode(input_gate);
        addNode(input_gate_activation);
        addNode(forget_gate);
        addNode(forget_gate_activation);
        addNode(output_gate);
        addNode(output_gate_activation);

        addNode(input_times_input_gate);
        addNode(cells_times_forget_gate);
        addNode(cells);
        addNode(cells_recurrent);
        addNode(cells_activation);
        addNode(cells_times_output_gate);

        addRecurrentNode(cells_recurrent);

        _inputs = inputs;
        _ingates = input_gate;
        _outtgates = output_gate;
        _forgetgates = forget_gate;
        _cells = cells;
        _output = cells_times_output_gate;

        reset();
    }

    void LSTM::addInput(Port *input) {
        _inputs->addInput(input);
    }

    void LSTM::addIngate(Port *in) {
        _ingates->addInput(in);
    }

    void LSTM::addOutgate(Port *out) {
        _outtgates->addInput(out);
    }

    void LSTM::addForgetgate(Port *forget) {
        _forgetgates->addInput(forget);
    }

    AbstractNode::Port *LSTM::output() {
        return _output->output();
    }
}
