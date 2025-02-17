//
// Created by jewoo on 2025-02-15.
//

#include "cw_rnn.h"
#include "dense.h"
#include "merge_sum.h"
#include <iostream>
#include <cassert>

namespace nnetcpp {
    CWRNN::CWRNN(unsigned int num_units, unsigned int size, Float learning_rate,
                 Float decay): _learning_rate(learning_rate), _decay(decay) {
        _output = new MergeSum;

        _units.resize(num_units);
        _unit_size = size / num_units;

        assert(_unit_size * num_units == size);

        for (unsigned int i = 0; i < num_units; ++i) {
            Unit &unit = _units[i];

            unit.sum = new MergeSum;
            unit.activation = new TanhActivation;
            unit.skip = new LinearActivation;
            unit.output = new MergeSum;

            for (unsigned int j = 0; j <= i; ++j) {
                Dense *dense = new Dense(_unit_size, _learning_rate, _decay);
                Unit &prev_unit = _units[j];

                dense->setInput(dense->output());

                unit.inputs.push_back(dense);
                unit.sum->addInput(dense->output());
                addNode(dense);

                if (j == 0) {
                    unit.activation->setInput(unit.sum->output());
                    unit.output->addInput(unit.activation->output());
                    unit.output->addInput(unit.skip->output());
                    unit.skip->setInput(unit.output->output());
                }

                dense->setInput(prev_unit.output->output());
            }

            addRecurrentNode(unit.output);
            addNode(unit.sum);
            addNode(unit.activation);
            addNode(unit.skip);
            addNode(unit.output);

            _output->addInput(unit.output->output());
        }
        reset();
    }

    void CWRNN::addInput(Port *input) {
        for (Unit &unit: _units) {
            Dense *dense = new Dense(_unit_size, _learning_rate, _decay);

            dense->setInput(input);

            unit.sum->addInput(dense->output());
            unit.inputs.push_back(dense);
            _inputs.push_back(dense);

            addNode(dense);
        }
    }

    AbstractNode::Port *CWRNN::output() {
        return _output->output();
    }

    void CWRNN::forward() {
        forUnits(
            currentTimestep(),
            [](Unit &enabled) {
                for (Dense *dense: enabled.inputs) {
                    dense->forward();
                }
                enabled.sum->forward();
                enabled.activation->forward();
            },
            [](Unit &disabled) {
                disabled.skip->forward();
            });
        forwardRecurrent();
    }

    void CWRNN::backward() {
        _output->backward();

        forUnits(
            currentTimestep(),
            [](Unit &enabled) {
                enabled.output->backward();
                enabled.activation->backward();
                enabled.sum->backward();
                for (Dense *dense: enabled.inputs) {
                    dense->backward();
                }
            },
            [](Unit &disabled) {
                disabled.output->backward();
                disabled.skip->backward();
            }
        );
        backwardRecurrent();
    }
}
