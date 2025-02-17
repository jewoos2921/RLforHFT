//
// Created by jewoo on 2025-02-12.
//
#include "abstract_merge_node.h"

namespace nnetcpp {
    AbstractMergeNode::AbstractMergeNode() {
    }

    void AbstractMergeNode::addInput(Port *input) {
        unsigned int dim = input->value.rows();

        if (_inputs.size() == 0) {
            _output.value = Vector::Zero(dim);
            _output.error = Vector::Zero(dim);
        }
    }

    AbstractNode::Port *AbstractMergeNode::output() {
        return &_output;
    }

    void AbstractMergeNode::update() {
    }

    void AbstractMergeNode::clearError() {
        _output.error.setZero();
        _output.value.setZero();
    }
}
