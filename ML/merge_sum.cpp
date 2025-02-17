//
// Created by jewoo on 2025-02-14.
//

#include "merge_sum.h"

namespace nnetcpp {
    void MergeSum::forward() {
        _output.value.setZero();

        for (Port *port: _inputs) {
            _output.value.noalias() += port->value;
        }
    }

    void MergeSum::backward() {
        for (Port *port: _inputs) {
            port->error.noalias() += _output.error;
        }
    }
}
