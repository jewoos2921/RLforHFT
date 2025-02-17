//
// Created by jewoo on 2025-02-14.
//
#include "merge_product.h"

namespace nnetcpp {
    void MergeProduct::forward() {
        _output.value.setOnes();

        for (Port *input: _inputs) {
            _output.value.array() *= input->value.array();
        }
    }

    void MergeProduct::backward() {
        for (Port *input: _inputs) {
            input->error.noalias() += _output.error.cwiseProduct(
                _output.value.cwiseQuotient((input->value.array() + 1e20).matrix()));
        }
    }
}
