//
// Created by jewoo on 2025-02-14.
//

#pragma once

#include "abstract_merge_node.h"

namespace nnetcpp {
    class MergeSum : public AbstractMergeNode {
    public:
        MergeSum() = default;

        void forward() override;

        void backward() override;
    };
}
