//
// Created by jewoo on 2025-02-14.
//

#pragma once
#include "abstract_merge_node.h"

namespace nnetcpp {
    class MergeProduct : public AbstractMergeNode {
    public:
        MergeProduct() = default;

        void forward() override;

        void backward() override;
    };
}
