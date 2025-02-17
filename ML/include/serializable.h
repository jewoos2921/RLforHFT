//
// Created by jewoo on 2025-02-16.
//

#pragma once

#include "json.hpp"

namespace ppo_cpp {
    class ISerializable {
    public:
        virtual void serialize(nlohmann::json &json) = 0;

        virtual void deserialize(nlohmann::json &json) = 0;
    };
}
