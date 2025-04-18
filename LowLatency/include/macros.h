//
// Created by jewoo on 2025-03-07.
//

#pragma once
#include <cstring>
#include <iostream>

namespace LL::Common {
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

    inline auto ASSERT(bool cond, const std::string &msg) noexcept {
        if (UNLIKELY(!cond)) {
            std::cerr << "ASSERT : " << msg << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    inline auto FATAL(const std::string &msg) noexcept {
        std::cerr << "FATAL : " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}
