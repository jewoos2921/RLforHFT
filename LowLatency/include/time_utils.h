//
// Created by jewoo on 2025-03-07.
//

#pragma once
#include <string>
#include <chrono>
#include <ctime>

namespace LL::Common {
    using Nanos = int64_t;

    constexpr Nanos NANOS_TO_MICROS{1000};
    constexpr Nanos MICROS_TO_MILLIS{1000};
    constexpr Nanos MILLIS_TO_SECS{1000};
    constexpr Nanos NANOS_TO_MILLS{NANOS_TO_MICROS * MICROS_TO_MILLIS};
    constexpr Nanos NANOS_TO_SECS{NANOS_TO_MILLS * MILLIS_TO_SECS};

    inline auto getCurrentNanos() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    inline auto &getCurrentTimeStr(std::string *time_str) {
        const auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        time_str->assign(ctime(&time));
        if (!time_str->empty())
            time_str->at(time_str->length() - 1) = '\0';
        return *time_str;
    }
}
