//
// Created by jewoo on 2025-03-07.
//

#pragma once
#include <string>
#include <fstream>
#include <cstdio>


#include "macros.h"
#include "lf_queue.h"
#include "thread_utils.h"
#include "time_utils.h"

namespace LL::Common {
    constexpr size_t LOG_QUEUE_SIZE = 8 * 1024 * 1024;

    enum class LogType: int8_t {
        CHAR = 0,
        INTEGER = 1,
        LONG_INTEGER = 2,
        LONG_LONG_INTEGER = 3,
        UNSIGNED_INTEGER = 4,
        UNSIGNED_LONG_INTEGER = 5,
        UNSIGNED_LONG_LONG_INTEGER = 6,
        FLOAT = 7,
        DOUBLE = 8,
    };

    struct LogElement {
        LogType type_ = LogType::CHAR;

        union {
            char char_;
            int int_;
            long long_;
            long long llong_;
            unsigned u_;
            unsigned long ulong_;
            unsigned long long ullong_;
            float float_;
            double double_;
        } union_;
    };

    class Logger final {
    public:
    private:
        const std::string file_name_;
        std::ofstream file_;
        LFQueue<LogElement> queue_;
        std::atomic<bool> running_ = {true};
        std::thread *logger_thread_ = {nullptr};
    };
}
