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
        auto flushQueue() const {
            while (running_) {
                for (auto next = queue_.getNextToRead();
                     queue_.size() && next; next = queue_.getNextToRead()) {
                    switch (next->type_) {
                        case LogType::CHAR: {
                            file_ << next->union_.char_;
                            break;
                        }

                        case LogType::INTEGER: {
                            file_ << next->union_.int_;
                            break;
                        }

                        case LogType::LONG_INTEGER: {
                            file_ << next->union_.long_;
                            break;
                        }

                        case LogType::LONG_LONG_INTEGER: {
                            file_ << next->union_.llong_;
                            break;
                        }

                        case LogType::UNSIGNED_INTEGER: {
                            file_ << next->union_.u_;
                            break;
                        }

                        case LogType::UNSIGNED_LONG_INTEGER: {
                            file_ << next->union_.ulong_;
                            break;
                        }

                        case LogType::UNSIGNED_LONG_LONG_INTEGER: {
                            file_ << next->union_.ullong_;
                            break;
                        }

                        case LogType::FLOAT: {
                            file_ << next->union_.float_;
                            break;
                        }

                        case LogType::DOUBLE: {
                            file_ << next->union_.double_;
                            break;
                        }

                        default: {
                            ASSERT(false, "Unknown LogType");
                            break;
                        }
                    }
                    queue_.updateReadIndex();
                }
                file_.flush();

                using namespace std::chrono_literals;
                std::this_thread::sleep_for(10ms);
            }
        }

        explicit Logger(const std::string &file_name) : file_name_(file_name), queue_(LOG_QUEUE_SIZE) {
            file_.open(file_name);
            ASSERT(file_.is_open(), "Could not open log file:" + file_name);
            logger_thread_ = createAndStartThread(-1, "Common/Logger " +
                                                      file_name_, [this]() { flushQueue(); });
            ASSERT(logger_thread_ != nullptr, "Could not create logger thread");;
        }

        ~Logger() {
            std::string time_str;
            std::cerr << getCurrentTimeStr(&time_str) << "Logger is shutting down" << file_name_ << std::endl;

            while (queue_.size()) {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(1s);
            }
            running_ = false;
            logger_thread_->join();

            file_.close();
            std::cerr << getCurrentTimeStr(&time_str) << " Logger for " << file_name_ << " is done" << std::endl;
        }

        auto pushValue(const LogElement &log_element) noexcept {
            *(queue_.getNextToWriteTo()) = log_element;
            queue_.updateWriteIndex();
        }

        auto pushValue(const char value) noexcept {
            pushValue(LogElement{LogType::CHAR, .union_ = {.char_ = value}});
        }

        auto pushValue(const int value) noexcept {
            pushValue(LogElement{LogType::INTEGER, .union_ = {.int_ = value}});
        }

        auto pushValue(const long value) noexcept {
            pushValue(LogElement{LogType::LONG_INTEGER, .union_ = {.long_ = value}});
        }

        auto pushValue(const long long value) noexcept {
            pushValue(LogElement{LogType::LONG_LONG_INTEGER, .union_ = {.llong_ = value}});
        }

        auto pushValue(const unsigned value) noexcept {
            pushValue(LogElement{LogType::UNSIGNED_INTEGER, .union_ = {.u_ = value}});
        }

        auto pushValue(const unsigned long value) noexcept {
            pushValue(LogElement{LogType::UNSIGNED_LONG_INTEGER, .union_ = {.ulong_ = value}});
        }

        auto pushValue(const unsigned long long value) noexcept {
            pushValue(LogElement{LogType::UNSIGNED_LONG_LONG_INTEGER, .union_ = {.ullong_ = value}});
        }

        auto pushValue(const float value) noexcept {
            pushValue(LogElement{LogType::FLOAT, .union_ = {.float_ = value}});
        }

        auto pushValue(const double value) noexcept {
            pushValue(LogElement{LogType::DOUBLE, .union_ = {.double_ = value}});
        }

        auto pushValue(const char *value) noexcept {
            while (*value) {
                pushValue(*value);
                ++value;
            }
        }

        auto pushValue(const std::string &value) noexcept {
            pushValue(value.c_str());
        }

        template<class T, typename... A>
        auto log(const char *s, const T &value, A... args) noexcept {
            while (*s) {
                if (*s == '%') {
                    if (UNLIKELY(*(s + 1) == '%'))
                        ++s;
                    else {
                        pushValue(value);
                        log(s + 1, args...);
                        return;
                    }
                }
                pushValue(*s);
            }
            FATAL("extra arguments provided to log()");
        }

        auto log(const char *s) noexcept {
            while (*s) {
                if (*s == '%') {
                    if (UNLIKELY(*(s + 1) == '%'))
                        ++s;
                    else
                        FATAL("no arguments provided to log()");
                }
                pushValue(*s++);
            }
        }

        Logger() = delete;

        Logger(const Logger &) = delete;

        Logger(const Logger &&) = delete;

        Logger &operator=(const Logger &) = delete;

        Logger &operator=(const Logger &&) = delete;

    private:
        const std::string file_name_;
        std::ofstream file_;

        LFQueue<LogElement> queue_;
        std::atomic<bool> running_ = {true};
        std::thread *logger_thread_ = {nullptr};
    };
}
