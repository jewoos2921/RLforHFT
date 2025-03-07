//
// Created by jewoo on 2025-03-07.
//

#pragma once

#include <functional>
#include "socket_utils.h"
#include "logging.h"

namespace LL::Common {
    constexpr size_t TCPBufferSize = 64 * 1024 * 1024;

    struct TCPSocket {
        explicit TCPSocket(Logger &logger) : logger_(logger) {
            outbound_data_.resize(TCPBufferSize);
            inbound_data_.resize(TCPBufferSize);
        }

        auto connect(const std::string &ip,
                     const std::string &iface,
                     int port, bool is_listening) -> int;

        TCPSocket() = delete;

        TCPSocket(const TCPSocket &) = delete;

        TCPSocket(const TCPSocket &&) = delete;

        TCPSocket &operator=(const TCPSocket &) = delete;

        TCPSocket &operator=(const TCPSocket &&) = delete;

        auto send(const void *data, size_t len) noexcept -> void;

        auto sendAndRecv() noexcept -> bool;

        int socket_fd_{-1};


        std::vector<char> outbound_data_;
        std::vector<char> inbound_data_;
        size_t next_send_valid_index_{0};
        size_t next_recv_valid_index_{0};

        sockaddr_in socket_attrib_{};
        std::function<void(TCPSocket *s, Nanos rx_time)> recv_callback_ = nullptr;
        std::string time_str_;
        Logger &logger_;
    };
}
