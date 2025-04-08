//
// Created by jewoo on 2025-03-22.
//

#pragma once

#include <functional>

#include "thread_utils.h"
#include "macros.h"
#include "tcp_server.h"

#include "client_request.h"
#include "client_response.h"
#include "fifo_sequencer.h"

namespace LL::Exchange {
    class OrderServer {
    public:
    private:
        const std::string iface_;
        const int port_ = 0;

        ClientResponseLFQueue *outgoing_response_ = nullptr;
        volatile bool run_ = false;
        std::string time_str_;
        Logger logger_;
    };
}
