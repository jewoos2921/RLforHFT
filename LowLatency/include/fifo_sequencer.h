//
// Created by jewoo on 2025-03-22.
//

#pragma once

#include "thread_utils.h"
#include "macros.h"


#include "client_request.h"

namespace LL::Exchange {
    constexpr size_t ME_MAX_PENDING_REQUESTS = 1024;

    class FIFOSequencer {
    public:
    private:
        ClientRequestLFQueue *incoming_requests_ = nullptr;
        std::string time_str_;
        Logger *logger_ = nullptr;
    };
}
