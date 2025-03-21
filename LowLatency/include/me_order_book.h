//
// Created by jewoo on 2025-03-08.
//

#pragma once
#include "types.h"

#include "mem_pool.h"
#include "logging.h"

#include "client_response.h"
#include "market_update.h"

#include "me_order.h"

using namespace LL::Common;

namespace LL::Exchange {
    class MatchingEngine;

    class MEOrderBook final {
    public:
    private:
        TickerId ticker_id_ = TickerId_INVALID;
        MatchingEngine *matching_engine_ = nullptr;
        ClientOrderHashMap cid_oid_to_order_;
        MemPool<MEOrdersAtPrice Orders_at_price_pool_;

    };
}
