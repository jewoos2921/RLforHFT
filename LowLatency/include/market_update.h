//
// Created by jewoo on 2025-03-21.
//

#pragma once

#include <sstream>

#include "types.h"

using namespace LL::Common;

namespace LL::Exchange {
    enum class MarketUpdateType: uint8_t {
        INVALID = 0,
        CLEAR = 1,
        ADD = 2,
        MODIFY = 3,
        CANCEL = 4,
        TRADE = 5,
        SNAPSHOT_START = 6,
        SNAPSHOT_END = 7
    };

    inline std::string marketUpdateTypeToString(MarketUpdateType type) {
        switch (type) {
            case MarketUpdateType::CLEAR:
                return "CLEAR";
            case MarketUpdateType::ADD:
                return "ADD";
            case MarketUpdateType::MODIFY:
                return "MODIFY";
            case MarketUpdateType::CANCEL:
                return "CANCEL";
            case MarketUpdateType::TRADE:
                return "TRADE";
            case MarketUpdateType::SNAPSHOT_START:
                return "SNAPSHOT_START";
            case MarketUpdateType::SNAPSHOT_END:
                return "SNAPSHOT_END";
            case MarketUpdateType::INVALID:
                return "INVALID";
        }
        return "UNKNOWN";
    }

#pragma pack(push, 1)
    struct MEMarketUpdate {
        MarketUpdateType type_ = MarketUpdateType::INVALID;

        OrderId orderId_ = OrderId_INVALID;
        TickerId tickerId_ = TickerId_INVALID;
        Side side_ = Side::INVALID;
        Price price_ = Price_INVALID;
        Quantity quantity_ = Quantity_INVALID;
        Priority priority_ = Priority_INVALID;

        auto toString() const {
            std::stringstream ss;
            ss << "MEMarketUpdate"
                    << "["
                    << " type: " << marketUpdateTypeToString(type_)
                    << "ticker: " << tickerIdToString(tickerId_)
                    << "oid: " << orderIdToString(orderId_)
                    << "side: " << sideToString(side_)
                    << "quantity: " << quantityToString(quantity_)
                    << "price: " << priceToString(price_)
                    << "priority: " << priorityToString(priority_)
                    << "]";
            return ss.str();
        }
    };

    struct MDPMarketUpdate {
        size_t seq_num_{0};
        MEMarketUpdate me_market_update_;

        auto toString() const {
            std::stringstream ss;
            ss << "MDPMarketUpdate" << " [" << " seq: " << seq_num_
                    << " " << me_market_update_.toString() << "]";
            return ss.str();
        }
    };
#pragma  pack(pop)

    using MEMarketUpdateLFQueue = LFQueue<MEMarketUpdate>;
    using MDPMarketUpdateLFQueue = LFQueue<MDPMarketUpdate>;
}
