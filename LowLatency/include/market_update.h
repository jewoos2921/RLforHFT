//
// Created by jewoo on 2025-03-21.
//

#pragma once

#include <sstream>
#include "types.h"

using namespace LL::Common;

namespace LL::Exchange {
#pragma pack(push, 1)
    enum class MarketUpdateType: uint8_t {
        INVALID = 0,
        ADD = 1,
        MODIFY = 2,
        CANCEL = 3,
        TRADE = 4
    };

    inline std::string marketUpdateTypeToString(MarketUpdateType type) {
        switch (type) {
            case MarketUpdateType::ADD:
                return "ADD";
            case MarketUpdateType::MODIFY:
                return "MODIFY";
            case MarketUpdateType::CANCEL:
                return "CANCEL";
            case MarketUpdateType::TRADE:
                return "TRADE";
            case MarketUpdateType::INVALID:
                return "INVALID";
        }
        return "UNKNOWN";
    }

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
                    << "price: " << priceToString(price_)
                    << "quantity: " << quantityToString(quantity_)
                    << "priority: " << priorityToString(priority_)
                    << "]";
            return ss.str();
        }
    };
#pragma  pack(pop)

    using MEMarketUpdateLFQueue = LFQueue<MEMarketUpdate>;
}
