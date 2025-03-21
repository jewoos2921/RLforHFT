//
// Created by jewoo on 2025-03-21.
//

#pragma once

#include <sstream>
#include "types.h"
#include "lf_queue.h"

using namespace LL::Common;

namespace LL::Exchange {
#pragma pack(push, 1)
    enum class ClientRequestType : uint8_t {
        INVALID = 0,
        NEW = 1,
        CANCEL = 2,
    };

    inline std::string clientRequestTypeToString(ClientRequestType type) {
        switch (type) {
            case ClientRequestType::NEW:
                return "NEW";
            case ClientRequestType::CANCEL:
                return "CANCEL";
            case ClientRequestType::INVALID:
                return "INVALID";
        }
        return "UNKNOWN";
    }

    struct MEClientRequest {
        ClientRequestType type_ = ClientRequestType::INVALID;

        ClientId client_id_ = ClientId_INVALID;
        TickerId tickerId_ = TickerId_INVALID;
        OrderId orderId_ = OrderId_INVALID;
        Side side_ = Side::INVALID;
        Price price_ = Price_INVALID;
        Quantity quantity_ = Quantity_INVALID;

        auto toString() const {
            std::stringstream ss;
            ss << "MEClientRequest"
                    << "["
                    << "type: " << clientRequestTypeToString(type_)
                    << "client: " << clientIdToString(client_id_)
                    << "ticker: " << tickerIdToString(tickerId_)
                    << "oid: " << orderIdToString(orderId_)
                    << "side: " << sideToString(side_)
                    << "qty: " << quantityToString(quantity_)
                    << "price: " << priceToString(price_)
                    << "]";
            return ss.str();
        }
    };
#pragma  pack(pop)

    using ClientRequestLFQueue = LFQueue<MEClientRequest>;
}
