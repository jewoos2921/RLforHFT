//
// Created by jewoo on 2025-03-21.
//

#pragma once

#include <sstream>

#include "types.h"
#include "lf_queue.h"

using namespace LL::Common;

namespace LL::Exchange {
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
#pragma pack(push, 1)
    struct MEClientRequest {
        ClientRequestType type_ = ClientRequestType::INVALID;

        ClientId client_id_ = ClientId_INVALID;
        TickerId ticker_id_ = TickerId_INVALID;
        OrderId order_id_ = OrderId_INVALID;
        Side side_ = Side::INVALID;
        Price price_ = Price_INVALID;
        Quantity quantity_ = Quantity_INVALID;

        auto toString() const {
            std::stringstream ss;
            ss << "MEClientRequest"
                    << "["
                    << "type: " << clientRequestTypeToString(type_)
                    << "client: " << clientIdToString(client_id_)
                    << "ticker: " << tickerIdToString(ticker_id_)
                    << "oid: " << orderIdToString(order_id_)
                    << "side: " << sideToString(side_)
                    << "qty: " << quantityToString(quantity_)
                    << "price: " << priceToString(price_)
                    << "]";
            return ss.str();
        }
    };

    struct OMClientRequest {
        size_t seq_num_{0};
        MEClientRequest me_client_request_;

        auto toString() const {
            std::stringstream ss;
            ss << "OMClientRequest" << " ["
                    << "seq: " << seq_num_ << " "
                    << me_client_request_.toString() << "]";
            return ss.str();
        }
    };
#pragma  pack(pop)

    using ClientRequestLFQueue = LFQueue<MEClientRequest>;
}
