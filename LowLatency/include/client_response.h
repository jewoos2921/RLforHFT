//
// Created by jewoo on 2025-03-21.
//

#pragma once

#include <sstream>
#include "types.h"
#include "lf_queue.h"

using namespace LL::Common;

namespace LL::Exchange {
    enum class ClientResponseType : uint8_t {
        INVALID = 0,
        ACCEPTED = 1,
        CANCELED = 2,
        FILLED = 3,
        CANCEL_REJECTED = 4,
    };

    inline std::string clientResponseTypeToString(ClientResponseType type) {
        switch (type) {
            case ClientResponseType::ACCEPTED:
                return "ACCEPTED";
            case ClientResponseType::CANCELED:
                return "CANCELED";
            case ClientResponseType::FILLED:
                return "FILLED";
            case ClientResponseType::CANCEL_REJECTED:
                return "CANCEL_REJECTED";
            case ClientResponseType::INVALID:
                return "INVALID";
        }
        return "UNKNOWN";
    }
#pragma pack(push, 1)
    struct MEClientResponse {
        ClientResponseType type_ = ClientResponseType::INVALID;
        ClientId client_id_ = ClientId_INVALID;
        TickerId ticker_id_ = TickerId_INVALID;
        OrderId client_order_id_ = OrderId_INVALID;
        OrderId market_order_id_ = OrderId_INVALID;
        Side side_ = Side::INVALID;
        Price price_ = Price_INVALID;
        Quantity exec_qty_ = Quantity_INVALID;
        Quantity leaves_qty_ = Quantity_INVALID;


        auto toString() const {
            std::stringstream ss;
            ss << "MEClientResponse"
                    << " ["
                    << "type:" << clientResponseTypeToString(type_)
                    << " client:" << clientIdToString(client_id_)
                    << " ticker:" << tickerIdToString(ticker_id_)
                    << " coid:" << orderIdToString(client_order_id_)
                    << " moid:" << orderIdToString(market_order_id_)
                    << " side:" << sideToString(side_)
                    << " exec_qty:" << quantityToString(exec_qty_)
                    << " leaves_qty:" << quantityToString(leaves_qty_)
                    << " price:" << priceToString(price_)
                    << "]";
            return ss.str();
        }
    };

    struct OMClientResponse {
        size_t seq_num_{0};
        MEClientResponse me_client_response_;

        auto toString() const {
            std::stringstream ss;
            ss << "OMClientResponse" << " ["
                    << "seq: " << seq_num_ << " "
                    << me_client_response_.toString() << "]";
            return ss.str();
        }
    };
#pragma  pack(pop)

    using ClientResponseLFQueue = LFQueue<MEClientResponse>;
}
