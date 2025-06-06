//
// Created by jewoo on 2025-03-21.
//

#include "me_order.h"

namespace LL::Exchange {
    auto MEOrder::toString() const noexcept -> std::string {
        std::stringstream ss;
        ss << "MEOrder" << "["
                << "ticker:" << tickerIdToString(ticker_id_) << " "
                << "cid:" << clientIdToString(client_id_) << " "
                << "oid:" << orderIdToString(client_order_id_) << " "
                << "moid:" << orderIdToString(market_order_id_) << " "
                << "side:" << sideToString(side_) << " "
                << "price:" << priceToString(price_) << " "
                << "qty:" << quantityToString(quantity_) << " "
                << "prio:" << priorityToString(priority_) << " "
                << "prev:" << orderIdToString(prev_order_ ? prev_order_->market_order_id_ : OrderId_INVALID) << " "
                << "next:" << orderIdToString(next_order_ ? next_order_->market_order_id_ : OrderId_INVALID) << "]";

        return ss.str();
    }
}
