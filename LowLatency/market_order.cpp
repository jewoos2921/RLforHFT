//
// Created by jewoo on 2025-03-23.
//

#include "market_order.h"

namespace LL::Trading {
    auto MarketOrder::toString() const -> std::string {
        std::stringstream ss;
        ss << "MarketOrder[" << "oid:" << orderIdToString(order_id_) << " "
                << "side: " << sideToString(side_) << " "
                << "price: " << priceToString(price_) << " "
                << "qty: " << quantityToString(qty_) << " "
                << "prio:" << priorityToString(priority_) << " "
                << "prev:" << orderIdToString(prev_order_ ? prev_order_->order_id_ : OrderId_INVALID) << " "
                << "next:" << orderIdToString(next_order_ ? next_order_->order_id_ : OrderId_INVALID) << "]";
        return ss.str();
    }
}
