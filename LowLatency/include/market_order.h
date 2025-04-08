//
// Created by jewoo on 2025-03-23.
//

#pragma once
#include <array>
#include <sstream>
#include <logging.h>
#include <types.h>

using namespace LL::Common;

namespace LL::Trading {
    struct MarketOrder {
        OrderId order_id_ = OrderId_INVALID;
        Side side_ = Side::INVALID;
        Price price_ = Price_INVALID;
        Quantity qty_ = Quantity_INVALID;
        Priority priority_ = Priority_INVALID;

        MarketOrder *prev_order_ = nullptr;
        MarketOrder *next_order_ = nullptr;

        MarketOrder() = default;

        auto toString() const -> std::string;

        MarketOrder(OrderId order_id, Side side, Price price, Quantity qty, Priority priority,
                    MarketOrder *prev_order, MarketOrder *next_order)
            : order_id_(order_id), side_(side), price_(price), qty_(qty), priority_(priority),
              prev_order_(prev_order), next_order_(next_order) {
        }
    };

    using OrderHashMap = std::array<MarketOrder *, ME_MAX_ORDER_IDS>;

    struct MarketOrderAtPrice {
        Side side_ = Side::INVALID;
        Price price_ = Price_INVALID;

        MarketOrder *first_mkt_order_ = nullptr;
        MarketOrderAtPrice *prev_entry_ = nullptr;
        MarketOrderAtPrice *next_entry_ = nullptr;

        MarketOrderAtPrice() = default;

        MarketOrderAtPrice(Side side, Price price,
                           MarketOrder *first_mkt_order,
                           MarketOrderAtPrice *prev_order, MarketOrderAtPrice *next_order)
            : side_(side), price_(price), first_mkt_order_(first_mkt_order), prev_entry_(prev_order),
              next_entry_(next_order) {
        }

        auto toString() const {
            std::stringstream ss;
            ss << "MarketOrder["
                    << "side: " << sideToString(side_) << " "
                    << "price: " << priceToString(price_) << " "
                    << "first_mkt_order: " << (first_mkt_order_ ? first_mkt_order_->toString() : "null") << " "
                    << "prev:" << priceToString(prev_entry_ ? prev_entry_->price_ : Price_INVALID) << " "
                    << "next:" << priceToString(next_entry_ ? next_entry_->price_ : Price_INVALID) << "]";
            return ss.str();
        }
    };

    using OrdersAtPRiceHashMap = std::array<MarketOrderAtPrice *, ME_MAX_PRICE_LEVELS>;

    struct BBO {
        Price bid_price_ = Price_INVALID, ask_price_ = Price_INVALID;
        Quantity bid_qty_ = Quantity_INVALID, ask_qty_ = Quantity_INVALID;

        auto toString() const {
            std::stringstream ss;
            ss << "BBO{"
                    << quantityToString(bid_qty_) << "@" << priceToString(bid_price_)
                    << "X"
                    << priceToString(ask_price_) << "@" << quantityToString(ask_qty_)
                    << "}";
            return ss.str();
        }
    };
}
