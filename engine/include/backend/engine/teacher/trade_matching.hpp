#pragma once

#include <cstddef>
#include <span>
#include <utility>

namespace teacher::detail {

// Sellers are sorted by descending gain. Only used sellers can permanently advance first;
// a seller excluded for being this buyer remains available to subsequent buyers.
inline std::size_t find_profitable_seller(std::span<const std::pair<double, std::size_t>> sellers,
                                          std::span<const char> used, std::size_t &first,
                                          std::size_t buyer, double buy_gain, double min_gain) {
    while (first < sellers.size() && used[sellers[first].second])
        ++first;
    auto candidate = first;
    while (candidate < sellers.size() &&
           (used[sellers[candidate].second] || sellers[candidate].second == buyer))
        ++candidate;
    if (candidate == sellers.size() || buy_gain + sellers[candidate].first <= min_gain)
        return sellers.size();
    return candidate;
}

} // namespace teacher::detail
