#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace query {

struct RangeQueryParams {
    double box_km = 2.0;
    std::int64_t window_seconds = 7 * 24 * 60 * 60;
    std::uint32_t min_results = 5;
    std::uint32_t full_accept_at = 20;
    std::size_t count = 100;
};

struct RangeQuery {
    std::int32_t trajectory_id;
    double lat;
    double lon;
    std::int64_t t;
    double min_lat;
    double max_lat;
    double min_lon;
    double max_lon;
    std::int64_t t_start;
    std::int64_t t_end;
    std::uint32_t results;
};

class RangeQuerySampler {
  public:
    explicit RangeQuerySampler(const core::TrajectoryDb &db);

    std::uint32_t count_results(const RangeQuery &box) const;

    std::vector<RangeQuery> sample(std::uint64_t seed, const RangeQueryParams &params) const;

  private:
    const core::TrajectoryDb &m_db;
    std::int64_t m_t_min = 0;
    std::vector<std::uint32_t> m_day_start;
    std::vector<std::uint32_t> m_by_day;
};

} // namespace query
