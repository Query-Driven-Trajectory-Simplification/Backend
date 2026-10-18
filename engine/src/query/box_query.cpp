#include "backend/engine/query/box_query.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

namespace query {

namespace {

constexpr std::int64_t kDaySeconds = 24 * 60 * 60;
constexpr double kKmPerDegree = 111.195;
constexpr std::size_t kAttemptsPerQuery = 1000;

} // namespace

BoxQuerySampler::BoxQuerySampler(const core::TrajectoryDb &db) : m_db(db) {
    const auto t = db.all_t();
    if (t.empty()) {
        return;
    }
    m_t_min = *std::min_element(t.begin(), t.end());
    const auto t_max = *std::max_element(t.begin(), t.end());
    const auto days = static_cast<std::size_t>((t_max - m_t_min) / kDaySeconds) + 1;

    m_day_start.assign(days + 1, 0);
    for (const auto value : t) {
        ++m_day_start[static_cast<std::size_t>((value - m_t_min) / kDaySeconds) + 1];
    }
    for (std::size_t d = 0; d < days; ++d) {
        m_day_start[d + 1] += m_day_start[d];
    }
    m_by_day.resize(t.size());
    auto next = m_day_start;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const auto day = static_cast<std::size_t>((t[i] - m_t_min) / kDaySeconds);
        m_by_day[next[day]++] = static_cast<std::uint32_t>(i);
    }
}

std::uint32_t BoxQuerySampler::count_results(const BoxQuery &box) const {
    if (m_day_start.empty() || box.t_end < m_t_min) {
        return 0;
    }
    const std::int64_t last_day = static_cast<std::int64_t>(m_day_start.size()) - 2;
    const std::int64_t first = box.t_start <= m_t_min ? 0 : (box.t_start - m_t_min) / kDaySeconds;
    const std::int64_t last = std::min((box.t_end - m_t_min) / kDaySeconds, last_day);

    const auto t = m_db.all_t();
    const auto lat = m_db.all_lat();
    const auto lon = m_db.all_lon();
    const auto offsets = m_db.offsets();

    std::vector<std::size_t> trajectories;
    for (auto day = first; day <= last; ++day) {
        const auto d = static_cast<std::size_t>(day);
        for (auto k = m_day_start[d]; k < m_day_start[d + 1]; ++k) {
            const auto i = m_by_day[k];
            if (t[i] < box.t_start || t[i] > box.t_end || lat[i] < box.min_lat ||
                lat[i] > box.max_lat || lon[i] < box.min_lon || lon[i] > box.max_lon) {
                continue;
            }
            trajectories.push_back(
                static_cast<std::size_t>(std::upper_bound(offsets.begin(), offsets.end(), i) -
                                         offsets.begin()) -
                1);
        }
    }
    std::sort(trajectories.begin(), trajectories.end());
    trajectories.erase(std::unique(trajectories.begin(), trajectories.end()), trajectories.end());
    return static_cast<std::uint32_t>(trajectories.size());
}

std::vector<BoxQuery> BoxQuerySampler::sample(std::uint64_t seed,
                                              const BoxQueryParams &params) const {
    std::vector<BoxQuery> boxes;
    if (m_db.size() == 0 || m_db.num_points() == 0) {
        return boxes;
    }
    boxes.reserve(params.count);

    const auto offsets = m_db.offsets();
    const auto t = m_db.all_t();
    const auto lat = m_db.all_lat();
    const auto lon = m_db.all_lon();
    const double half_lat = params.box_km / 2 / kKmPerDegree;

    std::mt19937_64 rng(seed);
    const std::size_t max_attempts = params.count * kAttemptsPerQuery;
    for (std::size_t attempt = 0; attempt < max_attempts && boxes.size() < params.count;
         ++attempt) {
        const auto trajectory = static_cast<std::size_t>(rng() % m_db.size());
        const auto length = offsets[trajectory + 1] - offsets[trajectory];
        if (length == 0) {
            continue;
        }
        const auto i = static_cast<std::size_t>(offsets[trajectory] + rng() % length);

        const double half_lon = half_lat / std::cos(lat[i] * std::numbers::pi / 180.0);
        BoxQuery box{
            .trajectory_id = m_db.trajectory(trajectory).trajectory_id,
            .lat = lat[i],
            .lon = lon[i],
            .t = t[i],
            .min_lat = lat[i] - half_lat,
            .max_lat = lat[i] + half_lat,
            .min_lon = lon[i] - half_lon,
            .max_lon = lon[i] + half_lon,
            .t_start = t[i] - params.window_seconds / 2,
            .t_end = t[i] + params.window_seconds / 2,
            .results = 0,
        };
        box.results = count_results(box);

        if (box.results < params.min_results) {
            continue;
        }
        if (box.results < params.full_accept_at) {
            const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;
            if (u * params.full_accept_at >= box.results) {
                continue;
            }
        }
        boxes.push_back(box);
    }
    return boxes;
}

} // namespace query
