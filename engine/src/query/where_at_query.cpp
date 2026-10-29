#include "backend/engine/query/where_at_query.hpp"
#include "backend/engine/core/interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace query {

namespace {

struct EligibleInterval {
    double start_time;
    double cumulative_duration;
};

void validate_config(const WhereAtConfig &config) {
    if (config.probes_per_trajectory == 0)
        throw std::invalid_argument("where-at: probes_per_trajectory must be positive");
    if (!std::isfinite(config.max_gap_seconds) || config.max_gap_seconds <= 0)
        throw std::invalid_argument("where-at: max_gap_seconds must be finite and positive");
}

std::vector<EligibleInterval> eligible_intervals(const core::TrajectoryView &trajectory,
                                                 double max_gap_seconds) {
    std::vector<EligibleInterval> intervals;
    intervals.reserve(trajectory.size() > 0 ? trajectory.size() - 1 : 0);
    double duration = 0;
    for (std::size_t j = 1; j < trajectory.size(); ++j) {
        const double start = static_cast<double>(trajectory.t[j - 1]);
        const double gap = static_cast<double>(trajectory.t[j]) - start;
        if (gap > max_gap_seconds)
            continue;

        duration += gap;
        intervals.push_back({start, duration});
    }
    return intervals;
}

double sample_time(const std::vector<EligibleInterval> &intervals, std::size_t probe_index,
                   std::size_t probe_count, std::mt19937_64 &rng) {
    const double duration = intervals.back().cumulative_duration;
    // This is somewhat crazy. std::generate_canonical takes the type of value to generate & a
    // precision value. The precision indicates the bits of entropy. For a double with a mantissa of
    // 53 bits, that value is 53. Adding it to probe_index picks a position within that probe's time
    // portion.
    const double random = std::generate_canonical<double, 53>(rng);
    const double fraction =
        (static_cast<double>(probe_index) + random) / static_cast<double>(probe_count);

    const double elapsed = std::min(duration * fraction, std::nextafter(duration, 0.0));
    const auto interval =
        std::ranges::upper_bound(intervals, elapsed, {}, &EligibleInterval::cumulative_duration);
    const double previous_duration =
        interval == intervals.begin() ? 0 : (interval - 1)->cumulative_duration;

    return interval->start_time + (elapsed - previous_duration);
}

} // namespace

WhereAtWorkload generate_where_at_queries(const core::TrajectoryDb &uncompressed_database,
                                          const WhereAtConfig &config, std::uint64_t seed) {
    validate_config(config);

    WhereAtWorkload result;
    std::size_t eligible = 0;
    for (std::size_t i = 0; i < uncompressed_database.size(); ++i) {
        const auto times = uncompressed_database.trajectory(i).t;
        eligible +=
            std::adjacent_find(times.begin(), times.end(), [&](auto a, auto b) {
                return static_cast<double>(b) - static_cast<double>(a) <= config.max_gap_seconds;
            }) != times.end();
    }
    if (eligible > result.queries.max_size() / config.probes_per_trajectory)
        throw std::length_error("where-at: too many probes");
    result.queries.reserve(eligible * config.probes_per_trajectory);
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < uncompressed_database.size(); i++) {
        const auto trajectory = uncompressed_database.trajectory(i);
        const auto intervals = eligible_intervals(trajectory, config.max_gap_seconds);
        if (intervals.empty()) {
            result.skipped_trajectories += 1;
            continue;
        }
        for (std::size_t probe_index = 0; probe_index < config.probes_per_trajectory;
             probe_index++) {
            const double t = sample_time(intervals, probe_index, config.probes_per_trajectory, rng);
            result.queries.push_back(
                {i, trajectory.trajectory_id, t, core::position_at(trajectory, t)});
        }
    }
    return result;
}

} // namespace query
