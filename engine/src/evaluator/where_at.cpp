#include "backend/engine/evaluator/where_at.hpp"
#include "backend/engine/core/interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace evaluator {

namespace {

struct TripScore {
    double hits_within_radius = 0;
    double average_hit_sum = 0;
    std::size_t query_count = 0;
};

double query_error(const core::TrajectoryDb &compressed, const query::WhereAtQuery &probe) {
    if (probe.trajectory_index >= compressed.size())
        throw std::invalid_argument("where-at: missing trajectory");

    const auto trajectory = compressed.trajectory(probe.trajectory_index);
    if (trajectory.trajectory_id != probe.trajectory_id || trajectory.size() == 0)
        throw std::invalid_argument("where-at: mismatched or empty trajectory");

    if (!std::isfinite(probe.t) || probe.t < static_cast<double>(trajectory.t.front()) ||
        probe.t > static_cast<double>(trajectory.t.back()) || !std::isfinite(probe.reference.x) ||
        !std::isfinite(probe.reference.y))
        throw std::invalid_argument("where-at: invalid query position or time");

    const auto position = core::position_at(trajectory, probe.t);
    const double error = core::distance(position, probe.reference);
    if (!std::isfinite(error))
        throw std::invalid_argument("where-at: non-finite interpolated error");
    return error;
}

void add_error(TripScore &trip, double error, const WhereAtScoringConfig &config) {
    if (error <= config.hit_radius_m)
        trip.hits_within_radius += 1;
    const double tolerance_range = config.max_tolerance_m - config.min_tolerance_m;
    trip.average_hit_sum +=
        std::clamp((config.max_tolerance_m - error) / tolerance_range, 0.0, 1.0);
    trip.query_count += 1;
}

WhereAtScores average_trip_scores(std::span<const TripScore> trips) {
    double within_radius_sum = 0;
    double average_hit_sum = 0;
    std::size_t evaluated_trajectories = 0;

    for (const auto &trip : trips) {
        if (trip.query_count == 0)
            continue;

        const double query_count = static_cast<double>(trip.query_count);
        within_radius_sum += trip.hits_within_radius / query_count;
        average_hit_sum += trip.average_hit_sum / query_count;
        evaluated_trajectories += 1;
    }

    if (evaluated_trajectories == 0)
        return {};

    const double trip_count = static_cast<double>(evaluated_trajectories);
    return {within_radius_sum / trip_count, average_hit_sum / trip_count};
}

} // namespace

WhereAtScores evaluate_where_at(const core::TrajectoryDb &compressed,
                                std::span<const query::WhereAtQuery> queries,
                                const WhereAtScoringConfig &config) {
    if (!std::isfinite(config.hit_radius_m) || config.hit_radius_m < 0 ||
        !std::isfinite(config.min_tolerance_m) || config.min_tolerance_m < 0 ||
        !std::isfinite(config.max_tolerance_m) ||
        config.max_tolerance_m <= config.min_tolerance_m)
        throw std::invalid_argument("where-at: invalid scoring distances");

    std::vector<TripScore> trips(compressed.size());
    for (const auto &probe : queries) {
        const double error = query_error(compressed, probe);
        add_error(trips[probe.trajectory_index], error, config);
    }
    return average_trip_scores(trips);
}

} // namespace evaluator
