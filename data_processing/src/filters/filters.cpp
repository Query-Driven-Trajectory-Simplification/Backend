#include "backend/data/filters/filters.hpp"
#include "backend/data/filters/boundingbox.hpp"
#include "backend/data/filters/mintrip.hpp"
#include "backend/data/filters/staypoint.hpp"
#include "backend/data/filters/timegap.hpp"

#include <format>

namespace filters {

namespace {

void record(StageStats& stage, const std::vector<geolife::Segment>& segments) {
    for (const auto& segment : segments) {
        stage.points += segment.size();
    }
    stage.segments += segments.size();
}

double ratio(std::size_t part, std::size_t whole) {
    return whole == 0 ? 0.0 : static_cast<double>(part) / static_cast<double>(whole);
}

}

std::vector<geolife::Segment> apply_filters(const geolife::Segment& points, FilterStats& stats) {
    std::vector<geolife::Segment> segments = {points};
    record(stats.stages[0], segments);

    segments = split_by_bbox(segments, kBeijing);
    record(stats.stages[1], segments);

    segments = split_by_time_gap(segments, kMaxGapSeconds);
    record(stats.stages[2], segments);

    segments = split_by_stay_points(segments, kDefaultStay);
    record(stats.stages[3], segments);

    segments = drop_short_trips(segments, kDefaultMinTrip);
    record(stats.stages[4], segments);

    return segments;
}

void print_stats(std::ostream& out, const FilterStats& stats) {
    out << std::format("{:<14}{:>12}{:>11}{:>12}{:>10}{:>10}{:>9}{:>10}\n",
                       "stage", "points", "% of input", "dropped", "% dropped", "segments", "x prev", "pts/seg");

    const StageStats& input = stats.stages.front();
    for (std::size_t i = 0; i < stats.stages.size(); ++i) {
        const StageStats& stage = stats.stages[i];
        out << std::format("{:<14}{:>12}{:>10.1f}%", stage.name, stage.points, 100 * ratio(stage.points, input.points));

        // Dropped points and segment growth are relative to the previous stage.
        if (i == 0) {
            out << std::format("{:>12}{:>10}{:>10}{:>9}", "-", "-", stage.segments, "-");
        } else {
            const StageStats& prev = stats.stages[i - 1];
            const std::size_t dropped = prev.points - stage.points;
            out << std::format("{:>12}{:>9.1f}%{:>10}{:>9.2f}",
                               dropped, 100 * ratio(dropped, prev.points),
                               stage.segments, ratio(stage.segments, prev.segments));
        }
        out << std::format("{:>10.1f}\n", ratio(stage.points, stage.segments));
    }
}

}
