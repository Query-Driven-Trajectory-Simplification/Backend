#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/kendalls_tau.hpp"
#include "backend/engine/evaluator/range_score.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace evaluator {

Evaluator::Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified,
                     std::uint64_t seed)
    : m_original(original), m_simplified(simplified), m_range_query_seed(seed) {
    if (original.size() != simplified.size())
        throw std::invalid_argument(
            "Evaluator: original and simplified hold different trajectories");
}

Scoretable Evaluator::evaluate() {
    return Scoretable{
        .knn_score = get_knn_score(),
        .range_score = get_range_score(),
        .where_at_score = get_where_at_score(),
    };
}

double Evaluator::get_knn_score() {
    knn_score(m_knn_neighbours);
    return m_knn_score;
}

double Evaluator::get_range_score() {
    range_score();
    return m_range_score;
}

double Evaluator::get_where_at_score() {
    where_at_score();
    return m_where_at_score;
}

void Evaluator::set_knn_neighbours(std::size_t k) { m_knn_neighbours = k; }

void Evaluator::set_range_query_params(query::RangeQueryParams params) {
    m_range_query_params = params;
}

void Evaluator::knn_score(std::size_t k) {
    const core::TrajectoryDb queries = query::generate_knn_queries(m_original);

    double sum = 0.0;
    std::size_t scored = 0;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto trajectory = queries.trajectory(i);
        const double tau = knn::kendalls_tau(query::knn(m_original, trajectory, k),
                                             query::knn(m_simplified, trajectory, k));
        if (!std::isnan(tau)) {
            sum += tau;
            ++scored;
        }
    }
    m_knn_score = scored == 0 ? std::numeric_limits<double>::quiet_NaN()
                              : sum / static_cast<double>(scored);
}

void Evaluator::range_score() {
    const query::RangeQuerySampler sampler(m_original);
    const auto queries = sampler.sample(m_range_query_seed, m_range_query_params);
    if (queries.empty()) {
        m_range_score = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    const query::RangeQuerySampler simplified_sampler(m_simplified);
    double sum = 0.0;
    std::size_t scored = 0;
    for (const auto &range_query : queries) {
        const auto original_results = range_query.results;
        if (original_results == 0) {
            continue;
        }

        const auto simplified_results = simplified_sampler.count_results(range_query);
        sum += range::query_score(original_results, simplified_results);
        ++scored;
    }

    m_range_score = scored == 0 ? std::numeric_limits<double>::quiet_NaN()
                                : sum / static_cast<double>(scored);
}

void Evaluator::where_at_score() {
    m_where_at_score = std::numeric_limits<double>::quiet_NaN();
}

} // namespace evaluator
