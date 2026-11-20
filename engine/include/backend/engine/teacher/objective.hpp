#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace teacher {

// Slack when pruning by an upper bound, so rounding cannot drop a candidate that would tie
inline constexpr double kBoundSlack = 1e-15;

using core::KeptSets;

// A kept set's coordinates, owned. Views are made on demand, so copies and moves stay safe
class PreparedTrajectory {
  public:
    PreparedTrajectory(const core::TrajectoryView &original, std::span<const std::size_t> kept);
    // Invalidates earlier views
    void assign(const core::TrajectoryView &original, std::span<const std::size_t> kept);
    core::TrajectoryView view() const { return {m_id, m_t, {}, {}, m_x, m_y, m_box}; }

  private:
    std::int32_t m_id;
    std::vector<std::int64_t> m_t;
    std::vector<double> m_x, m_y;
    core::BoundingBox m_box;
};

class PreparedTrajectories {
  public:
    PreparedTrajectories(const core::TrajectoryDb &db, const KeptSets &kept);
    std::vector<core::TrajectoryView> views() const;

  private:
    std::vector<PreparedTrajectory> m_trajectories;
};

inline constexpr std::size_t kTermCount = 4; // kNN, range, where-at, join
using TermWeights = std::array<double, kTermCount>;

struct ObjectiveConfig {
    double min_tolerance_m = 25.0; // where-at ramp, as in the evaluator
    double max_tolerance_m = 500.0;
    // Self: both sides compressed. HeldOut: each trajectory also queries its partners as original
    query::JoinMode join_mode = query::JoinMode::Self;
    bool held_out_term = false; // with Self: add the held-out form at join_eps_m as a term
    // kNN ranked on the compressed database; false: per-trajectory check. Curves always use the
    // latter
    bool ranking_knn = false;
    // The kNN query compressed too, the in-database setting; needs ranking_knn. Curves then share
    // each hit between the query and its neighbour, both against the fixed partners
    bool compressed_knn_query = false;
    TermWeights term_weights{1.0, 1.0, 1.0, 1.0};
};

// Throws std::invalid_argument unless finite, not negative and not all zero. Returns their sum
double check_term_weights(const TermWeights &weights);

struct JoinLink {
    std::uint32_t other;
    std::uint32_t pair; // index into the objective's pair list
    bool positive;
    std::uint8_t term = 0; // join term the pair belongs to
};

// J_T: weighted mean of normalised kNN, range, where-at and join on the teacher's workloads,
// (metric - floor) / (1 - floor) with floor = endpoints only, like J4.
//   range     macro F1 over nonempty reference boxes, like J4
//   where-at  average-hit ramp over i's probes
//   join      macro F1 over candidate pairs per join term (threshold, held-out form), each
//             normalised by its own floor, then averaged: J4's join grid mean
// Where-at and the per-trajectory kNN add up over trajectories. Range, join and ranking kNN
// are tracked under one-at-a-time changes by RangeState, JoinState and KnnState
class Objective {
  public:
    enum Term : std::size_t { Knn, Range, WhereAt, Join, TermCount };
    using Terms = std::array<double, TermCount>;

    // Keeps references to db and workloads
    Objective(const core::TrajectoryDb &db, const TeacherWorkloads &workloads,
              const ObjectiveConfig &config = {});

    // i's scaled separable terms, 0 with endpoints only
    double separable(std::size_t i, std::span<const std::size_t> kept) const;
    double separable(std::size_t i, const core::TrajectoryView &candidate) const;

    // Approximate score of i alone: separable, marginal range F1, and i's join F1 against fixed
    // partners (empty = original; the join ignores partners when held out). Exact above beat,
    // otherwise some value below beat
    double independent_value(std::size_t i, const core::TrajectoryView &candidate,
                             std::span<const core::TrajectoryView> partners,
                             double beat = -std::numeric_limits<double>::infinity(),
                             std::span<const double> range_weights = {}) const;
    // independent_value with every kNN check a hit
    double independent_bound(std::size_t i, const core::TrajectoryView &candidate,
                             std::span<const core::TrajectoryView> partners) const;

    // A bound followed by scoring without recomputing. Borrows the candidate's coordinates
    struct IndependentScore {
        double bound = 0.0;

      private:
        friend class Objective;
        std::size_t trajectory = 0;
        core::TrajectoryView candidate{};
        std::span<const core::TrajectoryView> partners{};
        Terms terms{};
        double join = 0.0;
    };
    IndependentScore prepare_independent(std::size_t i, const core::TrajectoryView &candidate,
                                         std::span<const core::TrajectoryView> partners,
                                         std::span<const double> range_weights = {}) const;
    double independent_value(const IndependentScore &score,
                             double beat = -std::numeric_limits<double>::infinity()) const;

    // Exact J_T
    double total(const KeptSets &kept) const;

    // Unscaled metrics
    Terms metrics(const KeptSets &kept) const;
    const Terms &floors() const { return m_floor; }
    // Recall@k of the database kNN queries ranked on compressed, as in metrics with ranking_knn
    double knn_recall(const core::TrajectoryDb &compressed) const;
    double scale(Term m) const { return m_scale[m]; }

    std::span<const JoinLink> links(std::size_t i) const { return m_links[i]; }
    std::size_t pair_count() const { return m_pair_i.size(); }
    std::size_t positive_count() const { return m_positive_count; }
    std::size_t slot(std::size_t t, std::size_t term) const { return t + term * m_db.size(); }
    std::size_t slots() const { return m_truth.size(); }
    std::size_t join_terms() const { return m_eps.size(); }
    double join_floor(std::size_t term) const { return m_join_floor[term]; }
    // The join metric from each term's macro F1
    double join_value(std::span<const double> f1s) const;
    std::size_t truth(std::size_t slot) const { return m_truth[slot]; }
    bool held_out(std::size_t term) const { return m_term_held_out[term]; }

    struct RangeMembership {
        std::uint32_t box;
        double weight; // marginal macro F1 of covering i with every other reference covered
    };
    // Marginal macro F1 of covering i, holding all other trajectories fixed (empty = original).
    // RangeState::weights reuses its counts when scoring many candidates against one state.
    std::vector<double> range_weights(std::size_t i,
                                      std::span<const core::TrajectoryView> partners) const;
    const core::TrajectoryDb &db() const { return m_db; }
    const TeacherWorkloads &workloads() const { return m_workloads; }
    const ObjectiveConfig &config() const { return m_config; }
    std::span<const RangeMembership> boxes(std::size_t i) const { return m_range[i]; }
    std::span<const std::uint32_t> probes(std::size_t i) const { return m_probes[i]; } // where-at
    double where_at_weight() const { return m_where_at_weight; }

    bool match(const core::TrajectoryView &a, const core::TrajectoryView &b,
               std::size_t term = 0) const;

  private:
    struct KnnMembership {
        std::uint32_t query;
        double cutoff;
    };

    Terms raw_separable(std::size_t i, std::span<const std::size_t> kept) const;
    Terms raw_separable(std::size_t i, const core::TrajectoryView &candidate,
                        bool knn = true) const;
    double independent_sum(const IndependentScore &score, double knn) const;
    std::size_t knn_checks(std::size_t i) const;
    double knn_check_weight() const;
    double join_metric(std::span<const core::TrajectoryView> state) const;
    std::vector<double> join_f1s(std::span<const core::TrajectoryView> state) const;

    const core::TrajectoryDb &m_db;
    const TeacherWorkloads &m_workloads;
    ObjectiveConfig m_config;

    std::vector<std::vector<KnnMembership>> m_knn;
    double m_knn_weight = 0; // 1 / (k * valid queries)
    std::vector<std::vector<RangeMembership>> m_range;
    std::vector<std::vector<std::uint32_t>> m_probes; // where-at query indices per trajectory
    double m_where_at_weight = 0;                     // 1 / trajectories with probes

    std::vector<std::uint32_t> m_pair_i, m_pair_j; // pairs used, positives first
    std::vector<std::uint8_t> m_pair_term;
    std::vector<double> m_eps; // threshold of each join term
    std::vector<char> m_term_held_out;
    std::size_t m_positive_count = 0;
    std::vector<std::vector<JoinLink>> m_links; // per trajectory
    std::vector<std::size_t> m_truth;           // positives per slot
    std::vector<std::size_t> m_with_truth;      // per join term, slots with any positive
    std::vector<double> m_join_floor;           // per join term, macro F1 with endpoints only
    std::vector<double> m_join_weight;          // per join term, 1 / (terms * (1 - floor))

    Terms m_floor{};
    Terms m_scale{};                          // weight / (weight sum * (1 - floor))
    std::vector<double> m_endpoint_separable; // scaled separable terms, endpoints only
};

// Exact range macro F1 under one-at-a-time changes. A subset of original samples cannot add
// false positives, so each box's F1 is 2 * found / (reference size + found).
class RangeState {
  public:
    // Empty state means original trajectories
    RangeState(const Objective &objective, std::span<const core::TrajectoryView> state);

    struct Proposal {
        std::size_t trajectory;
        double gain = 0.0;         // change in J_T
        std::vector<char> covered; // in boxes(trajectory) order
    };
    Proposal propose(std::size_t i, const core::TrajectoryView &candidate) const;
    // A proposal is valid until the next apply()
    void apply(const Proposal &proposal);
    double metric() const;
    std::vector<double> weights(std::size_t i) const;

  private:
    double f1(std::size_t box, std::size_t found) const;
    const Objective &m_objective;
    std::vector<std::size_t> m_found;
    std::vector<std::vector<char>> m_covered;
    std::size_t m_scored = 0;
    double m_sum = 0.0;
};

// The exact join metric under one-at-a-time changes; changing i only rescores i's pairs
class JoinState {
  public:
    JoinState(const Objective &objective, std::span<const core::TrajectoryView> state);

    struct Proposal {
        std::size_t trajectory;
        double gain = 0.0;
        std::vector<char> matched; // outcomes in links(trajectory) order
    };

    // Valid until the next apply()
    Proposal propose(std::size_t i, const core::TrajectoryView &candidate,
                     std::span<const core::TrajectoryView> state) const;
    void apply(const Proposal &proposal);

    double metric() const; // as Objective's join metric

  private:
    bool counted(std::size_t t, std::size_t found) const;
    double f1(std::size_t t, std::size_t hits, std::size_t found) const;
    std::size_t count(std::size_t t, const JoinLink &link) const {
        return m_objective.slot(t, link.term);
    }
    // Where the outcome of i's link is stored. Held out: the direction with i compressed
    std::size_t slot(std::size_t i, const JoinLink &link) const {
        return 2 * std::size_t{link.pair} + (m_objective.held_out(link.term) && i > link.other);
    }
    // Held out: only the query's counts change
    std::size_t sides(const JoinLink &link) const {
        return m_objective.held_out(link.term) ? 1 : 2;
    }

    const Objective &m_objective;
    std::vector<char> m_matched; // per pair and direction
    std::vector<std::size_t> m_hits, m_found;
    std::vector<double> m_sum; // per join term, sum of F1 over counted slots
    std::vector<long> m_counted;
};

// The ranking kNN metric under one-at-a-time changes. Each query watches its true top k and its
// nearest others in state and watch_states; unwatched trajectories are assumed to stay outside
class KnnState {
  public:
    KnnState(const Objective &objective, const KeptSets &state,
             std::span<const KeptSets> watch_states, std::size_t watch);

    bool watched(std::size_t i) const { return !m_watchers[i].empty(); }
    // Whether i's choice can change the metric: watched, or with compressed queries a query
    bool active(std::size_t i) const;
    // Original queries only. From every query watching i, in the order propose and apply expect
    std::vector<double> distances(std::size_t i, const core::TrajectoryView &candidate) const;
    // Change in J_T
    double propose(std::size_t i, std::span<const double> distances) const;
    void apply(std::size_t i, std::span<const double> distances);

    // Either query kind. With compressed queries a distance depends on both sides, so a move also
    // rescores i's own list, against the current state
    struct Move {
        std::size_t trajectory = 0;
        std::vector<double> watched; // as distances()
        std::vector<double> own;     // i's own list; empty with original queries
    };
    Move move(std::size_t i, const core::TrajectoryView &candidate,
              std::span<const core::TrajectoryView> state) const;
    double propose(const Move &move) const;
    void apply(const Move &move);

    double metric() const; // recall@k over the valid queries

  private:
    struct Watcher {
        std::uint32_t query, position; // position in the query's list
    };
    // True neighbours in e's top k, with the distance at position replaced
    std::size_t hits(std::size_t e, std::size_t position, double distance) const;
    // The same with e's whole row replaced
    std::size_t hits(std::size_t e, std::span<const double> row) const;
    // Distances from query, e in some state, to e's list in state
    std::vector<double> own_row(std::size_t e, const core::TrajectoryView &query,
                                std::span<const core::TrajectoryView> state) const;

    const Objective &m_objective;
    std::size_t m_valid = 0;
    std::vector<std::vector<std::uint32_t>> m_list; // per query, its true top k first
    std::vector<std::vector<double>> m_distance;    // per query, in list order
    std::vector<std::size_t> m_hits;                // per query
    std::size_t m_hit_sum = 0;
    std::vector<std::vector<Watcher>> m_watchers; // per trajectory
};

} // namespace teacher
