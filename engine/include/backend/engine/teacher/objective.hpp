#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
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

// Upper bound on the continuous Fréchet distance between a trajectory and a simplification
// keeping a subsequence of its points. Infinity if kept is not such a subsequence. segments:
// the original's edge lengths, if already known
double simplification_bound(const core::TrajectoryView &original, const core::TrajectoryView &kept,
                            std::span<const double> segments = {});

inline constexpr std::size_t kTermCount = 6; // kNN, range, where-at, join, shape, margin
using TermWeights = std::array<double, kTermCount>;

struct ObjectiveConfig {
    double min_tolerance_m = 25.0; // where-at ramp, as in the evaluator
    double max_tolerance_m = 500.0;
    double min_shape_m = 25.0; // shape ramp on each trajectory's deviation bound
    double max_shape_m = 500.0;
    TermWeights term_weights{1.0, 1.0, 1.0, 1.0, 0.0, 0.0};
};

// 1 up to min, 0 from max, linear between: the where-at and shape scores
inline double ramp(double value, double min, double max) {
    return std::clamp((max - value) / (max - min), 0.0, 1.0);
}

// Throws std::invalid_argument unless finite, not negative and not all zero. Returns their sum
double check_term_weights(const TermWeights &weights);

struct JoinLink {
    std::uint32_t other;
    std::uint32_t pair; // index into the objective's pair list
    bool positive;
    std::uint8_t term = 0; // join term the pair belongs to
};

// J_T: weighted mean of normalised terms on the teacher's workloads, (metric - floor) / (1 - floor)
// with floor = endpoints only, like J4. Queries are original, as in the held-out test.
//   kNN       recall@k of the database kNN queries ranked on the compressed database. Value curves
//             use a per-trajectory check instead: original neighbour within the query's cutoff
//   range     macro F1 over nonempty reference boxes, like J4
//   where-at  average-hit ramp over i's probes
//   join      held-out macro F1 per threshold: each trajectory queries its partners as original
//             against their compressed versions; each threshold normalised by its own floor, then
//             averaged: J4's join grid mean
//   shape     query-free: mean over trajectories of a ramp on simplification_bound, a prior
//             for queries the workloads do not contain. Off unless weighted
//   margin    query-free per hit: for each true neighbour of a database kNN query, the chance it
//             stays inside the top k if its distance moves anywhere within its deviation bound,
//             against its gap to the (k + 1)th neighbour, and for each outsider ranked past k
//             the chance it stays out, against its gap to the k-th. Needs workloads.ranked
//             (add_ranked_neighbours). Off unless weighted
// Where-at, shape, margin and the per-trajectory kNN add up over trajectories. Range, join and kNN
// are tracked under one-at-a-time changes by RangeState, JoinState and KnnState
class JoinState;

class Objective {
  public:
    enum Term : std::size_t { Knn, Range, WhereAt, Join, Shape, Margin, TermCount };
    using Terms = std::array<double, TermCount>;

    // Keeps references to db and workloads
    Objective(const core::TrajectoryDb &db, const TeacherWorkloads &workloads,
              const ObjectiveConfig &config = {});

    // i's scaled where-at, shape and margin, 0 with endpoints only
    double separable(std::size_t i, std::span<const std::size_t> kept) const;
    double separable(std::size_t i, const core::TrajectoryView &candidate,
                     double deviation = -1.0) const;

    // Approximate score of i alone: separable, per-trajectory kNN, marginal range F1 against fixed
    // partners (empty = original), and join (exact partner-slot marginal via join_state when given,
    // else i's own-slot fallback). Exact above beat, otherwise some value below beat
    double independent_value(std::size_t i, const core::TrajectoryView &candidate,
                             std::span<const core::TrajectoryView> partners,
                             double beat = -std::numeric_limits<double>::infinity(),
                             std::span<const double> range_weights = {},
                             const JoinState *join_state = nullptr) const;

    // A bound followed by scoring without recomputing. Borrows the candidate's coordinates
    struct IndependentScore {
        double bound = 0.0;

      private:
        friend class Objective;
        std::size_t trajectory = 0;
        core::TrajectoryView candidate{};
        Terms terms{};
        double join = 0.0;
        double deviation = 0.0; // simplification_bound of the candidate
    };
    IndependentScore prepare_independent(std::size_t i, const core::TrajectoryView &candidate,
                                         std::span<const core::TrajectoryView> partners,
                                         std::span<const double> range_weights = {},
                                         const JoinState *join_state = nullptr) const;
    double independent_value(const IndependentScore &score,
                             double beat = -std::numeric_limits<double>::infinity()) const;

    // Exact J_T, from the kept sets or from their metrics
    double total(const KeptSets &kept) const;
    double total(const Terms &metrics) const;
    // simplification_bound of candidate against original i, with i's edge lengths cached
    double deviation(std::size_t i, const core::TrajectoryView &candidate) const;

    // Unscaled metrics
    Terms metrics(const KeptSets &kept) const;
    const Terms &floors() const { return m_floor; }
    // Recall@k of the database kNN queries ranked on compressed, as in metrics
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
    // frechet_within(original j, candidate, radius), first on a coarse copy of j whose deviation
    // bound decides most tests
    bool original_within(std::size_t j, const core::TrajectoryView &candidate, double radius) const;
    // match(original of link.other, candidate), skipping the test when the originals' distance
    // bracket and the candidate's own error already decide it. own_error < 0: not yet computed
    bool held_out_match(std::size_t i, const JoinLink &link, const core::TrajectoryView &candidate,
                        double &own_error) const;

  private:
    struct KnnMembership {
        std::uint32_t query;
        double cutoff;
        // Bracket on the continuous distance from the query to original i; 0, infinity if unknown
        double lower = 0.0, upper = std::numeric_limits<double>::infinity();
    };
    // Whether the triangle bound alone decides d(query, candidate) <= cutoff, given a bracket on
    // d(query, original) and deviation bounding d(original, candidate); nullopt if it does not
    std::optional<bool> certify(double lower, double upper, double deviation, double cutoff) const;
    // m_pair_lower and m_pair_upper, once per pair across terms
    void bracket_pairs();

    // Knn slot left 0. deviation < 0: computed here if needed
    Terms raw_separable(std::size_t i, const core::TrajectoryView &candidate,
                        double deviation = -1.0) const;
    double independent_sum(const IndependentScore &score, double knn) const;
    double join_metric(std::span<const core::TrajectoryView> state) const;
    std::vector<double> join_f1s(std::span<const core::TrajectoryView> state) const;

    const core::TrajectoryDb &m_db;
    const TeacherWorkloads &m_workloads;
    ObjectiveConfig m_config;

    std::vector<std::vector<KnnMembership>> m_knn;
    double m_knn_weight = 0; // 1 / (k * valid queries)
    std::vector<std::vector<RangeMembership>> m_range;
    std::size_t m_scored_boxes = 0;                   // boxes with a nonempty reference
    std::vector<std::vector<std::uint32_t>> m_probes; // where-at query indices per trajectory
    double m_where_at_weight = 0;                     // 1 / trajectories with probes
    double m_shape_weight = 0;                        // 1 / trajectories
    std::vector<std::vector<double>> m_margins;       // per trajectory, one per kNN membership
    double m_margin_weight = 0;                       // 1 / memberships

    std::vector<std::uint32_t> m_pair_i, m_pair_j; // pairs used, positives first
    std::vector<std::uint8_t> m_pair_term;
    // Per pair, a bracket on the originals' continuous distance; 0, infinity if unknown
    std::vector<double> m_pair_lower, m_pair_upper;
    std::vector<double> m_eps; // threshold of each join term
    std::size_t m_positive_count = 0;
    std::vector<std::vector<JoinLink>> m_links; // per trajectory
    std::vector<std::size_t> m_truth;           // positives per slot
    std::vector<std::size_t> m_with_truth;      // per join term, slots with any positive
    std::vector<double> m_join_floor;           // per join term, macro F1 with endpoints only
    std::vector<double> m_join_weight;          // per join term, 1 / (terms * (1 - floor))

    // Continuous Fréchet only: brackets and deviation bounds decide tests, coarse copies screen
    // the rest
    bool m_certify = false;
    std::vector<PreparedTrajectory> m_coarse;    // per trajectory, RDP to about kCoarseM
    std::vector<double> m_coarse_error;          // its simplification_bound; < 0: not used
    std::vector<std::vector<double>> m_segments; // per trajectory, its edge lengths

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
    // Exact range J_T change from replacing i with candidate, without building a Proposal
    double gain(std::size_t i, const core::TrajectoryView &candidate) const;
    // Which of i's boxes the candidate covers; independent of the state
    std::vector<char> coverage(std::size_t i, const core::TrajectoryView &candidate) const;
    Proposal propose(std::size_t i, std::vector<char> covered) const;
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

// The exact held-out join metric under one-at-a-time changes. A pair has one outcome per direction,
// each found by its original query; changing i only rescores the directions with i compressed
class JoinState {
  public:
    JoinState(const Objective &objective, std::span<const core::TrajectoryView> state);

    struct Proposal {
        std::size_t trajectory;
        double gain = 0.0;
        std::vector<char> matched; // outcomes in links(trajectory) order
    };

    // Valid until the next apply()
    Proposal propose(std::size_t i, const core::TrajectoryView &candidate) const;
    // Exact range J_T change from replacing i with candidate, without building a Proposal
    double gain(std::size_t i, const core::TrajectoryView &candidate) const;
    // Exact join J_T change from replacing i with candidate, without building a Proposal.
    // own_error < 0: computed on demand and cached in own_error.
    double gain(std::size_t i, const core::TrajectoryView &candidate, double &own_error) const;
    // Outcomes of i's links with i as candidate, in links(i) order. They do not depend on the state
    std::vector<char> outcomes(std::size_t i, const core::TrajectoryView &candidate) const;
    Proposal propose(std::size_t i, std::vector<char> matched) const;
    void apply(const Proposal &proposal);

    double metric() const; // as Objective's join metric

  private:
    bool counted(std::size_t t, std::size_t found) const;
    double f1(std::size_t t, std::size_t hits, std::size_t found) const;
    void update_slot(const JoinLink &link, long delta, std::span<double> sum,
                     std::span<long> n_counted) const;
    // The query's counts: link.other queries i
    std::size_t count(const JoinLink &link) const {
        return m_objective.slot(link.other, link.term);
    }
    // Where the outcome of i's link is stored: the direction with i compressed
    std::size_t slot(std::size_t i, const JoinLink &link) const {
        return 2 * std::size_t{link.pair} + (i > link.other);
    }

    const Objective &m_objective;
    std::vector<char> m_matched; // per pair and direction
    std::vector<std::size_t> m_hits, m_found;
    std::vector<double> m_sum; // per join term, sum of F1 over counted slots
    std::vector<long> m_counted;
};

// The ranking kNN metric under one-at-a-time changes. Each query watches its true top k and its
// nearest others in state; unwatched trajectories are assumed to stay outside
class KnnState {
  public:
    KnnState(const Objective &objective, const KeptSets &state, std::size_t watch);

    // Whether i's choice can change the metric
    bool watched(std::size_t i) const { return !m_watchers[i].empty(); }
    // From every query watching i, in the order propose and apply expect
    std::vector<double> distances(std::size_t i, const core::TrajectoryView &candidate,
                                  bool parallel = false) const;
    // i's current distances, as distances() gives them for i's current version
    std::vector<double> stored(std::size_t i) const;
    // Change in J_T
    double propose(std::size_t i, std::span<const double> distances) const;
    // The same change from i becoming candidate, computing a distance only near a query's k-th
    // place. reach bounds d(i's current version, candidate), e.g. both deviation bounds: then
    // the stored distance alone often places the candidate. Exact above beat, otherwise <= beat
    double propose_bounded(std::size_t i, const core::TrajectoryView &candidate,
                           double reach = std::numeric_limits<double>::infinity(),
                           double beat = -std::numeric_limits<double>::infinity()) const;
    // Adds trajectory i to the list of every query that one of versions[i] comes within the query's
    // watch radius of, the distance of its watch-th nearest in state
    void watch(std::span<const std::vector<core::TrajectoryView>> versions,
               std::span<const core::TrajectoryView> state);
    void apply(std::size_t i, std::span<const double> distances);

    double metric() const; // recall@k over the valid queries

  private:
    struct Watcher {
        std::uint32_t query, position; // position in the query's list
    };
    // Change in J_T from a change in total hits
    double value(long change) const;
    // True neighbours in e's top k, with the distance at position replaced. Reads e's ranking
    std::size_t hits(std::size_t e, std::size_t position, double distance) const;
    // Re-ranks e after its row changed: m_rank[e] and m_hits[e]
    void refresh(std::size_t e);
    bool ranks_before(std::size_t e, double distance, std::size_t position,
                      std::size_t other) const;

    const Objective &m_objective;
    std::size_t m_valid = 0;
    std::size_t m_watch = 0;
    std::vector<std::vector<std::uint32_t>> m_list; // per query, its true top k first
    std::vector<std::vector<double>> m_distance;    // per query, in list order
    std::vector<std::size_t> m_hits;                // per query
    std::vector<std::vector<std::uint32_t>> m_rank; // per query, positions of its top k + 1
    std::size_t m_hit_sum = 0;
    std::vector<std::vector<Watcher>> m_watchers; // per trajectory
};

} // namespace teacher
