// Go/no-go experiment: can moving points between trajectories raise kNN recall@k?
//
// Splits the kNN query pool into a training and a test half. On the training half it builds an
// allocation that greedily maximises recall@k (RDP as selector, Fréchet distances precomputed
// on a grid of point counts), then scores it with the real kNN evaluator on both halves against
// size-proportional allocation and a few fixed reallocations. If the oracle cannot beat
// size-proportional on the test half, a learned allocator has nothing to learn.

#include "backend/engine/allocation/constants.hpp"
#include "backend/engine/allocation/size_proportional.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/evaluator/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/where_at_query.hpp"
#include "backend/engine/io/parquet_loader.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/selection/rdp.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/split/split.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <format>
#include <functional>
#include <optional>
#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

struct Options {
    std::string config_path;
    std::size_t train_queries = 500; // 0 = the whole training half
    std::size_t test_queries = 500;  // 0 = the whole test half
    std::size_t db_limit = 0;        // 0 = the whole database
    std::size_t candidate_factor = 4;
    std::size_t grid_cap = 128;
    bool features = false; // fit a feature-weighted allocation instead of running the oracle
    bool model = false;    // fit linear and MLP allocators on the extended features
    bool check = false;    // score allocators on all three query types
    bool combined = false; // train the 3-feature allocator on all three query types, SED
    std::size_t evaluations = 250; // training evaluations per model and budget
};

Options parse_options(int argc, char *argv[]) {
    if (argc < 2)
        throw std::invalid_argument(
            "usage: knn_alloc_oracle <config.json> [--train N] [--test N] [--db-limit N] "
            "[--candidates F] [--grid-cap N] [--features 1] [--model 1] [--evaluations N] [--check 1] [--combined 1]");
    Options options{.config_path = argv[1]};
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string_view flag = argv[i];
        const std::size_t value = std::stoul(argv[i + 1]);
        if (flag == "--train")
            options.train_queries = value;
        else if (flag == "--test")
            options.test_queries = value;
        else if (flag == "--db-limit")
            options.db_limit = value;
        else if (flag == "--candidates")
            options.candidate_factor = value;
        else if (flag == "--grid-cap")
            options.grid_cap = value;
        else if (flag == "--features")
            options.features = value != 0;
        else if (flag == "--model")
            options.model = value != 0;
        else if (flag == "--evaluations")
            options.evaluations = value;
        else if (flag == "--check")
            options.check = value != 0;
        else if (flag == "--combined")
            options.combined = value != 0;
        else
            throw std::invalid_argument(std::format("unknown flag {}", flag));
    }
    return options;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

struct Data {
    core::TrajectoryDb database;
    core::TrajectoryDb train_queries;
    core::TrajectoryDb test_queries;
};

std::vector<std::size_t> take(std::span<const std::size_t> indices, std::size_t count) {
    const std::size_t n = count == 0 ? indices.size() : std::min(count, indices.size());
    return {indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(n)};
}

Data load(const pipeline::Config &config, const Options &options) {
    const core::TrajectoryDb all = io::load_trajectories(config.data_path);
    const auto parts = split::split(all, config.split, config.seed);
    std::mt19937_64 rng(config.seed);

    auto database = parts.database;
    if (options.db_limit != 0 && options.db_limit < database.size()) {
        std::ranges::shuffle(database, rng);
        database.resize(options.db_limit);
        std::ranges::sort(database);
    }

    // Disjoint halves of the held-out pool: one to fit the oracle, one to test it
    auto pool = parts.queries;
    std::ranges::shuffle(pool, rng);
    const auto half = pool.begin() + static_cast<std::ptrdiff_t>(pool.size() / 2);
    const std::vector<std::size_t> train(pool.begin(), half);
    const std::vector<std::size_t> test(half, pool.end());
    return {.database = core::subset(all, database),
            .train_queries = core::subset(all, take(train, options.train_queries)),
            .test_queries = core::subset(all, take(test, options.test_queries))};
}

// Same rule as pipeline::run, so shares mean the same here as in the main results
std::size_t total_budget(const core::TrajectoryDb &trajectories, double share) {
    const std::size_t endpoints = 2 * trajectories.size();
    const double extra = share * static_cast<double>(trajectories.num_points() - endpoints);
    return endpoints + static_cast<std::size_t>(std::llround(extra));
}

std::vector<std::size_t> sizes_of(const core::TrajectoryDb &trajectories) {
    std::vector<std::size_t> sizes(trajectories.size());
    for (std::size_t i = 0; i < sizes.size(); ++i)
        sizes[i] = trajectories.trajectory(i).size();
    return sizes;
}

// Adds extra points to budgets in proportion to weights, never past sizes. A trajectory that
// fills up drops out and its share goes to the others.
void weighted_fill(std::vector<std::size_t> &budgets, const std::vector<std::size_t> &sizes,
                   const std::vector<double> &weights, std::size_t extra) {
    std::vector<std::size_t> open;
    for (std::size_t i = 0; i < budgets.size(); ++i) {
        if (budgets[i] < sizes[i] && weights[i] > 0.0)
            open.push_back(i);
    }
    while (extra > 0 && !open.empty()) {
        double weight_sum = 0.0;
        for (const auto i : open)
            weight_sum += weights[i];

        std::size_t given = 0;
        std::vector<std::pair<double, std::size_t>> remainders;
        for (const auto i : open) {
            const double share = static_cast<double>(extra) * weights[i] / weight_sum;
            const auto add = std::min(static_cast<std::size_t>(share), sizes[i] - budgets[i]);
            budgets[i] += add;
            given += add;
            remainders.emplace_back(share - static_cast<double>(add), i);
        }
        extra -= given;

        // Every share was below one point: hand out single points by largest remainder
        if (given == 0) {
            std::ranges::sort(remainders, std::greater{});
            for (const auto &[remainder, i] : remainders) {
                if (extra == 0)
                    break;
                ++budgets[i];
                --extra;
            }
        }
        std::erase_if(open, [&](std::size_t i) { return budgets[i] >= sizes[i]; });
    }
}

// Every trajectory keeps its endpoints, the rest is shared by size^alpha.
// alpha = 0 is uniform, alpha = 1 is (close to) size-proportional.
std::vector<std::size_t> power_budget(const std::vector<std::size_t> &sizes, double alpha,
                                      std::size_t total) {
    std::vector<std::size_t> budgets(sizes.size(), allocation::kMinPointsPerTrajectory);
    std::vector<double> weights(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i)
        weights[i] = std::pow(static_cast<double>(sizes[i]), alpha);
    weighted_fill(budgets, sizes, weights,
                  total - allocation::kMinPointsPerTrajectory * sizes.size());
    return budgets;
}

// Discrete Fréchet distance, squared, between a full query and a simplified trajectory.
// Same recurrence as query::knn, without its cutoffs.
double squared_frechet(const core::TrajectoryView &query, std::span<const double> x,
                       std::span<const double> y) {
    const auto dist = [&](std::size_t i, std::size_t j) {
        const double dx = query.x[i] - x[j];
        const double dy = query.y[i] - y[j];
        return dx * dx + dy * dy;
    };
    std::vector<double> row(x.size());
    double value = 0.0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        value = std::max(value, dist(0, j));
        row[j] = value;
    }
    for (std::size_t i = 1; i < query.size(); ++i) {
        double diagonal = row[0];
        row[0] = std::max(row[0], dist(i, 0));
        for (std::size_t j = 1; j < x.size(); ++j) {
            const double above = row[j];
            row[j] = std::max(std::min({above, diagonal, row[j - 1]}), dist(i, j));
            diagonal = above;
        }
    }
    return row.back();
}

// For each training query, its true top-k plus near misses (the candidates), and the Fréchet
// distance from the query to every candidate simplified to every point count on its grid.
struct KnnTable {
    std::size_t k = 0;
    std::vector<std::vector<std::size_t>> candidates; // per query, database indices
    std::vector<std::vector<std::size_t>> truth;      // per query, original top-k
    std::vector<std::size_t> members;                 // database indices that are candidates
    std::vector<std::size_t> member_of;               // database index -> position in members
    std::vector<std::vector<std::size_t>> grid;       // per member, increasing point counts
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> slots; // per member: (q, slot)
    std::vector<std::vector<std::vector<double>>> distance; // [query][slot][grid index]
};

std::vector<std::size_t> point_grid(std::size_t size, std::size_t cap,
                                    std::span<const std::size_t> extra) {
    std::vector<std::size_t> grid;
    for (std::size_t n = 2; n <= std::min(size, cap); n = n < 4 ? n + 1 : n + n / 2)
        grid.push_back(n);
    if (size <= cap)
        grid.push_back(size);
    for (const auto n : extra)
        grid.push_back(std::clamp(n, allocation::kMinPointsPerTrajectory, size));
    std::ranges::sort(grid);
    const auto [first, last] = std::ranges::unique(grid);
    grid.erase(first, last);
    return grid;
}

// extra_counts[i] lists point counts that database trajectory i must have on its grid
KnnTable build_table(const core::TrajectoryDb &database, const core::TrajectoryDb &queries,
                     std::size_t k, std::size_t candidate_factor, std::size_t grid_cap,
                     const std::vector<std::vector<std::size_t>> &extra_counts) {
    KnnTable table;
    table.k = k;
    table.candidates.resize(queries.size());
    core::parallel_for(queries.size(), [&](std::size_t q) {
        table.candidates[q] = query::knn(database, queries.trajectory(q), candidate_factor * k);
    });

    table.member_of.assign(database.size(), kNone);
    table.truth.resize(queries.size());
    table.distance.resize(queries.size());
    for (std::size_t q = 0; q < queries.size(); ++q) {
        const auto &candidates = table.candidates[q];
        table.truth[q].assign(candidates.begin(),
                              candidates.begin() +
                                  static_cast<std::ptrdiff_t>(std::min(k, candidates.size())));
        table.distance[q].resize(candidates.size());
        for (std::size_t slot = 0; slot < candidates.size(); ++slot) {
            const auto j = candidates[slot];
            if (table.member_of[j] == kNone) {
                table.member_of[j] = table.members.size();
                table.members.push_back(j);
                table.slots.emplace_back();
            }
            table.slots[table.member_of[j]].emplace_back(q, slot);
        }
    }

    table.grid.resize(table.members.size());
    core::parallel_for(table.members.size(), [&](std::size_t m) {
        const auto trajectory = database.trajectory(table.members[m]);
        table.grid[m] = point_grid(trajectory.size(), grid_cap, extra_counts[table.members[m]]);
        for (const auto &[q, slot] : table.slots[m])
            table.distance[q][slot].resize(table.grid[m].size());

        std::vector<double> x;
        std::vector<double> y;
        for (std::size_t g = 0; g < table.grid[m].size(); ++g) {
            const auto kept = selection::rdp::simplify(trajectory, table.grid[m][g]);
            x.clear();
            y.clear();
            for (const auto index : kept) {
                x.push_back(trajectory.x[index]);
                y.push_back(trajectory.y[index]);
            }
            for (const auto &[q, slot] : table.slots[m])
                table.distance[q][slot][g] = squared_frechet(queries.trajectory(q), x, y);
        }
    });
    return table;
}

std::size_t grid_index(const std::vector<std::size_t> &grid, std::size_t n) {
    const auto it = std::ranges::upper_bound(grid, n);
    return it == grid.begin() ? 0 : static_cast<std::size_t>(it - grid.begin()) - 1;
}

// Hits of query q among its candidates when every member m is simplified to grid[m][state[m]],
// except member moved, which uses grid index moved_to. Mirrors query::knn's (distance, index)
// order. Trajectories outside the candidates are assumed to stay outside the top-k.
std::size_t hits(const KnnTable &table, std::size_t q, const std::vector<std::size_t> &state,
                 std::size_t moved = kNone, std::size_t moved_to = 0) {
    const auto &candidates = table.candidates[q];
    std::vector<std::pair<double, std::size_t>> ranked(candidates.size());
    for (std::size_t slot = 0; slot < candidates.size(); ++slot) {
        const auto m = table.member_of[candidates[slot]];
        const auto g = m == moved ? moved_to : state[m];
        ranked[slot] = {table.distance[q][slot][g], candidates[slot]};
    }
    const auto top = std::min(table.k, ranked.size());
    std::ranges::partial_sort(ranked, ranked.begin() + static_cast<std::ptrdiff_t>(top));
    std::size_t count = 0;
    for (std::size_t i = 0; i < top; ++i)
        count += std::ranges::find(table.truth[q], ranked[i].second) != table.truth[q].end();
    return count;
}

std::vector<std::size_t> state_for(const KnnTable &table, const std::vector<std::size_t> &budgets) {
    std::vector<std::size_t> state(table.members.size());
    for (std::size_t m = 0; m < state.size(); ++m)
        state[m] = grid_index(table.grid[m], budgets[table.members[m]]);
    return state;
}

double predicted_recall(const KnnTable &table, const std::vector<std::size_t> &state) {
    std::size_t total = 0;
    for (std::size_t q = 0; q < table.candidates.size(); ++q)
        total += hits(table, q, state);
    return static_cast<double>(total) / static_cast<double>(table.k * table.candidates.size());
}

struct Move {
    double gain_per_point = 0.0;
    std::size_t to = kNone;
};

struct OracleResult {
    std::vector<std::size_t> budgets;
    double predicted_train_recall = 0.0;
    std::size_t points_moved = 0;
};

// Starting from start, repeatedly gives the member whose next grid step gains the most training
// hits per point exactly that step, until no step gains or the budget runs out. Whatever is left
// is shared by size.
OracleResult greedy_oracle(const KnnTable &table, const std::vector<std::size_t> &sizes,
                           std::vector<std::size_t> start, std::size_t total) {
    auto state = state_for(table, start);
    std::size_t used = 0;
    for (std::size_t i = 0; i < start.size(); ++i) {
        if (table.member_of[i] != kNone)
            start[i] = table.grid[table.member_of[i]][state[table.member_of[i]]];
        used += start[i];
    }
    if (used > total)
        throw std::logic_error("greedy_oracle: start allocation exceeds the budget");
    std::size_t remaining = total - used;

    std::vector<std::size_t> current(table.candidates.size());
    for (std::size_t q = 0; q < current.size(); ++q)
        current[q] = hits(table, q, state);

    const auto best_move = [&](std::size_t m) {
        Move best;
        const auto &grid = table.grid[m];
        for (std::size_t g = state[m] + 1; g < grid.size(); ++g) {
            const std::size_t cost = grid[g] - grid[state[m]];
            if (cost > remaining)
                break;
            long gain = 0;
            for (const auto &[q, slot] : table.slots[m])
                gain += static_cast<long>(hits(table, q, state, m, g)) -
                        static_cast<long>(current[q]);
            const double per_point = static_cast<double>(gain) / static_cast<double>(cost);
            if (gain > 0 && per_point > best.gain_per_point)
                best = {per_point, g};
        }
        return best;
    };

    std::vector<Move> moves(table.members.size());
    core::parallel_for(moves.size(), [&](std::size_t m) { moves[m] = best_move(m); });

    std::size_t moved = 0;
    std::vector<std::size_t> affected;
    std::vector<char> seen(table.members.size(), 0);
    while (true) {
        std::size_t pick = kNone;
        for (std::size_t m = 0; m < moves.size(); ++m) {
            if (moves[m].to != kNone &&
                (pick == kNone || moves[m].gain_per_point > moves[pick].gain_per_point))
                pick = m;
        }
        if (pick == kNone)
            break;

        const auto &grid = table.grid[pick];
        const std::size_t cost = grid[moves[pick].to] - grid[state[pick]];
        if (cost > remaining) {
            moves[pick] = best_move(pick); // the budget shrank since this move was scored
            continue;
        }
        state[pick] = moves[pick].to;
        remaining -= cost;
        moved += cost;

        // Only members sharing a query with pick can have changed gains
        affected.clear();
        for (const auto &[q, slot] : table.slots[pick]) {
            current[q] = hits(table, q, state);
            for (const auto j : table.candidates[q]) {
                const auto m = table.member_of[j];
                if (!seen[m]) {
                    seen[m] = 1;
                    affected.push_back(m);
                }
            }
        }
        core::parallel_for(affected.size(),
                           [&](std::size_t a) { moves[affected[a]] = best_move(affected[a]); });
        for (const auto m : affected)
            seen[m] = 0;
    }

    OracleResult result{.budgets = std::move(start), .points_moved = moved};
    for (std::size_t m = 0; m < state.size(); ++m)
        result.budgets[table.members[m]] = table.grid[m][state[m]];
    std::size_t sum = 0;
    for (const auto b : result.budgets)
        sum += b;
    std::vector<double> weights(sizes.begin(), sizes.end());
    weighted_fill(result.budgets, sizes, weights, total - sum);

    std::size_t hit_total = 0;
    for (const auto h : current)
        hit_total += h;
    result.predicted_train_recall =
        static_cast<double>(hit_total) / static_cast<double>(table.k * current.size());
    return result;
}

// Recall@k per query, from the evaluator's answers
std::vector<double> per_query_recall(const core::TrajectoryDb &database,
                                     const query::KnnWorkload &workload,
                                     const evaluator::KnnEvaluation &evaluation) {
    std::vector<double> recalls(workload.entries.size());
    for (std::size_t i = 0; i < recalls.size(); ++i) {
        const auto &reference = workload.entries[i].reference;
        const auto &found = evaluation.answers[i].neighbors;
        std::size_t shared = 0;
        for (const auto index : reference)
            shared += std::ranges::find(found, database.trajectory(index).trajectory_id) !=
                      found.end();
        recalls[i] = static_cast<double>(shared) / static_cast<double>(reference.size());
    }
    return recalls;
}

struct Scores {
    std::vector<double> train;
    std::vector<double> test;
};

Scores score(const core::TrajectoryDb &database, const std::vector<std::size_t> &budgets,
             const query::KnnWorkload &train, const query::KnnWorkload &test) {
    const auto simplification =
        selection::simplify_database(database, selection::AlgorithmType::RDP, budgets);
    const auto compressed = core::materialize(database, simplification);
    return {per_query_recall(database, train, evaluator::evaluate_knn(compressed, train)),
            per_query_recall(database, test, evaluator::evaluate_knn(compressed, test))};
}

double mean(const std::vector<double> &values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

// Mean and standard error of the per-query difference a - b (same queries, so paired)
std::pair<double, double> paired_difference(const std::vector<double> &a,
                                            const std::vector<double> &b) {
    std::vector<double> diff(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        diff[i] = a[i] - b[i];
    const double m = mean(diff);
    double squares = 0.0;
    for (const auto d : diff)
        squares += (d - m) * (d - m);
    const double n = static_cast<double>(diff.size());
    return {m, n > 1 ? std::sqrt(squares / (n - 1) / n) : 0.0};
}

void print_row(std::string_view name, const Scores &scores, const Scores &baseline,
               double predicted) {
    const auto [train_diff, train_se] = paired_difference(scores.train, baseline.train);
    const auto [test_diff, test_se] = paired_difference(scores.test, baseline.test);
    std::cout << std::format("  {:<30} {:>9.4f} {:>9.4f}   {:>+8.4f} ± {:.4f}   {:>+8.4f} ± {:.4f}",
                             name, mean(scores.train), mean(scores.test), train_diff, train_se,
                             test_diff, test_se);
    if (!std::isnan(predicted))
        std::cout << std::format("   {:>9.4f}", predicted);
    std::cout << '\n';
}

query::KnnWorkload workload_from(const core::TrajectoryDb &queries, const KnnTable &table,
                                 std::size_t k) {
    query::KnnWorkload workload{queries, {}, k};
    workload.entries.resize(queries.size());
    for (std::size_t q = 0; q < queries.size(); ++q)
        workload.entries[q] = {q, table.truth[q]};
    return workload;
}

// Per-trajectory features, each divided by its median so the exponents are scale-free
struct Features {
    std::vector<double> size;
    std::vector<double> density; // distinct trajectories per grid cell, averaged over its cells
    std::vector<double> curve;   // path length / bounding-box diagonal, 1 = straight line
};

void normalize_by_median(std::vector<double> &values) {
    std::vector<double> sorted = values;
    std::ranges::nth_element(sorted, sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2));
    const double median = sorted[sorted.size() / 2];
    for (auto &v : values)
        v /= median;
}

Features trajectory_features(const core::TrajectoryDb &database, double cell_m) {
    const std::size_t n = database.size();
    Features features{std::vector<double>(n), std::vector<double>(n), std::vector<double>(n)};
    std::vector<std::vector<std::int64_t>> cells(n);
    core::parallel_for(n, [&](std::size_t i) {
        const auto trajectory = database.trajectory(i);
        double length = 0.0;
        for (std::size_t p = 0; p < trajectory.size(); ++p) {
            const auto cx = static_cast<std::int64_t>(std::floor(trajectory.x[p] / cell_m));
            const auto cy = static_cast<std::int64_t>(std::floor(trajectory.y[p] / cell_m));
            cells[i].push_back((cx << 32) ^ (cy & 0xffffffff));
            if (p > 0)
                length += std::hypot(trajectory.x[p] - trajectory.x[p - 1],
                                     trajectory.y[p] - trajectory.y[p - 1]);
        }
        std::ranges::sort(cells[i]);
        const auto [first, last] = std::ranges::unique(cells[i]);
        cells[i].erase(first, last);
        const auto &box = trajectory.bbox;
        const double diagonal = std::hypot(box.x_max - box.x_min, box.y_max - box.y_min);
        features.size[i] = static_cast<double>(trajectory.size());
        features.curve[i] = std::max(1.0, length / std::max(diagonal, 1.0));
    });

    std::unordered_map<std::int64_t, std::uint32_t> visitors;
    for (const auto &list : cells)
        for (const auto cell : list)
            ++visitors[cell];
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (const auto cell : cells[i])
            sum += visitors[cell];
        features.density[i] = sum / static_cast<double>(cells[i].size());
    }
    normalize_by_median(features.size);
    normalize_by_median(features.density);
    normalize_by_median(features.curve);
    return features;
}

struct Exponents {
    double size = 0.0;
    double density = 0.0;
    double curve = 0.0;

    auto operator<=>(const Exponents &) const = default;
};

// Endpoints for everyone, the rest shared by size^a * density^b * curve^c
std::vector<std::size_t> feature_budget(const Features &features,
                                        const std::vector<std::size_t> &sizes, Exponents e,
                                        std::size_t total) {
    std::vector<std::size_t> budgets(sizes.size(), allocation::kMinPointsPerTrajectory);
    std::vector<double> weights(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i)
        weights[i] = std::exp(e.size * std::log(features.size[i]) +
                              e.density * std::log(features.density[i]) +
                              e.curve * std::log(features.curve[i]));
    weighted_fill(budgets, sizes, weights,
                  total - allocation::kMinPointsPerTrajectory * sizes.size());
    return budgets;
}

std::vector<double> recall_of(const core::TrajectoryDb &database,
                              const std::vector<std::size_t> &budgets,
                              const query::KnnWorkload &workload) {
    const auto simplification =
        selection::simplify_database(database, selection::AlgorithmType::RDP, budgets);
    const auto compressed = core::materialize(database, simplification);
    return per_query_recall(database, workload, evaluator::evaluate_knn(compressed, workload));
}

// Coordinate search over the free exponents, maximising mean training recall@k
Exponents fit_exponents(const std::function<double(Exponents)> &train_recall, Exponents start,
                        std::array<bool, 3> free) {
    static constexpr std::array<double, 9> kValues{-1.0, -0.75, -0.5, -0.25, 0.0,
                                                   0.25, 0.5,   0.75, 1.0};
    Exponents best = start;
    double best_recall = train_recall(best);
    for (int round = 0; round < 2; ++round) {
        bool improved = false;
        for (std::size_t c = 0; c < 3; ++c) {
            if (!free[c])
                continue;
            for (const double value : kValues) {
                Exponents candidate = best;
                (c == 0 ? candidate.size : c == 1 ? candidate.density : candidate.curve) = value;
                const double recall = train_recall(candidate);
                if (recall > best_recall + 1e-12) {
                    best = candidate;
                    best_recall = recall;
                    improved = true;
                }
            }
        }
        if (!improved)
            break;
    }
    return best;
}

void run_features(const pipeline::Config &config, const Data &data) {
    const auto &[database, train_queries, test_queries] = data;
    const std::size_t k = config.knn.k;
    const auto sizes = sizes_of(database);

    auto start = std::chrono::steady_clock::now();
    const auto features = trajectory_features(database, 500.0);
    const auto train = query::generate_knn_queries(database, train_queries,
                                                   {.k = k, .query_count = 0}, config.seed);
    const auto test = query::generate_knn_queries(database, test_queries,
                                                  {.k = k, .query_count = 0}, config.seed);
    std::cout << std::format("features and ground truth in {:.1f}s\n", seconds_since(start));

    for (const double share : config.budget_shares) {
        const std::size_t total = total_budget(database, share);
        start = std::chrono::steady_clock::now();
        std::map<Exponents, double> cache;
        const auto train_recall = [&](Exponents e) {
            const auto it = cache.find(e);
            if (it != cache.end())
                return it->second;
            const double r = mean(recall_of(database, feature_budget(features, sizes, e, total),
                                            train));
            cache.emplace(e, r);
            return r;
        };

        std::cout << std::format("\n{:g}% budget ({} points)\n", share * 100, total);
        std::cout << std::format("  {:<26} {:>22} {:>9} {:>9}   {:>17}   {:>17}\n", "allocation",
                                 "exponents (size,dens,curve)", "train R@k", "test R@k",
                                 "test Δ vs prop", "test Δ vs s^0.25");

        const auto proportional =
            allocation::proportional_budget(database, total_budget(database, share));
        const auto prop_test = recall_of(database, proportional, test);
        const auto fixed = Exponents{.size = 0.25};
        const auto fixed_test = recall_of(database, feature_budget(features, sizes, fixed, total),
                                          test);

        const auto row = [&](std::string_view name, std::optional<Exponents> e,
                             const std::vector<double> &train_scores,
                             const std::vector<double> &test_scores) {
            const auto [dp, sp] = paired_difference(test_scores, prop_test);
            const auto [df, sf] = paired_difference(test_scores, fixed_test);
            const std::string exps =
                e ? std::format("({:+.2f},{:+.2f},{:+.2f})", e->size, e->density, e->curve) : "-";
            std::cout << std::format(
                "  {:<26} {:>22} {:>9.4f} {:>9.4f}   {:>+8.4f} ± {:.4f}   {:>+8.4f} ± {:.4f}\n",
                name, exps, mean(train_scores), mean(test_scores), dp, sp, df, sf);
        };
        row("size-proportional", std::nullopt, recall_of(database, proportional, train),
            prop_test);
        row("size^0.25 (fixed)", fixed,
            recall_of(database, feature_budget(features, sizes, fixed, total), train), fixed_test);

        const std::array<std::pair<std::string_view, std::array<bool, 3>>, 4> variants{{
            {"fit: size", {true, false, false}},
            {"fit: size+density", {true, true, false}},
            {"fit: size+curve", {true, false, true}},
            {"fit: size+density+curve", {true, true, true}},
        }};
        for (const auto &[name, free] : variants) {
            const auto e = fit_exponents(train_recall, fixed, free);
            const auto budgets = feature_budget(features, sizes, e, total);
            row(name, e, recall_of(database, budgets, train), recall_of(database, budgets, test));
        }
        std::cout << std::format("  ({} train evaluations, {:.1f}s)\n", cache.size(),
                                 seconds_since(start));
    }
}

// RDP insertion order: endpoints first, then splits in the order rdp::simplify takes them. The
// first n entries are exactly the points rdp::simplify(trajectory, n) keeps.
std::vector<std::uint32_t> rdp_order(const core::TrajectoryView &trajectory) {
    struct Split {
        double error;
        std::size_t index;
        std::size_t first;
        std::size_t last;
        bool operator<(const Split &other) const { return error < other.error; }
    };
    const std::size_t n = trajectory.size();
    const auto &x = trajectory.x;
    const auto &y = trajectory.y;
    std::priority_queue<Split> heap;
    const auto push = [&](std::size_t first, std::size_t last) {
        if (last - first < 2)
            return;
        const double dx = x[last] - x[first];
        const double dy = y[last] - y[first];
        const double len_sq = dx * dx + dy * dy;
        Split split{-1.0, first + 1, first, last};
        for (std::size_t k = first + 1; k < last; ++k) {
            double s = 0.0;
            if (len_sq > 0.0)
                s = std::clamp(((x[k] - x[first]) * dx + (y[k] - y[first]) * dy) / len_sq, 0.0,
                               1.0);
            const double ex = x[k] - (x[first] + s * dx);
            const double ey = y[k] - (y[first] + s * dy);
            const double d = ex * ex + ey * ey;
            if (d > split.error) {
                split.error = d;
                split.index = k;
            }
        }
        heap.push(split);
    };
    std::vector<std::uint32_t> order{0, static_cast<std::uint32_t>(n - 1)};
    push(0, n - 1);
    while (!heap.empty()) {
        const Split split = heap.top();
        heap.pop();
        order.push_back(static_cast<std::uint32_t>(split.index));
        push(split.first, split.index);
        push(split.index, split.last);
    }
    return order;
}

struct OrderedDb {
    const core::TrajectoryDb &database;
    std::vector<std::vector<std::uint32_t>> orders;

    core::TrajectoryDb compress(const std::vector<std::size_t> &budgets) const {
        std::vector<std::vector<std::size_t>> kept(orders.size());
        core::parallel_for(orders.size(), [&](std::size_t i) {
            const auto n = std::min(budgets[i], orders[i].size());
            kept[i].assign(orders[i].begin(), orders[i].begin() + static_cast<std::ptrdiff_t>(n));
            std::ranges::sort(kept[i]);
        });
        core::Simplification simplification;
        for (const auto &list : kept)
            simplification.add_trajectory(list);
        return core::materialize(database, simplification);
    }
};

std::vector<double> recall_with(const OrderedDb &ordered, const std::vector<std::size_t> &budgets,
                                const query::KnnWorkload &workload) {
    const auto compressed = ordered.compress(budgets);
    return per_query_recall(ordered.database, workload,
                            evaluator::evaluate_knn(compressed, workload));
}

constexpr std::array<std::string_view, 9> kFeatureNames{
    "size", "density", "curve", "duration", "interval", "speed", "extent", "nn1", "nn10"};

// Log features, z-scored. Row i holds trajectory i's features in kFeatureNames order.
struct FeatureMatrix {
    std::vector<std::array<double, 9>> rows;
    std::array<double, 9> stddev{}; // of the log feature before z-scoring
};

FeatureMatrix extended_features(const core::TrajectoryDb &database) {
    const std::size_t n = database.size();
    const auto basic = trajectory_features(database, 500.0); // size, density, curve
    FeatureMatrix matrix{std::vector<std::array<double, 9>>(n)};

    core::parallel_for(n, [&](std::size_t i) {
        const auto t = database.trajectory(i);
        double length = 0.0;
        for (std::size_t p = 1; p < t.size(); ++p)
            length += std::hypot(t.x[p] - t.x[p - 1], t.y[p] - t.y[p - 1]);
        const double duration = std::max(1.0, static_cast<double>(t.t.back() - t.t.front()));
        const double extent = std::hypot(t.bbox.x_max - t.bbox.x_min, t.bbox.y_max - t.bbox.y_min);

        // Endpoint lower bound on Fréchet to every other trajectory: nearest and 10th nearest
        std::vector<double> bounds;
        bounds.reserve(n - 1);
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i)
                continue;
            const auto o = database.trajectory(j);
            const double start = std::hypot(t.x.front() - o.x.front(), t.y.front() - o.y.front());
            const double end = std::hypot(t.x.back() - o.x.back(), t.y.back() - o.y.back());
            bounds.push_back(std::max(start, end));
        }
        const std::size_t tenth = std::min<std::size_t>(9, bounds.size() - 1);
        std::ranges::nth_element(bounds, bounds.begin() + static_cast<std::ptrdiff_t>(tenth));
        const double nn10 = bounds[tenth];
        const double nn1 = *std::min_element(bounds.begin(),
                                             bounds.begin() + static_cast<std::ptrdiff_t>(tenth + 1));

        matrix.rows[i] = {basic.size[i],
                          basic.density[i],
                          basic.curve[i],
                          duration,
                          duration / static_cast<double>(t.size() - 1),
                          std::max(length, 1.0) / duration,
                          std::max(extent, 1.0),
                          std::max(nn1, 1.0),
                          std::max(nn10, 1.0)};
    });

    for (std::size_t f = 0; f < 9; ++f) {
        double sum = 0.0;
        double squares = 0.0;
        for (auto &row : matrix.rows) {
            row[f] = std::log(row[f]);
            sum += row[f];
            squares += row[f] * row[f];
        }
        const double m = sum / static_cast<double>(n);
        const double sd = std::sqrt(std::max(1e-12, squares / static_cast<double>(n) - m * m));
        matrix.stddev[f] = sd;
        for (auto &row : matrix.rows)
            row[f] = (row[f] - m) / sd;
    }
    return matrix;
}

constexpr std::size_t kHidden = 6;

// Parameter layout: linear weights (9) | hidden weights (kHidden x 9) | hidden biases (kHidden)
// | output weights (kHidden). Linear models leave the rest at zero.
constexpr std::size_t kParams = 9 + kHidden * 9 + 2 * kHidden;

std::vector<std::size_t> model_budget(const FeatureMatrix &features,
                                      const std::vector<std::size_t> &sizes,
                                      const std::vector<double> &params, std::size_t total) {
    std::vector<std::size_t> budgets(sizes.size(), allocation::kMinPointsPerTrajectory);
    std::vector<double> weights(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const auto &x = features.rows[i];
        double score = 0.0;
        for (std::size_t f = 0; f < 9; ++f)
            score += params[f] * x[f];
        for (std::size_t h = 0; h < kHidden; ++h) {
            const double out = params[9 + kHidden * 9 + kHidden + h];
            if (out == 0.0)
                continue;
            double a = params[9 + kHidden * 9 + h];
            for (std::size_t f = 0; f < 9; ++f)
                a += params[9 + h * 9 + f] * x[f];
            score += out * std::tanh(a);
        }
        weights[i] = std::exp(std::clamp(score, -30.0, 30.0));
    }
    weighted_fill(budgets, sizes, weights,
                  total - allocation::kMinPointsPerTrajectory * sizes.size());
    return budgets;
}

// (1+1) evolution strategy with the 1/5 success rule, maximising f. Only indices in free move.
std::vector<double> evolve(const std::function<double(const std::vector<double> &)> &f,
                           std::vector<double> x, const std::vector<std::size_t> &free,
                           double sigma, std::size_t evaluations, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    double best = f(x);
    for (std::size_t e = 1; e < evaluations; ++e) {
        auto candidate = x;
        for (const auto i : free)
            candidate[i] += sigma * normal(rng);
        const double value = f(candidate);
        if (value > best) {
            best = value;
            x = std::move(candidate);
            sigma = std::min(1.0, sigma * 1.5);
        } else {
            sigma = std::max(1e-3, sigma * std::pow(1.5, -0.25));
        }
    }
    return x;
}

void run_model(const pipeline::Config &config, const Data &data, const Options &options) {
    const auto &[database, train_queries, test_queries] = data;
    const std::size_t k = config.knn.k;
    const auto sizes = sizes_of(database);

    auto start = std::chrono::steady_clock::now();
    OrderedDb ordered{database, std::vector<std::vector<std::uint32_t>>(database.size())};
    core::parallel_for(database.size(),
                       [&](std::size_t i) { ordered.orders[i] = rdp_order(database.trajectory(i)); });
    // The cached order must reproduce the engine's RDP exactly
    for (std::size_t i = 0; i < database.size(); i += std::max<std::size_t>(1, database.size() / 50)) {
        for (const std::size_t n : {2, 3, 7, 20, 64}) {
            const auto trajectory = database.trajectory(i);
            std::vector<std::size_t> prefix(
                ordered.orders[i].begin(),
                ordered.orders[i].begin() +
                    static_cast<std::ptrdiff_t>(std::min(n, ordered.orders[i].size())));
            std::ranges::sort(prefix);
            if (prefix != selection::rdp::simplify(trajectory, n))
                throw std::logic_error("rdp_order does not match rdp::simplify");
        }
    }
    const auto features = extended_features(database);
    const auto train = query::generate_knn_queries(database, train_queries,
                                                   {.k = k, .query_count = 0}, config.seed);
    const auto test = query::generate_knn_queries(database, test_queries,
                                                  {.k = k, .query_count = 0}, config.seed);
    std::cout << std::format("orders, features and ground truth in {:.1f}s\n",
                             seconds_since(start));

    std::vector<std::size_t> three{0, 1, 2};
    std::vector<std::size_t> nine(9);
    std::iota(nine.begin(), nine.end(), 0);
    std::vector<std::size_t> all(kParams);
    std::iota(all.begin(), all.end(), 0);

    for (const double share : config.budget_shares) {
        const std::size_t total = total_budget(database, share);
        start = std::chrono::steady_clock::now();
        std::size_t evaluations = 0;
        const auto train_recall = [&](const std::vector<double> &params) {
            ++evaluations;
            return mean(recall_with(ordered, model_budget(features, sizes, params, total), train));
        };

        std::cout << std::format("\n{:g}% budget ({} points)\n", share * 100, total);
        std::cout << std::format("  {:<26} {:>9} {:>9}   {:>17}   {:>17}   {:>17}\n", "allocation",
                                 "train R@k", "test R@k", "test Δ vs prop", "test Δ vs s^0.25",
                                 "test Δ vs lin-3");

        const auto proportional = allocation::proportional_budget(database, total);
        const auto prop_test = recall_with(ordered, proportional, test);

        // size^0.25 in z-scored log space
        std::vector<double> fixed(kParams, 0.0);
        fixed[0] = 0.25 * features.stddev[0];
        const auto fixed_test = recall_with(ordered, model_budget(features, sizes, fixed, total), test);

        auto linear3 = evolve(train_recall, fixed, three, 0.15, options.evaluations / 2, 1);
        const auto linear3_test =
            recall_with(ordered, model_budget(features, sizes, linear3, total), test);
        auto linear9 = evolve(train_recall, linear3, nine, 0.15, options.evaluations, 2);

        // MLP residual starts as linear9 (output weights zero) with random hidden weights
        auto mlp_start = linear9;
        std::mt19937_64 rng(3);
        std::normal_distribution<double> init(0.0, 0.5 / 3.0);
        for (std::size_t i = 9; i < 9 + kHidden * 9; ++i)
            mlp_start[i] = init(rng);
        auto mlp = evolve(train_recall, mlp_start, all, 0.1, options.evaluations, 4);

        const auto row = [&](std::string_view name, const std::vector<double> &train_scores,
                             const std::vector<double> &test_scores) {
            const auto [dp, sp] = paired_difference(test_scores, prop_test);
            const auto [df, sf] = paired_difference(test_scores, fixed_test);
            const auto [dl, sl] = paired_difference(test_scores, linear3_test);
            std::cout << std::format("  {:<26} {:>9.4f} {:>9.4f}   {:>+8.4f} ± {:.4f}   "
                                     "{:>+8.4f} ± {:.4f}   {:>+8.4f} ± {:.4f}\n",
                                     name, mean(train_scores), mean(test_scores), dp, sp, df, sf,
                                     dl, sl);
        };
        const auto model_row = [&](std::string_view name, const std::vector<double> &params) {
            const auto budgets = model_budget(features, sizes, params, total);
            row(name, recall_with(ordered, budgets, train), recall_with(ordered, budgets, test));
        };
        row("size-proportional", recall_with(ordered, proportional, train), prop_test);
        model_row("size^0.25 (fixed)", fixed);
        model_row("linear, 3 features", linear3);
        model_row("linear, 9 features", linear9);
        model_row("MLP residual, 9 features", mlp);

        std::cout << "  linear-9 weights per log-feature std:";
        for (std::size_t f = 0; f < 9; ++f)
            std::cout << std::format(" {}={:+.2f}", kFeatureNames[f], linear9[f] / features.stddev[f]);
        std::cout << std::format("\n  ({} train evaluations, {:.1f}s)\n", evaluations,
                                 seconds_since(start));
    }
}

// Exponents (size, density, curve) the --features run fitted on training kNN queries with RDP
std::optional<Exponents> fitted_linear3(double share) {
    if (share == 0.0025)
        return Exponents{.size = 0.25};
    if (share == 0.005)
        return Exponents{.density = 0.25};
    if (share == 0.01)
        return Exponents{.size = 0.25, .density = 0.25, .curve = -0.25};
    if (share == 0.02)
        return Exponents{.size = 0.25, .density = 0.5, .curve = -0.5};
    return std::nullopt;
}

// Every query type at once: does the kNN allocator help or hurt range and where-at?
void run_check(const pipeline::Config &config, const Data &data) {
    const auto &[database, train_queries, test_queries] = data;
    const auto sizes = sizes_of(database);

    auto start = std::chrono::steady_clock::now();
    const auto features = trajectory_features(database, 500.0);
    // kNN on the test half only: the linear-3 exponents were fitted on the training half
    const auto knn = query::generate_knn_queries(database, test_queries,
                                                 {.k = config.knn.k, .query_count = 0}, config.seed);
    const auto where_at = query::generate_where_at_queries(database, config.where_at, config.seed);
    const auto range = query::generate_range_queries(database, config.range, config.seed);
    std::cout << std::format("workloads: {} knn, {} where-at probes, {} range in {:.1f}s\n",
                             knn.entries.size(), where_at.queries.size(), range.queries.size(),
                             seconds_since(start));

    for (const double share : config.budget_shares) {
        const std::size_t total = total_budget(database, share);
        start = std::chrono::steady_clock::now();
        std::cout << std::format("\n{:g}% budget ({} points)\n", share * 100, total);
        std::cout << std::format("  {:<32} {:>9} {:>9} {:>9} {:>9} {:>9}\n", "selector + allocation",
                                 "knn R@k", "range", "w-at hit", "w-at 100m", "mean(3)");

        std::vector<std::pair<std::string, std::vector<std::size_t>>> allocations;
        allocations.emplace_back("size-proportional",
                                 allocation::proportional_budget(database, total));
        allocations.emplace_back("size^0.25",
                                 feature_budget(features, sizes, Exponents{.size = 0.25}, total));
        if (const auto e = fitted_linear3(share))
            allocations.emplace_back(std::format("linear-3 ({:+.2f},{:+.2f},{:+.2f})", e->size,
                                                 e->density, e->curve),
                                     feature_budget(features, sizes, *e, total));

        for (const auto algorithm : {selection::AlgorithmType::RDP, selection::AlgorithmType::SED}) {
            for (const auto &[name, budgets] : allocations) {
                const auto simplification =
                    selection::simplify_database(database, algorithm, budgets);
                const auto compressed = core::materialize(database, simplification);
                const evaluator::Evaluator evaluator(
                    database, compressed,
                    {.knn = &knn, .where_at = where_at.queries, .range = range.queries},
                    config.where_at_scoring);
                const auto scores = evaluator.evaluate().scores;
                const double combined = (scores.knn.recall_at_k + scores.range_recall +
                                         scores.where_at.average_hit) /
                                        3.0;
                std::cout << std::format("  {:<32} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f}\n",
                                         std::format("{} {}",
                                                     algorithm == selection::AlgorithmType::RDP
                                                         ? "RDP"
                                                         : "SED",
                                                     name),
                                         scores.knn.recall_at_k, scores.range_recall,
                                         scores.where_at.average_hit, scores.where_at.within_radius,
                                         combined);
            }
        }
        std::cout << std::format("  ({:.1f}s)\n", seconds_since(start));
    }
}

struct MultiScores {
    double knn = 0.0;
    double range = 0.0;
    double where_at = 0.0;
    double mean() const { return (knn + range + where_at) / 3.0; }
};

struct MultiWorkload {
    query::KnnWorkload knn;
    query::WhereAtWorkload where_at;
    query::RangeWorkload range;
};

MultiScores multi_scores(const core::TrajectoryDb &database, const std::vector<std::size_t> &budgets,
                         const MultiWorkload &workload,
                         const evaluator::WhereAtScoringConfig &scoring) {
    const auto simplification =
        selection::simplify_database(database, selection::AlgorithmType::SED, budgets);
    const auto compressed = core::materialize(database, simplification);
    const evaluator::Evaluator evaluator(database, compressed,
                                         {.knn = &workload.knn,
                                          .where_at = workload.where_at.queries,
                                          .range = workload.range.queries},
                                         scoring);
    const auto scores = evaluator.evaluate().scores;
    return {scores.knn.recall_at_k, scores.range_recall, scores.where_at.average_hit};
}

// The 3-feature allocator trained on the mean of kNN recall@k, range recall and where-at
// average hit on training workloads, with SED as selector, then scored on test workloads
void run_combined(const pipeline::Config &config, const Data &data, const Options &options) {
    const auto &[database, train_queries, test_queries] = data;
    const std::size_t k = config.knn.k;
    const auto sizes = sizes_of(database);

    auto start = std::chrono::steady_clock::now();
    const auto features = extended_features(database);
    const auto grid_features = trajectory_features(database, 500.0);
    auto range_config = config.range;
    range_config.query_count = 2000;
    const auto workload = [&](const core::TrajectoryDb &queries, std::uint64_t seed) {
        return MultiWorkload{
            query::generate_knn_queries(database, queries, {.k = k, .query_count = 0}, seed),
            query::generate_where_at_queries(database, config.where_at, seed),
            query::generate_range_queries(database, range_config, seed)};
    };
    const auto train = workload(train_queries, config.seed + 1);
    const auto test = workload(test_queries, config.seed + 2);
    std::cout << std::format("workloads: train {} knn, {} range, {} where-at; test {} knn, {} "
                             "range, {} where-at ({:.1f}s)\n",
                             train.knn.entries.size(), train.range.queries.size(),
                             train.where_at.queries.size(), test.knn.entries.size(),
                             test.range.queries.size(), test.where_at.queries.size(),
                             seconds_since(start));

    const std::vector<std::size_t> three{0, 1, 2};
    for (const double share : config.budget_shares) {
        const std::size_t total = total_budget(database, share);
        start = std::chrono::steady_clock::now();
        const auto budget_of = [&](const std::vector<double> &params) {
            return model_budget(features, sizes, params, total);
        };
        const auto reward = [&](const std::vector<double> &params) {
            return multi_scores(database, budget_of(params), train, config.where_at_scoring).mean();
        };

        std::vector<double> fixed(kParams, 0.0);
        fixed[0] = 0.25 * features.stddev[0];
        const auto learned = evolve(reward, fixed, three, 0.15, options.evaluations, 5);

        std::cout << std::format("\n{:g}% budget ({} points), SED selector\n", share * 100, total);
        std::cout << std::format("  {:<28} {:>10} | {:>8} {:>8} {:>8} {:>8}\n", "allocation",
                                 "train mean", "knn", "range", "w-at", "mean");
        const auto row = [&](std::string_view name, const std::vector<std::size_t> &budgets) {
            const auto tr = multi_scores(database, budgets, train, config.where_at_scoring);
            const auto te = multi_scores(database, budgets, test, config.where_at_scoring);
            std::cout << std::format("  {:<28} {:>10.4f} | {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}\n",
                                     name, tr.mean(), te.knn, te.range, te.where_at, te.mean());
        };
        row("size-proportional", allocation::proportional_budget(database, total));
        row("size^0.25", budget_of(fixed));
        if (const auto e = fitted_linear3(share))
            row("linear-3 trained on kNN only", feature_budget(grid_features, sizes, *e, total));
        row("linear-3 trained on all three", budget_of(learned));
        std::cout << std::format(
            "  learned exponents (size, density, curve): {:+.3f} {:+.3f} {:+.3f}  "
            "({} evaluations, {:.1f}s)\n",
            learned[0] / features.stddev[0], learned[1] / features.stddev[1],
            learned[2] / features.stddev[2], options.evaluations, seconds_since(start));
    }
}

void run(const Options &options) {
    const auto config = pipeline::load_config(options.config_path);
    const std::size_t k = config.knn.k;

    auto start = std::chrono::steady_clock::now();
    const Data data = load(config, options);
    const auto &[database, train_queries, test_queries] = data;
    const auto sizes = sizes_of(database);
    std::cout << std::format("database: {} trajectories, {} points (mean {:.0f} per trajectory)\n",
                             database.size(), database.num_points(),
                             static_cast<double>(database.num_points()) /
                                 static_cast<double>(database.size()));
    std::cout << std::format("queries: {} train, {} test, k={}, candidates={}k, RDP selector\n",
                             train_queries.size(), test_queries.size(), k,
                             options.candidate_factor);
    std::cout << std::format("loaded in {:.1f}s\n", seconds_since(start));
    if (options.features) {
        run_features(config, data);
        return;
    }
    if (options.combined) {
        run_combined(config, data, options);
        return;
    }
    if (options.check) {
        run_check(config, data);
        return;
    }
    if (options.model) {
        run_model(config, data, options);
        return;
    }

    // Every allocation the table is asked about must be on the grid
    std::vector<std::vector<std::size_t>> extra_counts(database.size());
    std::vector<std::vector<std::size_t>> proportional;
    std::vector<std::vector<std::size_t>> half_proportional;
    std::vector<std::vector<std::size_t>> half_uniform;
    for (const double share : config.budget_shares) {
        proportional.push_back(
            allocation::proportional_budget(database, total_budget(database, share)));
        half_proportional.push_back(
            allocation::proportional_budget(database, total_budget(database, share / 2)));
        half_uniform.push_back(power_budget(sizes, 0.0, total_budget(database, share / 2)));
        for (std::size_t i = 0; i < database.size(); ++i) {
            extra_counts[i].push_back(proportional.back()[i]);
            extra_counts[i].push_back(half_proportional.back()[i]);
            extra_counts[i].push_back(half_uniform.back()[i]);
        }
    }

    start = std::chrono::steady_clock::now();
    const auto table = build_table(database, train_queries, k, options.candidate_factor,
                                   options.grid_cap, extra_counts);
    std::size_t cells = 0;
    for (const auto &grid : table.grid)
        cells += grid.size();
    std::cout << std::format("table: {} candidate trajectories, {:.1f} grid points each, {:.1f}s\n",
                             table.members.size(),
                             static_cast<double>(cells) /
                                 static_cast<double>(std::max<std::size_t>(1, table.members.size())),
                             seconds_since(start));

    start = std::chrono::steady_clock::now();
    const auto train = workload_from(train_queries, table, k);
    const auto test = query::generate_knn_queries(database, test_queries, {.k = k, .query_count = 0},
                                                  config.seed);
    std::cout << std::format("test ground truth in {:.1f}s\n", seconds_since(start));

    for (std::size_t s = 0; s < config.budget_shares.size(); ++s) {
        const double share = config.budget_shares[s];
        const std::size_t total = total_budget(database, share);
        start = std::chrono::steady_clock::now();
        std::cout << std::format("\n{:g}% budget ({} points)\n", share * 100, total);
        std::cout << std::format("  {:<30} {:>9} {:>9}   {:>17}   {:>17}   {:>9}\n", "allocation",
                                 "train R@k", "test R@k", "train Δ vs prop", "test Δ vs prop",
                                 "predicted");

        const auto baseline = score(database, proportional[s], train, test);
        print_row("size-proportional", baseline, baseline,
                  predicted_recall(table, state_for(table, proportional[s])));
        for (const double alpha : {0.5, 0.25, 0.0, -0.5}) {
            const auto name = alpha == 0.0 ? std::string("uniform") : std::format("size^{}", alpha);
            print_row(name, score(database, power_budget(sizes, alpha, total), train, test),
                      baseline, std::numeric_limits<double>::quiet_NaN());
        }

        const std::vector<std::size_t> minimum(database.size(),
                                               allocation::kMinPointsPerTrajectory);
        const auto from_minimum = greedy_oracle(table, sizes, minimum, total);
        print_row(std::format("oracle from 2 ({} pts moved)", from_minimum.points_moved),
                  score(database, from_minimum.budgets, train, test), baseline,
                  from_minimum.predicted_train_recall);

        const auto from_half = greedy_oracle(table, sizes, half_proportional[s], total);
        print_row(std::format("oracle on half ({} pts moved)", from_half.points_moved),
                  score(database, from_half.budgets, train, test), baseline,
                  from_half.predicted_train_recall);

        const auto from_uniform = greedy_oracle(table, sizes, half_uniform[s], total);
        print_row(std::format("oracle on half unif ({} moved)", from_uniform.points_moved),
                  score(database, from_uniform.budgets, train, test), baseline,
                  from_uniform.predicted_train_recall);
        std::cout << std::format("  ({:.1f}s)\n", seconds_since(start));
    }
}

} // namespace

int main(int argc, char *argv[]) {
    try {
        run(parse_options(argc, argv));
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
