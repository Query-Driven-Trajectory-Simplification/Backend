
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/selection/rdp.hpp"
#include "backend/engine/selection/sed.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

namespace selection {

std::vector<std::size_t> simplify_trajectory(const core::TrajectoryView &trajectory,
                                             AlgorithmType algorithm, std::size_t points_to_keep) {

    switch (algorithm) {
    case AlgorithmType::RDP:
        return rdp::simplify(trajectory, points_to_keep);
    case AlgorithmType::SED:
        return sed::simplify(trajectory, points_to_keep);
    }

    throw std::invalid_argument("simplify_trajectory: unknown algorithm");
}

core::Simplification simplify_database(const core::TrajectoryDb &trajectories,
                                       AlgorithmType algorithm,
                                       std::span<const std::size_t> budgets) {

    if (budgets.size() != trajectories.size()) {
        throw std::invalid_argument("simplify_database: need one allocation budget per trajectory");
    }
    if (trajectories.size() == 0) {
        throw std::invalid_argument("simplify_database: no trajectories to be simplified");
    }

    const std::size_t n = trajectories.size();

    core::Simplification simplification;

    if (n == 0) {
        return simplification;
    }

    const unsigned int available = std::thread::hardware_concurrency();
    const std::size_t num_threads =
        std::min(n, static_cast<std::size_t>(available == 0 ? 1 : available));

    std::vector<std::vector<std::size_t>> results(n);
    std::vector<std::exception_ptr> errors(num_threads);

    std::atomic<std::size_t> next_index{0};

    {
        std::vector<std::jthread> workers;
        workers.reserve(num_threads);

        for (std::size_t t = 0; t < num_threads; ++t) {
            workers.emplace_back([&, t] {
                try {
                    while (true) {
                        // Atomically claim a unique trajectory index.
                        const std::size_t i = next_index.fetch_add(1, std::memory_order_relaxed);

                        if (i >= n) {
                            break;
                        }

                        results[i] =
                            simplify_trajectory(trajectories.trajectory(i), algorithm, budgets[i]);
                    }
                } catch (...) {
                    errors[t] = std::current_exception();
                }
            });
        }
    }

    for (const auto &error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }

    for (auto &result : results) {
        simplification.add_trajectory(std::move(result));
    }

    return simplification;
}

} // namespace selection
