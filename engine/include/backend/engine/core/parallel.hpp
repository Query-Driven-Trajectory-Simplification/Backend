#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

namespace core {

// Calls fn(i) for every i in [0, n), spread over up to hardware_concurrency threads.
template <typename Fn> void parallel_for(std::size_t n, Fn &&fn) {
    if (n == 0) {
        return;
    }

    // get cores
    const unsigned int available = std::thread::hardware_concurrency();
    const std::size_t num_threads =
        std::min(n, static_cast<std::size_t>(available == 0 ? 1 : available));

    std::vector<std::exception_ptr> errors(num_threads);
    std::atomic<std::size_t> next_index{0};

    {
        std::vector<std::jthread> workers;
        workers.reserve(num_threads);

        for (std::size_t t = 0; t < num_threads; ++t) {
            workers.emplace_back([&, t] {
                try {
                    while (true) {
                        const std::size_t i = next_index.fetch_add(1, std::memory_order_relaxed);

                        if (i >= n) {
                            break;
                        }

                        fn(i);
                    }
                } catch (...) {
                    errors[t] = std::current_exception();
                    next_index.store(n, std::memory_order_relaxed);
                }
            });
        }
    }

    for (const auto &error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

} // namespace core
