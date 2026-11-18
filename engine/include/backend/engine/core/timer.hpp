#pragma once

#include <chrono>

namespace core {

// Wall-clock seconds since start
inline double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace core
