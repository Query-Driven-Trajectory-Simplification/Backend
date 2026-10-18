#include "backend/engine/selection/sed.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <queue>
#include <vector>
#include <algorithm>

namespace selection::sed {

double synchronous_factor(double point_time, double start_time, double end_time) {
    return (point_time - start_time) / (end_time - start_time);
}

double synchronous_distance(const core::TrajectoryView &trajectory, std::size_t first,
                            std::size_t last, std::size_t point) {
    if(trajectory.t[first] == trajectory.t[last]){
        return 0.0;
    }

    double c = synchronous_factor(static_cast<double>(trajectory.t[point]),
                                  static_cast<double>(trajectory.t[first]),
                                  static_cast<double>(trajectory.t[last]));

    double sync_x = trajectory.x[first] + c * (trajectory.x[last] - trajectory.x[first]);
    double sync_y = trajectory.y[first] + c * (trajectory.y[last] - trajectory.y[first]);
    double sync_distance = std::hypot(sync_x - trajectory.x[point], sync_y - trajectory.y[point]);

    return sync_distance;
}

struct point_t {
    std::size_t prev;
    std::size_t next;
};
struct removal_t {
    double neightbour_sed;
    std::size_t index;
    std::size_t version;

    bool operator<(const removal_t &other) const { return neightbour_sed > other.neightbour_sed; }
};

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep) {

    size_t n = trajectory.size();
    if (points_to_keep >= n || n <= 2) {
        std::vector<std::size_t> all_points_kept(n);
        for (std::size_t i = 0; i < n; ++i)
            all_points_kept[i] = i;
        return all_points_kept;
    }

    std::vector<bool> keep_indicies(n, true);

    std::priority_queue<removal_t> heap;
    std::vector<point_t> points(n);
    points[0] = {.prev = 0,.next=1};
    points[n-1] = {.prev= n-2, .next = n-1};
    std::vector<std::size_t> versions(n,0);
  
    for (std::size_t i = 1; i < n - 1; ++i) {
        points[i] = {i-1, i+1};
        heap.push({.neightbour_sed = synchronous_distance(trajectory, i - 1, i + 1, i),
                   .index = i,
                   .version = 0});
    }
    std::size_t kept_points = n;


    while (kept_points > points_to_keep && !heap.empty()) {
        const removal_t top = heap.top(); heap.pop();
        if(!keep_indicies[top.index] || top.version != versions[top.index]){
            continue;
        }
        const size_t prev = points[top.index].prev;
        const size_t next = points[top.index].next;
        keep_indicies[top.index] = false;
        points[prev].next = next;
        points[next].prev = prev;
        kept_points--;

        for(size_t curr : {prev,next}){
            if (curr == 0 || curr == n-1){
                continue;
            }
            double cost = 0.0;
            const size_t first = points[curr].prev;
            const size_t last = points[curr].next;
            for (size_t k = first + 1; k < last; ++k)
                cost = std::max(cost, synchronous_distance(trajectory, first, last, k));
            versions[curr]++;
            heap.push({cost,curr,versions[curr]});
        }

        
    }
    std::vector<size_t> simplified_trajectory;
    for(size_t i = 0; i < points.size(); ++i){

        if(keep_indicies[i]) simplified_trajectory.push_back(i);

    }
    return simplified_trajectory;
}

} // namespace selection::sed