#include "backend/engine/allocation/size_proportional.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>

namespace allocation {



std::vector<std::size_t> proportional_budget(const core::TrajectoryDb& trajectories, std::size_t total_budget) {

    //split budget into start and end points, and the rest of the budget
    std::size_t num_trips = trajectories.size();
    std::vector<std::size_t> budgets;


    for (size_t i = 0; i < num_trips; ++i) {
        std::size_t curr_size = trajectories.trip(i).size();

        std::size_t share = curr_size * total_budget / trajectories.num_points();
        if(share > curr_size) {
            share = curr_size;
        }
        budgets.push_back(share);
    }
    //TODO: IMPLEMENT FIND LEFTOVERS
    //TODO: ASSIGN LEFTOVERS TO LARGEST REMAINDERS
    //TODO: MAKE SURE NO TRIP GETS LESS THAN 2 POINTS
    return budgets;

}

}