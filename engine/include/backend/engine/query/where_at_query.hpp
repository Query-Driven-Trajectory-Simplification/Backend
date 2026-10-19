// notes for later
// requirements. Sometimes have x sometimes y, like 100m where-at, 50 etc, sometimes both
// generate_where_at_queries(trajectories, amount, ???) ??? = where at 50m, 10m? idk yet
// Should be randomly distributed based on the points. This is somewhat tough.
// Currently we can only see whole trajectories, taking random trajectory would favor short
// trajectories We should maybe pick random values between 0 and points.total(). But then we need a
// way to index into the trajectory - that has that specific point, which we do not have. We can
// some binary search thingy using offset to find the trajectory, and then index with the remainder
