# Range query data (`range_queries.json`)

The backend writes `range_queries.json` next to `heatmap.json` (see `heatmap.md`). It lists the range queries that every simplification method is tested on. The frontend draws them as a second layer that can be switched on and off, on top of the heat map.

## What a range query is

- A range query is **exactly one heat-map cell**: a box of 0.02° latitude × 0.02° longitude (about 2.2 km × 1.7 km) over one 7-day window.
- The query asks: *which trajectories have at least one point inside this box during this week?*

The queries are drawn from the heat map:

- Only cells with `weight ≥ min_weight` (currently 2) can be drawn.
- A cell's chance of being drawn is proportional to its weight, so busy areas get more queries.
- Draws are with replacement, so the same cell can appear twice in one seed. That's rare, about once per 100.
- There is one set of `queries_per_seed` (100) queries per **seed**. Results are reported per seed, then averaged.

## Format

Each query uses the same cell fields as `heatmap.json`, so the same `HeatCell` type and drawing code apply.

```json
{
  "min_weight": 2,
  "queries_per_seed": 100,
  "seeds": [
    { "seed": 1, "queries": [
      { "lat_idx": 28, "lon_idx": 34, "t_idx": 87,
        "min_lat": 39.96, "max_lat": 39.98, "min_lon": 116.3801, "max_lon": 116.4001,
        "t_start": 1228959092, "t_end": 1229563892, "weight": 43 }
    ]}
  ]
}
```

```ts
import type { HeatCell } from "./heatmap";  // same fields, see heatmap.md

export interface QuerySet {
  seed: number;
  queries: HeatCell[];  // in draw order; the index in this list is the query id
}

export interface RangeQueries {
  min_weight: number;
  queries_per_seed: number;
  seeds: QuerySet[];
}
```

`weight` is the cell's heat-map value: how many trajectories pass through the box, not counting their first and last points. The real answer to a query also counts first and last points, so it can be slightly larger.

## How to draw it

- **Seed selector:** show one seed at a time.
- **Week selector:** reuse the heat map's week selector and show only queries with the selected `t_idx`. Optionally add an "all weeks" mode, since each seed has only 100 queries.
- **Draw each query as an outlined rectangle** with no fill, so the heat map stays visible underneath. Show `weight` and the week (`new Date(t_start * 1000)`) on hover.
- **Toggle:** a checkbox that shows or hides this layer.
- **Repeats:** if a cell appears more than once in the same seed, draw it once with a count label.
- **Matching against the heat map:** to find a query's heat-map cell, match on `lat_idx`, `lon_idx` and `t_idx`. Don't compare the floating-point bounds.

## Size

5 seeds × 100 queries, about 90 KB.
