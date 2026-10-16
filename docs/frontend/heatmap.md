# Heat map data (`heatmap.json`)

The backend writes `heatmap.json`. It shows how many GeoLife trajectories pass through each part of Beijing, week by week. The frontend uses it to draw a heat map layer that can be switched on and off. Later, the range-query boxes will be drawn from the same cells.

## What a cell is

- Beijing is divided into a grid of cells, each **0.02° latitude × 0.02° longitude** (about 2.2 km × 1.7 km).
- Each cell also covers a **7-day window**, so the same patch of map is a different cell every week.
- `weight` is the **number of distinct trajectories** with at least one point in the cell during that week. Each trajectory's first and last point are not counted.
- Only cells with `weight ≥ 1` are in the file. A missing cell means weight 0.

## Format

```json
{
  "cell_size": { "lat": 0.02, "lon": 0.02, "seconds": 604800 },
  "cells": [
    { "lat_idx": 29, "lon_idx": 31, "t_idx": 81,
      "min_lat": 39.98, "max_lat": 40.0, "min_lon": 116.3201, "max_lon": 116.3401,
      "t_start": 1225330292, "t_end": 1225935092, "weight": 295 }
  ]
}
```

```ts
export interface HeatCell {
  lat_idx: number;  // grid row, 0..109 (south to north)
  lon_idx: number;  // grid column, 0..84 (west to east)
  t_idx: number;    // week number, 0 = week starting 2007-04-12 01:31:32 UTC
  min_lat: number; max_lat: number;  // degrees, WGS84; cell covers [min, max)
  min_lon: number; max_lon: number;
  t_start: number; t_end: number;    // Unix seconds, UTC; week covers [t_start, t_end)
  weight: number;   // distinct trajectories in this cell and week
}

export interface HeatMap {
  cell_size: { lat: number; lon: number; seconds: number };
  cells: HeatCell[];  // sorted by t_idx, then lat_idx, then lon_idx
}
```

## How to draw it

- **Pick one week at a time.** Group the cells by `t_idx` and add a week selector (slider or dropdown) that shows `new Date(t_start * 1000)`. If several weeks are drawn at once, their rectangles overlap.
- **Draw each cell as a rectangle** from `[min_lat, min_lon]` to `[max_lat, max_lon]`, colored by `weight`.
- **Use a log color scale.** Most cells have weight 1–5, while the busiest have about 300 (Haidian / Tsinghua, autumn 2008).
- **Add a toggle** to show or hide the layer.
- **Don't add weeks together to get an all-time map.** The same trajectory can appear in many weeks, so the sum counts it more than once. If an all-time view is needed, ask the backend for it.

## Size

About 35,000 cells and 6 MB of JSON, roughly 1 MB once the server gzips it. Load it once with `fetch` and keep it in memory.
