# Query-Driven Trajectory Simplification

C++ backend for simplifying GPS trajectories under a point budget and measuring how well the
simplified data still answers kNN, range and where-at-time queries.

## Requirements

- CMake 4.0+, Ninja and a C++23 compiler (Clang recommended)
- Apache Arrow + Parquet and nlohmann_json
- GoogleTest and magic_enum are downloaded automatically by CMake

**Windows (vcpkg):**

```sh
vcpkg install arrow[parquet] nlohmann-json --triplet x64-windows
```

**macOS (Homebrew):**

```sh
brew install cmake ninja apache-arrow nlohmann-json
```

## Build

```sh
# Windows
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
# macOS
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

cmake --build build
```

This produces `build/data_processing/convert` and `build/engine/run`.

## Data

Data is not tracked by git. Download [GeoLife Trajectories 1.3](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
and copy the contents of its `Data/` folder into `data/geolife_raw/`:

```
data/
├── geolife_raw/
│   ├── 000/Trajectory/*.plt
│   ├── 001/Trajectory/*.plt
│   └── ...
└── geolife.parquet        <- created by convert
```

Convert the raw files to a single filtered Parquet file ([convert.cpp](data_processing/apps/convert.cpp)):

```sh
./build/data_processing/convert data/geolife_raw data/geolife.parquet
```

## Run

Run from the repository root, since paths in the config are relative to it:

```sh
./build/engine/run engine/configs/geolife.json
```

Results are printed to stdout.

### Config

`data_path` is the only key without a default and must be set. Any other key can be left out,
and it then falls back to its default. [engine/configs/geolife.json](engine/configs/geolife.json) sets every key.

| Key | Meaning |
| --- | --- |
| `data_path` | Parquet file produced by `convert` |
| `seed` | Seed for all random sampling |
| `budget_shares` | Share of interior points to retain after reserving every endpoint: `B = 2T + round(share × (P − 2T))`. Report `B/P` for total retention |
| `split` | `type` (`Random` or `Time`) and `query_percentage` held out for queries |
| `algorithms` | Simplification algorithms to run: `RDP`, `SED`, `UNIFORM`, `RDP_GLOBAL`. The global variant allocates the total budget by PED and runs once per share, independently of `allocation_alphas` |
| `knn` | `k` and `query_count` (`0` = every trajectory in the query split) |
| `range` | Box size (m), time window (s) and sampling limits for range queries |
| `where_at` | Probes per trajectory and max time gap (s) between points to probe |
| `where_at_scoring` | Tolerance range (m) for the primary average-hit score, and the hit radius (m) for the within-radius diagnostic |
| `frechet` | Fréchet variant for kNN, join and clustering: `Discrete` (kept vertices only, the default) or `Continuous` (polyline distance, unaffected by adding a vertex on an edge) |
| `join` | `mode`, threshold `eps_m` (m), `query_count` and `self_join`. `HeldOut` (the default) joins `query_count` original trajectories from the query split (`0` = all) against the simplified database. `Self` joins the database with itself, both sides simplified. `self_join: true` also scores the self join next to a held-out join, as a diagnostic outside the combined score. `score_eps_m` lists more thresholds: the combined score's join term then averages the normalised join at `eps_m` and at each of them |
| `clustering` | Route clustering metric: `enabled`, neighbour threshold `eps_m` (m) and `min_size` per cluster. Reuses the self join when one is scored with the same eps |
| `allocation_alphas` | Allocation exponents to test: budgets follow size^α (`1` = size-proportional, `0` = equal). Exports for α ≠ 1 get an `_a<α>` suffix |

`RDP_GLOBAL` is database-wide top-down PED: reserve endpoints, then repeatedly split the
segment with the largest perpendicular error anywhere in the database. It uses no query
workloads and receives exactly the same total point budget as the teacher.

Continuous kNN uses the same fixed millimetre-grid ranking in the evaluator and teacher tracker,
with ties resolved by database index. Teacher workload caches from earlier ranking versions are
rebuilt automatically. Recompute older kNN/J4 results when comparing runs across this change.

The teacher's range term uses macro F1 like J4. The loop tracks box coverage counts exactly;
independent choices and value curves use F1 marginal gains against fixed partners.

## Tests

```sh
ctest --test-dir build
```

## Project structure

```
data_processing/        Raw GeoLife -> Parquet
├── apps/convert.cpp    Entry point
├── readers/            .plt reader
└── filters/            Cleaning filters (bounding box, time gaps, duplicates, min size)

engine/                 Simplification and evaluation
├── apps/run.cpp        Entry point
├── configs/            Experiment configs
├── core/               Trajectory database, projection, interpolation
├── io/                 Parquet loading
├── split/              Train/query split
├── selection/          Simplification algorithms (RDP, SED, UNIFORM)
├── allocation/         Distributing the point budget across trajectories
├── query/              Query workload generation (kNN, range, where-at)
├── evaluator/          Scoring simplified data against the original
└── pipeline/           Config parsing and the end-to-end run
```

Each engine module has its header in `include/backend/engine/<module>/` and its source in `src/<module>/`.
