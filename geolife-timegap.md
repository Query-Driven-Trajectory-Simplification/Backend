# GeoLife: Effect of the Time-Gap Split

`data_processing` splits a GeoLife track into separate trajectories wherever two consecutive points are more than `kMaxGapSeconds` apart (`data_processing/include/backend/data/filters/timegap.hpp`). Changing that gap from 5 to 60 minutes gives fewer, longer trajectories.

| | 5-minute gap (before) | 60-minute gap (now) |
|---|---|---|
| Trajectories | 34,193 | **19,800** |
| Total points | ~18.9M | **19.3M** |
| Mean points per trajectory | 552 | **973** |
| Median | 352 | **527** |
| 5th–95th percentile | 115–1,654 | **130–3,262** |
| Longest | 12,488 | **35,498** |
| Length CV | 1.12 | **1.40** |
| Var(log n) | 0.68 | **0.97** |

The 60-minute version is `data/geolife.parquet` as regenerated on 2026-11-04 at 20:08.

**Duration with the 60-minute gap:** median 39 minutes, mean 74 minutes, 95th percentile about 4.3 hours, longest about 70 hours.

## What it means

- **Lengths vary more** (Var(log n) 0.68 → 0.97), so the allocation effect on GeoLife should be larger, between the old GeoLife and the Gulf AIS data (1.35).
- **The GeoLife results in [findings.md](findings.md)** (sections 5 and 13–16) were computed on the 5-minute version and will not reproduce on the new file. Keep a copy of the old parquet, or rerun those experiments and record which version each result uses.
- **Trajectories can now contain stops of up to an hour,** e.g. a parked hour between two trips. Where-at is unaffected, since it only probes intervals with gaps of at most 60 s. kNN and join now compare longer, multi-trip trajectories. State the gap setting in the report.
