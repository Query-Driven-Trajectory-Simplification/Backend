# Final 5-Seed Teacher Benchmark Results (`2026-11-24`)

## TL;DR

- **Setup:** 5 completely fresh held-out real splits (`seeds = 7, 99, 123, 2024, 777`; dev seed `42` excluded), all 4 budgets (`0.25%, 0.5%, 1%, 2%`), full GeoLife (`19,800` trajectories / `19,258,933` points, `85/15` split → `16,830` DB / `2,970` held-out queries), updated [geolife.json](engine/configs/geolife.json), equal teacher weights `[1, 1, 1, 1, 1, 1]` (zero tuned weights).
- **Headline result:** `BUDGET_LOOP_TRADE` wins **5/5 seeds at every single budget** with very low variance (`std ≤ 0.0034`):
  - **`0.25%`:** **`0.5228 ± 0.0034`** vs. `RDP_a0` `0.3936` (`+0.1292`), `PARTNER_l0.5` `0.4131` (`+0.1096`), `RDP_GLOBAL` `0.2333` (`+0.2895`)
  - **`0.5%`:** **`0.6837 ± 0.0029`** vs. `RDP_a0` `0.5655` (`+0.1182`), `PARTNER_l0.5` `0.5737` (`+0.1099`), `RDP_GLOBAL` `0.3911` (`+0.2926`)
  - **`1%`:** **`0.8247 ± 0.0032`** vs. `RDP_a025` `0.7243` (`+0.1004`), `RDP_a0` `0.7216` (`+0.1031`), `RDP_GLOBAL` `0.6407` (`+0.1840`)
  - **`2%`:** **`0.9225 ± 0.0025`** vs. `RDP_a025` `0.8453` (`+0.0770`), `SED_a0` `0.8415` (`+0.0810`), `RDP_GLOBAL` `0.8190` (`+0.1035`)
- **Close to the best baseline at about half the storage:** Teacher at **`1%`** (`J4 = 0.8247`, `1.20 %` of points kept) beats `RDP_GLOBAL` at **`2%`** (`0.8190`, `2.20 %` kept) and is just below `RDP_a0` (`0.8337`) and the best baseline `RDP_a025` (`0.8453`) at **`2%`**; Teacher at **`0.5%`** (`0.6837`) beats `RDP_GLOBAL` at **`1%`** (`0.6407`).
- **Held-out generalization is real (not just Range/Where-At):** Allocation-only `BUDGET_RDP` (pure RDP points) beats every non-teacher baseline on **held-out Join, unseen Join thresholds (`75, 125, 175 m`), and untrained Route Clustering** at all 4 budgets, and every one except `PARTNER` on **held-out kNN** (`PARTNER_l0.5` leads it by `0.001` to `0.010` at every budget; it uses the join partners, so it is not purely geometric). Adding recipe choice + loop + trading (`BUDGET_LOOP_TRADE`) then yields another `+0.046` to `+0.058` `J4`.
- **Where the gain comes from:** allocation is `44–64 %` of the teacher's gain over `RDP_a0`, recipe choice `29–55 %`, loop + trades `0–7 %` (largest at `0.25%`, where they lift held-out kNN by `+0.054`). Under held-out queries and continuous Fréchet the problem is close to separable once points are priced against one global budget (Section 5).
- **What transfers from training to held-out:** about two thirds of the teacher's training gain (`65–72 %` of its `J_T4` gain over `RDP_a0`). Range and Where-At transfer almost fully, held-out kNN partly, the held-out join least (Section 4).
- **Metric cleanliness:** The updated tolerances (`1 km` Range, `25–200 m` Where-At, `[50, 100, 150, 200] m` Join) give clean headroom all the way to `2%` (`0.39 → 0.57 → 0.72 → 0.85` for best geometric baseline) and `256k` training boxes keep the train–test Range gap tiny (`+0.013` to `+0.026`).

---

## 1. Experimental Setup & Endpoints-Only Floors

- **Codebase:** `opti-identical` (`573c957` + `where_at_within_radius` CSV header rename), parallel `where_at`, 273/273 unit tests passing.
- **Query Suite (`geolife.json`):**
  - **Held-out kNN:** `k = 10`, Continuous Fréchet, all `2,970` held-out queries against simplified DB.
  - **Range:** `20,000` test boxes (`256,000` teacher training boxes), `1 km × 1 km × 7 days`, `min_results = 5`, `full_accept_at = 20`, point-in-box F1.
  - **Where-At:** `128` probes/trajectory (`max_gap = 60 s`), linear ramp `25 m → 200 m` (`avg_hit` primary in `J4`, `@50 m` secondary).
  - **Held-out Join:** Continuous Fréchet macro F1 averaged across 4 floor-normalised thresholds **`[50, 100, 150, 200] m`**; plus 3 completely unseen intermediate thresholds **`[75, 125, 175] m`** checked separately.
  - **Route Clustering (untrained):** Connected-component clusters (`min_size = 5`) averaged across `[50, 100, 150, 200] m` (kept out of `J4`).
- **Budget:** `B = 2T + round(share × (P − 2T))`. Every trajectory keeps its endpoints and the share applies to interior points, so total retention `B/P` is higher than the share: `0.45 %`, `0.70 %`, `1.20 %` and `2.20 %` of points for the four shares (`220×`, `142×`, `83×`, `45×` compression). Quote `B/P` when comparing with papers that budget over all points.
- **Combined metric:** $J_4 = \frac{1}{4}\big(\text{norm}(\text{kNN}) + \text{norm}(\text{Range}) + \text{norm}(\text{Where-At}) + \text{norm}(\text{Join})\big)$, where $\text{norm}(x) = (x - \text{floor}) / (1 - \text{floor})$.

### Endpoints-only Floors (5-seed mean ± std)

| Metric | Endpoints-only Floor |
|---|---:|
| Held-out kNN Recall@10 | `0.4128 ± 0.0066` |
| Range F1 (`1 km`) | `0.4961 ± 0.0012` |
| Where-At avg hit (`25–200 m`) | `0.1495 ± 0.0007` |
| Held-out Join F1 (`150 m` primary) | `0.1179 ± 0.0079` |
| Route Clustering (`150 m`, untrained) | `0.4639 ± 0.0053` |
| Self-Join F1 (`150 m`) | `0.3884 ± 0.0021` |
| Self-kNN Recall@10 | `0.4804 ± 0.0010` |

---

## 2. Main Results Across All 4 Budgets (5-Seed Mean)

### Budget = 0.25% (4.4 points per trajectory, 0.45 % of points kept, 220× compression)

**Normalised (the 4 `J4` terms + untrained Clustering):**

| Method | **J4 (mean ± std)** | Held-out kNN | Range (1 km) | Where-At | Held-out Join | Clust *(untrained)* |
|---|---:|---:|---:|---:|---:|---:|
| **`BUDGET_LOOP_TRADE`** | **0.5228 ± 0.0034** | **0.7262** | 0.4883 | **0.3459** | 0.5306 | 0.5320 |
| `BUDGET_LOOP` | 0.5163 ± 0.0039 | 0.6893 | 0.5026 | 0.3378 | 0.5356 | 0.5569 |
| `BUDGET_CHOICE` | 0.5142 ± 0.0036 | 0.6725 | **0.5096** | 0.3389 | 0.5357 | 0.5568 |
| `BUDGET_RDP` *(alloc only)* | 0.4767 ± 0.0037 | 0.7002 | 0.3548 | 0.2879 | **0.5640** | **0.5963** |
| `PARTNER_l0.5` | 0.4131 ± 0.0049 | 0.7017 | 0.3152 | 0.2521 | 0.3835 | 0.5673 |
| `RDP_a0` | 0.3936 ± 0.0050 | 0.6714 | 0.3127 | 0.2484 | 0.3418 | 0.4637 |
| `RDP_a025` | 0.3691 ± 0.0048 | 0.6380 | 0.3538 | 0.2145 | 0.2701 | 0.3820 |
| `SED_a0` | 0.3620 ± 0.0048 | 0.5655 | 0.3131 | 0.3097 | 0.2595 | 0.2204 |
| `UNI_a0` | 0.3058 ± 0.0026 | 0.4209 | 0.3448 | 0.2888 | 0.1686 | 0.0608 |
| `RDP_GLOBAL` | 0.2333 ± 0.0030 | 0.4934 | 0.3217 | 0.0937 | 0.0242 | 0.2335 |

**Raw (unnormalised):**

| Method | kNN R@10 | Range F1 | Where-At | Where@50m | Join F1 (150m) | **Raw 4-Avg** |
|---|---:|---:|---:|---:|---:|---:|
| **`BUDGET_LOOP_TRADE`** | **0.8392** | **0.7422** | **0.4437** | **0.2973** | 0.6868 | **0.6780** |
| `BUDGET_RDP` | 0.8240 | 0.6749 | 0.3944 | 0.2579 | **0.7161** | 0.6523 |
| `PARTNER_l0.5` | 0.8248 | 0.6549 | 0.3639 | 0.2361 | 0.5190 | 0.5907 |
| `RDP_a0` | 0.8070 | 0.6536 | 0.3608 | 0.2286 | 0.4794 | 0.5752 |
| `SED_a0` | 0.7448 | 0.6539 | 0.4129 | 0.2608 | 0.4127 | 0.5561 |
| `RDP_a025` | 0.7874 | 0.6744 | 0.3320 | 0.1925 | 0.4193 | 0.5533 |
| `RDP_GLOBAL` | 0.7025 | 0.6582 | 0.2292 | 0.1067 | 0.1399 | 0.4325 |

---

### Budget = 0.5% (6.9 points per trajectory, 0.70 % of points kept, 142× compression)

**Normalised:**

| Method | **J4 (mean ± std)** | Held-out kNN | Range (1 km) | Where-At | Held-out Join | Clust *(untrained)* |
|---|---:|---:|---:|---:|---:|---:|
| **`BUDGET_LOOP_TRADE`** | **0.6837 ± 0.0029** | 0.8364 | 0.7061 | **0.5228** | 0.6696 | 0.7402 |
| `BUDGET_LOOP` | 0.6805 ± 0.0025 | 0.8164 | 0.7261 | 0.5144 | 0.6651 | 0.7530 |
| `BUDGET_CHOICE` | 0.6791 ± 0.0025 | 0.7992 | **0.7353** | 0.5171 | 0.6648 | 0.7508 |
| `BUDGET_RDP` *(alloc only)* | 0.6268 ± 0.0044 | 0.8329 | 0.5252 | 0.4405 | **0.7086** | **0.7833** |
| `PARTNER_l0.5` | 0.5737 ± 0.0053 | **0.8426** | 0.4543 | 0.4046 | 0.5931 | 0.7047 |
| `RDP_a0` | 0.5655 ± 0.0063 | 0.8296 | 0.4633 | 0.4136 | 0.5556 | 0.6303 |
| `RDP_a025` | 0.5525 ± 0.0048 | 0.8093 | 0.4987 | 0.3970 | 0.5051 | 0.5486 |
| `SED_a0` | 0.5335 ± 0.0057 | 0.7276 | 0.4870 | 0.4979 | 0.4214 | 0.3817 |
| `UNI_a0` | 0.4418 ± 0.0036 | 0.5452 | 0.5100 | 0.4519 | 0.2600 | 0.1363 |
| `RDP_GLOBAL` | 0.3911 ± 0.0035 | 0.6605 | 0.4942 | 0.2510 | 0.1587 | 0.3621 |

**Raw (unnormalised):**

| Method | kNN R@10 | Range F1 | Where-At | Where@50m | Join F1 (150m) | **Raw 4-Avg** |
|---|---:|---:|---:|---:|---:|---:|
| **`BUDGET_LOOP_TRADE`** | 0.9039 | **0.8519** | **0.5941** | **0.4324** | 0.7991 | **0.7873** |
| `BUDGET_RDP` | 0.9019 | 0.7607 | 0.5241 | 0.3701 | **0.8339** | 0.7552 |
| `PARTNER_l0.5` | **0.9075** | 0.7250 | 0.4936 | 0.3565 | 0.7144 | 0.7101 |
| `RDP_a0` | 0.8999 | 0.7295 | 0.5013 | 0.3562 | 0.6945 | 0.7063 |
| `RDP_a025` | 0.8880 | 0.7474 | 0.4872 | 0.3281 | 0.6537 | 0.6941 |
| `SED_a0` | 0.8400 | 0.7415 | 0.5730 | 0.4091 | 0.5697 | 0.6810 |
| `RDP_GLOBAL` | 0.8007 | 0.7451 | 0.3630 | 0.1810 | 0.2925 | 0.5503 |

---

### Budget = 1% (11.7 points per trajectory, 1.20 % of points kept, 83× compression)

**Normalised:**

| Method | **J4 (mean ± std)** | Held-out kNN | Range (1 km) | Where-At | Held-out Join | Clust *(untrained)* |
|---|---:|---:|---:|---:|---:|---:|
| **`BUDGET_LOOP_TRADE`** | **0.8247 ± 0.0032** | 0.9058 | 0.8812 | **0.7207** | 0.7914 | 0.8604 |
| `BUDGET_LOOP` | 0.8237 ± 0.0028 | 0.9000 | 0.8901 | 0.7179 | 0.7866 | 0.8582 |
| `BUDGET_CHOICE` | 0.8232 ± 0.0030 | 0.8913 | **0.8952** | 0.7201 | 0.7863 | 0.8552 |
| `BUDGET_RDP` *(alloc only)* | 0.7669 ± 0.0022 | 0.9176 | 0.6980 | 0.6172 | **0.8350** | **0.8834** |
| `RDP_a025` | 0.7243 ± 0.0040 | 0.9080 | 0.6326 | 0.5907 | 0.7660 | 0.7412 |
| `RDP_a0` | 0.7216 ± 0.0041 | 0.9158 | 0.6015 | 0.5879 | 0.7810 | 0.7850 |
| `PARTNER_l0.5` | 0.7198 ± 0.0042 | **0.9201** | 0.5935 | 0.5710 | 0.7944 | 0.8395 |
| `SED_a0` | 0.6987 ± 0.0064 | 0.8353 | 0.6590 | 0.6937 | 0.6069 | 0.5736 |
| `RDP_GLOBAL` | 0.6407 ± 0.0019 | 0.8238 | 0.6176 | 0.5069 | 0.6145 | 0.6112 |
| `UNI_a0` | 0.5835 ± 0.0049 | 0.6541 | 0.6738 | 0.6245 | 0.3815 | 0.2590 |

**Raw (unnormalised):**

| Method | kNN R@10 | Range F1 | Where-At | Where@50m | Join F1 (150m) | **Raw 4-Avg** |
|---|---:|---:|---:|---:|---:|---:|
| **`BUDGET_LOOP_TRADE`** | 0.9447 | **0.9401** | **0.7624** | **0.6140** | 0.8926 | **0.8849** |
| `BUDGET_RDP` | 0.9516 | 0.8478 | 0.6744 | 0.5236 | **0.9255** | 0.8498 |
| `RDP_a025` | 0.9460 | 0.8148 | 0.6519 | 0.5040 | 0.8927 | 0.8264 |
| `RDP_a0` | 0.9505 | 0.7992 | 0.6495 | 0.5154 | 0.8912 | 0.8226 |
| `PARTNER_l0.5` | **0.9531** | 0.7952 | 0.6351 | 0.5051 | 0.8866 | 0.8175 |
| `SED_a0` | 0.9032 | 0.8282 | 0.7395 | 0.5968 | 0.7238 | 0.7987 |
| `RDP_GLOBAL` | 0.8966 | 0.8073 | 0.5806 | 0.3691 | 0.9038 | 0.7971 |

---

### Budget = 2% (21.5 points per trajectory, 2.20 % of points kept, 45× compression)

**Normalised:**

| Method | **J4 (mean ± std)** | Held-out kNN | Range (1 km) | Where-At | Held-out Join | Clust *(untrained)* |
|---|---:|---:|---:|---:|---:|---:|
| **`BUDGET_LOOP_TRADE`** | **0.9225 ± 0.0025** | 0.9491 | 0.9613 | 0.8857 | 0.8937 | 0.9350 |
| `BUDGET_CHOICE` | **0.9225 ± 0.0023** | 0.9456 | **0.9640** | **0.8867** | 0.8936 | 0.9265 |
| `BUDGET_LOOP` | 0.9221 ± 0.0020 | 0.9478 | 0.9628 | 0.8854 | 0.8925 | 0.9311 |
| `BUDGET_RDP` *(alloc only)* | 0.8751 ± 0.0021 | 0.9571 | 0.8366 | 0.7777 | **0.9292** | **0.9351** |
| `RDP_a025` | 0.8453 ± 0.0024 | 0.9522 | 0.7554 | 0.7526 | 0.9208 | 0.8919 |
| `SED_a0` | 0.8415 ± 0.0048 | 0.9176 | 0.7980 | 0.8529 | 0.7974 | 0.7701 |
| `RDP_a0` | 0.8337 ± 0.0029 | 0.9553 | 0.7307 | 0.7403 | 0.9083 | 0.9013 |
| `PARTNER_l0.5` | 0.8283 ± 0.0037 | **0.9581** | 0.7225 | 0.7249 | 0.9078 | 0.9244 |
| `RDP_GLOBAL` | 0.8190 ± 0.0021 | 0.9106 | 0.7346 | 0.7111 | 0.9197 | 0.7943 |
| `UNI_a0` | 0.7146 ± 0.0056 | 0.7468 | 0.8110 | 0.7758 | 0.5248 | 0.4405 |

**Raw (unnormalised):**

| Method | kNN R@10 | Range F1 | Where-At | Where@50m | Join F1 (150m) | **Raw 4-Avg** |
|---|---:|---:|---:|---:|---:|---:|
| **`BUDGET_LOOP_TRADE`** | 0.9701 | **0.9805** | **0.9028** | **0.8069** | 0.9599 | **0.9533** |
| `BUDGET_RDP` | 0.9748 | 0.9176 | 0.8109 | 0.6921 | **0.9736** | 0.9192 |
| `SED_a0` | 0.9516 | 0.8982 | 0.8749 | 0.7773 | 0.8881 | 0.9032 |
| `RDP_a025` | 0.9719 | 0.8767 | 0.7896 | 0.6778 | 0.9658 | 0.9010 |
| `RDP_a0` | 0.9738 | 0.8643 | 0.7791 | 0.6729 | 0.9544 | 0.8929 |
| `RDP_GLOBAL` | 0.9475 | 0.8663 | 0.7543 | 0.6187 | 0.9637 | 0.8829 |

---

## 3. Ablation Ladder & Seed-by-Seed Consistency

### Where the Teacher's Gain Comes From (`ΔJ4` step by step, 5-seed mean)

| Budget | `RDP_a0` | `+Budget Alloc` (`BUDGET_RDP`) | `+Recipe Choice` (`BUDGET_CHOICE`) | `+Loop` (`BUDGET_LOOP`) | `+Trades` (`BUDGET_LOOP_TRADE`) | Total vs. `RDP_a0` | Total vs. Best Non-Teacher |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`0.25%`** | 0.3936 | `+0.0831` | `+0.0375` | `+0.0021` | `+0.0065` | **`+0.1292`** | **`+0.1096 ± 0.0021`** (5/5) |
| **`0.5%`** | 0.5655 | `+0.0613` | `+0.0523` | `+0.0014` | `+0.0032` | **`+0.1182`** | **`+0.1099 ± 0.0030`** (5/5) |
| **`1%`** | 0.7216 | `+0.0453` | `+0.0563` | `+0.0005` | `+0.0010` | **`+0.1031`** | **`+0.1004 ± 0.0013`** (5/5) |
| **`2%`** | 0.8337 | `+0.0414` | `+0.0474` | `-0.0004` | `+0.0004` | **`+0.0888`** | **`+0.0770 ± 0.0019`** (5/5) |

Key structural takeaways:
1. **Budget allocation (`BUDGET_RDP`)** is the foundation: it delivers `+0.041` to `+0.083` `J4` using pure RDP vertices alone and wins held-out kNN, held-out Join, unseen Join thresholds, and untrained Route Clustering at every budget.
2. **Allocation plus recipe choice (`BUDGET_CHOICE`)** captures **93–100%** of the total Teacher `J4` gain. Recipe choice alone adds `+0.038` to `+0.056` over `BUDGET_RDP` (`29–55 %` of the gain) by combining RDP, SED, Greedy, and hybrids per trajectory; allocation alone is `44–64 %`.
3. **Loop + pairwise budget trading (`BUDGET_LOOP_TRADE`) rebalances the metrics at tight budgets:** Although the net `J4` lift over `BUDGET_CHOICE` looks modest (`+0.0086` at `0.25%`, `+0.0046` at `0.5%`, `+0.0015` at `1%`), it systematically pulls back excess Range to repair coupled **held-out kNN** (flipping kNN from tying/losing to `RDP_a0` under `BUDGET_CHOICE` into beating it at `0.25%` and `0.5%`):

| Budget | `Δ` Held-out kNN (`CHOICE → TRADE`) | `Δ` Range | `Δ` Where-At | `Δ` Held-out Join | Net `ΔJ4` |
|---|---:|---:|---:|---:|---:|
| **`0.25%`** | **`+0.0537`** (`0.6725 → 0.7262`) | `-0.0213` | `+0.0070` | `-0.0051` | `+0.0086` |
| **`0.5%`** | **`+0.0372`** (`0.7992 → 0.8364`) | `-0.0292` | `+0.0057` | `+0.0048` | `+0.0046` |
| **`1%`** | **`+0.0145`** (`0.8913 → 0.9058`) | `-0.0140` | `+0.0006` | `+0.0051` | `+0.0015` |
| **`2%`** | **`+0.0035`** (`0.9456 → 0.9491`) | `-0.0027` | `-0.0010` | `+0.0001` | `+0.0000` |

Most of this comes from the loop's exact kNN ranking replacing the value curves' per-trajectory kNN check, not from trajectories reacting to each other's choices (Section 5).

---

## 4. Generalization & Robustness Checks

### Unseen Intermediate Held-Out Join Thresholds (`75, 125, 175 m`, 5-seed mean `norm_join`)

| Budget | `BUDGET_RDP` | `BUDGET_LOOP_TRADE` | `PARTNER_l0.5` | `RDP_a0` | `RDP_a025` | `RDP_GLOBAL` | `SED_a0` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`0.25%`** | **0.5888 ± 0.0147** | 0.5514 ± 0.0126 | 0.4084 ± 0.0222 | 0.3631 ± 0.0129 | 0.2875 ± 0.0160 | 0.0222 ± 0.0038 | 0.2814 ± 0.0153 |
| **`0.5%`** | **0.7370 ± 0.0141** | 0.6943 ± 0.0113 | 0.6181 ± 0.0189 | 0.5819 ± 0.0212 | 0.5341 ± 0.0204 | 0.1414 ± 0.0068 | 0.4508 ± 0.0190 |
| **`1%`** | **0.8625 ± 0.0098** | 0.8188 ± 0.0110 | 0.8279 ± 0.0125 | 0.8189 ± 0.0169 | 0.8064 ± 0.0184 | 0.6851 ± 0.0064 | 0.6345 ± 0.0209 |
| **`2%`** | **0.9491 ± 0.0062** | 0.9211 ± 0.0068 | 0.9278 ± 0.0097 | 0.9289 ± 0.0088 | 0.9402 ± 0.0061 | 0.9443 ± 0.0043 | 0.8238 ± 0.0149 |

### Alternative Aggregations (`J4` Arithmetic vs. Geometric vs. Harmonic Mean, 5-seed mean)

| Budget | `BUDGET_LOOP_TRADE` (Arith / Geo / Harm) | `BUDGET_RDP` (Arith / Geo / Harm) | Best Geometric Baseline (Arith / Geo / Harm) |
|---|---|---|---|
| **`0.25%`** | **0.5228 / 0.5051 / 0.4877** | 0.4767 / 0.4481 / 0.4213 | `RDP_a0`: 0.3936 / 0.3653 / 0.3436 |
| **`0.5%`** | **0.6837 / 0.6743 / 0.6646** | 0.6268 / 0.6078 / 0.5894 | `RDP_a0`: 0.5655 / 0.5451 / 0.5275 |
| **`1%`** | **0.8247 / 0.8214 / 0.8179** | 0.7669 / 0.7579 / 0.7490 | `RDP_a025`: 0.7243 / 0.7140 / 0.7041 |
| **`2%`** | **0.9225 / 0.9219 / 0.9213** | 0.8751 / 0.8721 / 0.8691 | `RDP_a025`: 0.8453 / 0.8403 / 0.8353 |

### Train–Test Proxy Gaps (`terms.csv`, `BUDGET_LOOP_TRADE`, 5-seed mean)

| Budget | `kNN` (train − test) | `Range` (train − test) | `Where-At` (train − test) | `Join` (train − test) |
|---|---:|---:|---:|---:|
| **`0.25%`** | `+0.0587` | `+0.0132` | `+0.0005` | `+0.1573` |
| **`0.5%`** | `+0.0489` | `+0.0222` | `+0.0011` | `+0.1838` |
| **`1%`** | `+0.0377` | `+0.0260` | `+0.0023` | `+0.1603` |
| **`2%`** | `+0.0262` | `+0.0175` | `+0.0050` | `+0.0935` |

- `256k` training boxes kept the Range train–test gap down to **`+0.013` to `+0.026`** across all 4 budgets (down ~38% vs. `128k` boxes at `1 km`, for only `+7%` extra wall time).
- **Important finding: the join generalisation gap.** The training join term scores every database trajectory as an *original* query against its compressed partners: one side compressed, as in the held-out join, so the two measure the same thing on different queries. At `1%`, `BUDGET_LOOP_TRADE` trains to `0.952` but reaches `0.791` held out (`+0.010` over `RDP_a0`), whereas pure-RDP `BUDGET_RDP` reaches **`0.835`** held out (`0.863` on unseen thresholds). Under equal-weight `J4`, the optimizer accepts `-0.044` Join and `-0.012` kNN relative to `BUDGET_RDP` to gain `+0.183` Range and `+0.104` Where-At (`+0.058` net `J4`). Report the per-term train/held-out table below in the thesis rather than calling `BUDGET_LOOP_TRADE` a universal ceiling.
- **The gap is a budget effect, not a shape effect.** `BUDGET_RDP` keeps only RDP points and already has most of it (`0.957` training vs `0.835` held out at `1%`): budgets fitted to the training pairs transfer only partly. The recipe-choice step, which also re-allocates budgets, then costs `0.028` to `0.049` held-out join where training says it costs `0.000` to `0.025`. `Teacher vs BUDGET_RDP diagnosis 2026-11-22` (seed 42, earlier metrics) found all of the held-out kNN and join loss against `BUDGET_RDP` on trajectories the teacher gives fewer points. At `0.25%` most of it was one cluster of near-duplicate long trajectories on a repeated route, which `BUDGET_RDP` gives 10–11 points and the teacher 2–3.
- **Not new with the tightened metrics, but wider:** in the 2026-11-22 run (earlier metrics, `1%`) the teacher scored `0.976` training vs `0.882` held out, with `RDP_a0` at `0.875` and `BUDGET_RDP` at `0.909`. The gap grew from `0.094` to `0.161`.

#### Training vs held-out per term (5-seed mean, normalised, training / held-out)

| Budget | Arm | kNN | Range | Where-At | Join |
|---|---|---:|---:|---:|---:|
| **`0.25%`** | `RDP_a0` | 0.672 / 0.671 | 0.313 / 0.313 | 0.248 / 0.248 | 0.349 / 0.342 |
| | `BUDGET_RDP` | 0.725 / 0.700 | 0.357 / 0.355 | 0.288 / 0.288 | 0.715 / 0.564 |
| | `BUDGET_CHOICE` | 0.700 / 0.672 | 0.525 / 0.510 | 0.339 / 0.339 | 0.690 / 0.536 |
| | `BUDGET_LOOP_TRADE` | 0.785 / 0.726 | 0.502 / 0.488 | 0.346 / 0.346 | 0.688 / 0.531 |
| **`1%`** | `RDP_a0` | 0.917 / 0.916 | 0.604 / 0.602 | 0.588 / 0.588 | 0.786 / 0.781 |
| | `BUDGET_RDP` | 0.929 / 0.918 | 0.702 / 0.698 | 0.617 / 0.617 | 0.957 / 0.835 |
| | `BUDGET_CHOICE` | 0.911 / 0.891 | 0.923 / 0.895 | 0.723 / 0.720 | 0.948 / 0.786 |
| | `BUDGET_LOOP_TRADE` | 0.943 / 0.906 | 0.907 / 0.881 | 0.723 / 0.721 | 0.952 / 0.791 |

A query-agnostic method shows no gap (`RDP_a0`). Of the teacher's training gain over `RDP_a0` (`J_T4`), `70 %`, `66 %`, `65 %` and `72 %` reaches held-out `J4` at the four budgets.

#### Join change by step (5-seed mean, training / held-out)

| Budget | Allocation (`RDP_a0 → BUDGET_RDP`) | Recipe choice | Loop + trades |
|---|---:|---:|---:|
| **`0.25%`** | +0.367 / +0.222 | −0.025 / −0.028 | −0.002 / −0.005 |
| **`0.5%`** | +0.313 / +0.153 | −0.025 / −0.044 | +0.005 / +0.005 |
| **`1%`** | +0.171 / +0.054 | −0.008 / −0.049 | +0.003 / +0.005 |
| **`2%`** | +0.074 / +0.021 | −0.000 / −0.036 | +0.001 / +0.000 |

On the unseen thresholds at `1%` the teacher ties `RDP_a0` on held-out join (`0.715 / 0.845 / 0.896` against `0.720 / 0.846 / 0.890` at `75 / 125 / 175 m`), while `BUDGET_RDP` reaches `0.773 / 0.889 / 0.925`.

### Range Distribution Shift & Segment-in-Box Robustness (`seed 7`, frozen `1 km`-trained Teacher)

To test whether the Range gains are an artifact of vertex-only (`point-in-box`) semantics or of the `1 km` data-distribution generator, the **exact same frozen simplifications** on `seed 7` (`0.25%`, `1%`, `2%`; all 51 kept-set hashes match the final run) were re-evaluated under 4 shifted Range workloads. One seed only. The outputs are in the run folder under `range_shift_s7/`; the code that generated them was not saved, so the check cannot be rerun as is.
- **`seg1km`:** 3D `[x, y, t]` Liang–Barsky segment-in-box intersection on the same 20k `1 km` boxes over all 16,830 DB trajectories (counting chord-cutting false positives; floor = `0.6509`, `3.31` avg FP/query).
- **`pt500m`:** Box-size shift down (`10,000` boxes, `500 m × 500 m`; floor = `0.3732`).
- **`pt2km`:** Box-size shift up (`10,000` boxes, `2 km × 2 km`, RL4QDTS/MLSimp size; floor = `0.6109`).
- **`gauss1km`:** Spatial distribution shift (`5,000` `1 km` boxes with centres drawn from a smooth 2D Gaussian fitted to Beijing's spatial mean/std rather than data points; floor = `0.2271`).

#### Normalised Range Score Under Each Shift (`seed 7`)

| Budget | Method | `pt1km` *(in-dist)* | `seg1km` *(segment)* | `pt500m` *(size ↓)* | `pt2km` *(size ↑)* | `gauss1km` *(spatial shift)* |
|---|---|---:|---:|---:|---:|---:|
| **`0.25%`** | **`BUDGET_LOOP_TRADE`** | **0.4920** | 0.6589 | **0.3405** | 0.4925 | 0.2471 |
| | `BUDGET_RDP` | 0.3590 | 0.5983 | 0.2891 | 0.3913 | 0.2265 |
| | `PARTNER_l0.5` | 0.3197 | 0.6588 | 0.2508 | 0.4022 | 0.2223 |
| | `RDP_a025` | 0.3583 | 0.7499 | 0.2622 | 0.4681 | 0.2588 |
| | `RDP_a0` | 0.3161 | 0.6653 | 0.2416 | 0.4052 | 0.2243 |
| | `RDP_GLOBAL` | 0.3238 | **0.8526** | 0.1713 | **0.5057** | **0.2783** |
| | `RECIPE_GREEDY` | 0.5261 | 0.5796 | 0.3326 | 0.4972 | 0.2203 |
| **`1%`** | **`BUDGET_LOOP_TRADE`** | **0.8838** | 0.9567 | **0.6731** | **0.9017** | **0.6998** |
| | `BUDGET_RDP` | 0.6996 | 0.9355 | 0.5815 | 0.7797 | 0.5698 |
| | `RDP_a025` | 0.6355 | 0.9586 | 0.5378 | 0.7416 | 0.5163 |
| | `RDP_a0` | 0.6039 | 0.9365 | 0.5101 | 0.7080 | 0.4866 |
| | `RDP_GLOBAL` | 0.6193 | **0.9722** | 0.5052 | 0.7484 | 0.5401 |
| | `RECIPE_GREEDY` | 0.8560 | 0.9035 | 0.6333 | 0.8578 | 0.5971 |
| **`2%`** | **`BUDGET_LOOP_TRADE`** | **0.9615** | **0.9886** | **0.8076** | **0.9684** | **0.8625** |
| | `BUDGET_RDP` | 0.8369 | 0.9849 | 0.7296 | 0.9021 | 0.7344 |
| | `RDP_a025` | 0.7571 | 0.9854 | 0.6721 | 0.8448 | 0.6520 |
| | `RDP_a0` | 0.7324 | 0.9766 | 0.6496 | 0.8227 | 0.6227 |
| | `RDP_GLOBAL` | 0.7357 | 0.9881 | 0.6283 | 0.8449 | 0.6713 |

#### Implied `J4` Under Each Range Variant (`seed 7`, same kNN / Where-At / Join)

| Budget | Method | `J4(pt1km)` | `J4(seg1km)` | `J4(500m)` | `J4(2km)` | `J4(gauss1km)` |
|---|---|---:|---:|---:|---:|---:|
| **`0.25%`** | **`BUDGET_LOOP_TRADE`** | **0.5184** | **0.5601** | **0.4806** | **0.5186** | **0.4572** |
| | `BUDGET_RDP` | 0.4738 | 0.5336 | 0.4563 | 0.4818 | 0.4406 |
| | Best non-teacher baseline | 0.4091 (`PARTNER`) | 0.4939 (`PARTNER`) | 0.3919 (`PARTNER`) | 0.4298 (`PARTNER`) | 0.3848 (`PARTNER`) |
| | `RDP_a0` | 0.3872 | 0.4746 | 0.3686 | 0.4095 | 0.3643 |
| **`1%`** | **`BUDGET_LOOP_TRADE`** | **0.8205** | **0.8387** | **0.7678** | **0.8250** | **0.7745** |
| | `BUDGET_RDP` | 0.7644 | 0.8234 | 0.7349 | 0.7844 | 0.7319 |
| | Best non-teacher baseline | 0.7201 (`RDP_a025`) | 0.8009 (`RDP_a025`) | 0.6957 (`RDP_a025`) | 0.7466 (`RDP_a025`) | 0.6903 (`RDP_a025`) |
| **`2%`** | **`BUDGET_LOOP_TRADE`** | **0.9185** | **0.9253** | **0.8800** | **0.9202** | **0.8938** |
| | `BUDGET_RDP` | 0.8717 | 0.9087 | 0.8449 | 0.8880 | 0.8461 |
| | Best non-teacher baseline | 0.8433 (`RDP_a025`) | 0.9003 (`RDP_a025`) | 0.8220 (`RDP_a025`) | 0.8652 (`RDP_a025`) | 0.8170 (`RDP_a025`) |

- **`BUDGET_LOOP_TRADE` wins `J4` in all 15/15 (budget × shift) cells**, even though it was trained only on `1 km` point-in-box data-distribution queries.
- **Box-size & spatial shifts transfer strongly:** At `1%–2%`, the `1 km`-trained Teacher beats every baseline by `+0.135` on `500 m`, `+0.124–0.160` on `2 km`, and `+0.160–0.211` on smooth 2D Gaussian centres (`+0.055` to `+0.084` on `J4`).
- **Under segment-in-box the Range advantage itself mostly disappears; the `J4` lead stays.** On `seg1km` Range the teacher (`0.6589`) is below `RDP_a0` (`0.6653`), `RDP_a025` (`0.7499`) and `RDP_GLOBAL` (`0.8526`) at `0.25%`, and below `RDP_a025` and `RDP_GLOBAL` at `1%` (`0.9567` against `0.9586` and `0.9722`). It only edges `RDP_GLOBAL` at `2%` (`0.9886` vs `0.9881`), and plain `RECIPE_GREEDY` drops below `RDP_a0`. So point-in-box semantics does inflate the vertex-placing recipes. `BUDGET_LOOP_TRADE` still leads overall `J4` (`+0.066` at `0.25%`, `+0.038` at `1%`, `+0.025` at `2%` over the best baseline) through the other three terms.
- **Spatial shift at `0.25%`:** on Gaussian centres the teacher's Range (`0.2471`) is below `RDP_GLOBAL` (`0.2783`) and `RDP_a025` (`0.2588`). The transfer claim above holds at `1%–2%`.

---

## 5. Where Coordination Matters

**Short version:** for the held-out headline, coordination between trajectories adds little beyond sharing the global budget. With held-out queries and continuous Fréchet, `J4` is close to a sum of per-trajectory values under one budget, which is what the allocator solves. Collective simplification still matters, through the budget: allocation is the largest single step.

### Which queries couple trajectories

| Query | What it compares | Coupling between trajectories |
|---|---|---|
| Where-At | compressed `i` against its own original | none |
| Range (point-in-box) | compressed `i` against a box | none; per-box F1 only aggregates outcomes |
| Held-out join | original query against compressed `i` | none; per-query F1 only aggregates outcomes |
| Held-out kNN | original query against all compressed trajectories, ranked | competition for top-k places |
| Self join, clustering | compressed `i` against compressed `j` | real |
| Self kNN | compressed `i` against compressed `j`, ranked | real |

With `δ_i` the deviation bound of trajectory `i` (`simplification_bound`) under continuous Fréchet:

- **Original query `Q`:** `|d(Q, i′) − d(Q, i)| ≤ δ_i`. The error depends on one trajectory, so the best strategy is per-trajectory fidelity where it counts, priced against the budget.
- **Compressed query `Q′`:** `|d(Q′, i′) − d(Q, i)| ≤ δ_Q + δ_i`. Two deviations enter one answer and can cancel: two near-duplicate routes simplified alike still match. This is where coordination can pay.
- **Held-out kNN couples only near ties:** `i` pushes `j` out of a top k only when its deviation exceeds the gap between their distances. The margin term already prices that per trajectory.

### What the loop and trades add (`BUDGET_CHOICE → BUDGET_LOOP_TRADE`, 5 seeds)

| Budget | `J4` | `J4-self` | Held-out kNN | Self kNN | Range | Self join |
|---|---:|---:|---:|---:|---:|---:|
| **`0.25%`** | `+0.0086` | `+0.0147` | `+0.054` | `+0.090` | `-0.021` | `-0.017` |
| **`0.5%`** | `+0.0046` | `+0.0127` | `+0.037` | `+0.074` | `-0.029` | `-0.000` |
| **`1%`** | `+0.0015` | `+0.0067` | `+0.014` | `+0.035` | `-0.014` | `+0.005` |
| **`2%`** | `+0.0000` | `+0.0026` | `+0.003` | `+0.013` | `-0.003` | `+0.001` |

- The held-out gain is all kNN, paid for with Range, and it fades with budget: a trajectory can move another's answers by at most its deviation.
- `J4-self` gains more at every budget, but through self kNN, whose queries and reference neighbours are the teacher's own training kNN queries. The self join, where both sides are compressed, does not gain. The teacher does not train on the self join, so this says nothing either way about coordination there.

### Strict coordination: reacting to the others' choices

The control `BUDGET_LOOP_NO_UPDATE` (removed 2026-11-22) makes one pass against the start state and never reacts to the others' moves. It separates the loop's exact rescoring from reacting. Earlier runs, under earlier metrics:

| Run | `BUDGET_CHOICE` | Loop, no reacting | Full loop |
|---|---:|---:|---:|
| `Teacher baseline 2026-11-21`, seed 42, `0.25%`, `J4` | `0.5940` | `0.5969` | `0.5980` |
| `Held-out teacher and kNN proxy` (2026-11-18), 3 seeds, `0.25%`, `J4` gain over `RDP_a0` | `+0.1296` | `+0.1326` | `+0.1336` |
| Same run, `0.5%` | `+0.1129` | `+0.1149` | `+0.1159` |

Reacting was worth about `+0.001` `J4`. Most of the loop's gain is its exact kNN ranking replacing the curves' per-trajectory kNN check.

### Value curves against original partners (seeds 7 and 99)

The value curves score each trajectory against RDP at equal budgets. `Teacher original-partner curves 2026-11-24` scored them against the original trajectories instead (the patch was not kept in the repo):

- `J_T` is identical at every step. Held-out `J4` is lower by `0.012` (`0.25%`) and `0.005` (`1%`) after recipe choice, and by `0.004` and `0.003` for the final teacher, about 3 % of its gain. The loop and trades do about twice the work (`+0.017` against `+0.009` at `0.25%`).
- The change can only move Range: the curves' kNN check ignores competitors either way, and a held-out join outcome depends on the compressed trajectory alone. Against originals every box looks fully answered, so Range is priced low (`-0.088` Range after recipe choice at `0.25%`). That is the level of the Range price, not coordination between particular trajectories. For the final teacher the curve partners barely matter.

### How the coordination effect shrank

- **Discrete Fréchet (before 2026-11-06):** a match depended on which vertices both sides kept. Mixed RDP/SED self-join pairs matched `0.25` against `0.55` for RDP/RDP (150 m, `0.25%`), and switching route groups together paid.
- **Continuous Fréchet:** mixed pairs matched `0.745` against `0.79` (SED/SED) and `0.92` (RDP/RDP).
- **Held-out headline (2026-11-15):** the query side became an original, which removed most of what was left.

### For the thesis

- **A structural result:** with original queries and continuous Fréchet, QDTS is nearly separable once points are priced against one global budget. Interaction is confined to kNN near-ties and to in-database queries.
- **Constructive towards prior work:** RL4QDTS's collective simplification includes the global budget, and that is where the value is here. MLSimp scores points within each trajectory and samples against one global budget, which is the structure these results support. Both papers evaluate with vertex-based measures (EDR, synchronised Euclidean), where coordination plausibly matters more; check this against the papers before writing it.
- **To show coordination where it should exist:** restore the self-join training term and `BUDGET_LOOP_NO_UPDATE` (both removed 2026-11-22), and score the self join and clustering. The 2026-11-06 notes suggest a real but modest effect under continuous Fréchet.

---

## 6. Why the Teacher Is Needed, and the Student

The teacher's role does not depend on how simple the student turns out to be:

- **Label source:** a student needs per-trajectory budget and recipe targets, and nobody knows them without solving the problem. MLSimp says it "lack[s] ground truth optimal simplified trajectories" and uses self-supervision. The teacher produces those labels against exact query trackers, as the strong-branching expert does in Gasse et al. 2019 and the lookahead cut selector in Paulus et al. 2022.
- **What to learn:** allocation plus recipe choice per trajectory carries 93–100 % of the gain, so a per-trajectory student is enough. Without the teacher there would be no way to know a point-level model is unnecessary.
- **Yardstick:** report the student as the share of the teacher's gain it recovers, `R = (J_student − J_base) / (J_teacher − J_base)`, with speed-up and keep-set agreement.
- **A simple student is not a "simple baseline wins" result:** it is a learned model on oracle labels, which is new for QDTS. RDP and Uniform still lose by about `0.1` `J4`.

Student designs, in order:

1. **Budget and recipe predictor** (e.g. LightGBM): budget regression plus a 6-way recipe classifier, fed to the existing allocator. Budgets alone (`BUDGET_RDP`) are 44–64 % of the gain; recipe choice is most of the rest. The teacher's budgets depend on the rest of the database (box density, join partners, kNN neighbours), so the features must capture that; this is untested.
2. **Value-curve predictor trained through the allocator** (decision-focused), with held-out-style pseudo queries in the loss. Plain imitation copies the teacher's join overfitting (Section 4); this design is how a student could beat the teacher on held-out join.
3. **Workload conditioning** for skewed or shifted workloads.

**Go/no-go for design 1:** `budgets.csv` holds every trajectory's teacher budget for all 5 seeds and 4 budgets. Train on it, predict budgets for held-out trajectories, keep RDP points, and score `J4` against `BUDGET_RDP`. Split the trajectories by GeoLife user, not by seed: each seed's database shares most of its trajectories with the others, so a seed split leaks. `budgets.csv` does not record the chosen recipe, so the recipe half needs a small export change. At `0.25%`, keep a cheap loop pass after the student or learn its kNN correction: that is where loop and trades add `+0.054` held-out kNN.

---

## 7. Runtime Summary (24 CPU Cores, Release Build)

| Phase | 5-Seed Mean ± Std | Notes |
|---|---:|---|
| `teacher_0.0025` | `100.8 ± 6.2 s` | All 4 teacher arms (`RDP` + `CHOICE` + `LOOP` + `TRADE`) |
| `teacher_0.005` | `132.1 ± 5.4 s` | All 4 teacher arms |
| `teacher_0.01` | `157.1 ± 7.2 s` | All 4 teacher arms |
| `teacher_0.02` | `218.6 ± 8.0 s` | All 4 teacher arms |
| `test_workloads` | `62.9 ± 8.7 s` | Held-out kNN, 20k Range, 2.15M Where-At, 7 Join thresholds, Clustering |
| `teacher_setup` *(one-time cache build)* | `366 s` (`D2` ranked rebuild) / `701 s` (cold) | Cached on disk in `cache/` (`~14 s` on subsequent runs) |

---

## 8. Key Findings & Team Discussion Points

1. **Why we tightened the metrics (`573c957`, now frozen):** Under the old config (`2 km` Range, `500 m` Where-At, `[100..300] m` Join), endpoints-only Range already scored `0.62`, Where-At gave `53%` credit at `250 m` error, and `2%` baselines saturated at `0.91`. Tightening to `1 km` Range (`256k` train boxes), `25–200 m` Where-At (`@50 m`), and `[50, 100, 150, 200] m` Join (unseen `[75, 125, 175] m`) lowered floors (`Range 0.50`, `Where-At 0.15`), gave smooth baseline headroom across all 4 budgets (`0.39 → 0.57 → 0.72 → 0.85`), and kept `2 km` in Section 4's robustness table for RL4QDTS/MLSimp continuity. Teacher weights remain strictly equal `[1,1,1,1,1,1]` (zero tuning). The change was made after earlier results had been seen, so keep this rationale with the results and report the old configuration as a robustness table. The shape and margin terms, and the shape ramp (`25–500 m`), were designed and checked on the dev seed and pseudo splits under the old metrics and were not re-checked after the change. The `2 km` check in Section 4 is one seed and Range only.
2. **Align everyone on one dataset & config:** Confirm teammates are using the post-Nov-5 `geolife.parquet` (`19,800` trajectories, `19,258,933` points, `60 min` timegap) and the frozen `geolife.json`. Earlier teammate table had `Range` floor `0.796` vs. `0.496` here (likely pre-Nov-5 `5 min` parquet or old `kNN τ` config; not checked).
3. **Running `MLSimp` fairly on these exact 5 seeds (`7, 99, 123, 2024, 777`):** `MLSimp`'s neural model (`train.py` / `score_points.py`) does not use range boxes, so **no retraining is needed** — just re-export `range_hits.bin` under the `1 km` `geolife.json` and re-run `select_points.py`. Evaluate `MLSimp` with **both** `100` boxes (`range_hits.bin`, paper default) and all `256k` boxes (`range_hits_all.bin`, equalised workload knowledge) so the comparison is completely transparent. This workflow has not been checked against the MLSimp code; confirm it in `train.py` / `select_points.py` before relying on it.
4. **Framing `BUDGET_RDP` vs. `BUDGET_LOOP_TRADE` in the thesis:** Always report all 4 per-term columns and show both arms side by side (never call `BUDGET_LOOP_TRADE` a universal "ceiling"). `BUDGET_RDP` beats every non-teacher baseline on **held-out Join, unseen Join, and untrained Clustering** at all 4 budgets, and all but `PARTNER` on **held-out kNN**; `BUDGET_LOOP_TRADE` adds `+0.046` to `+0.058` more `J4` via a multi-objective Pareto move (`+0.13` to `+0.21` on Range/Where-At for `-0.01` kNN / `-0.04` Join at `1%–2%`), while Loop + Trades rebalances `BUDGET_CHOICE` by recovering `+0.037` to `+0.054` held-out kNN at `0.25%–0.5%`.
5. **Robustness checks already done:** Section 4 shows `BUDGET_LOOP_TRADE` wins `J4` in **15/15** `(budget × Range shift)` cells on seed 7 (`seg1km`, `500 m`, `2 km`, and `1 km` 2D Gaussian centres), though under `seg1km` its Range term alone trails the RDP baselines at `0.25%` and `1%`, and it wins under raw mean, geometric mean, and harmonic mean. Earlier `p1001` query-free controls (`Teacher post-fix pseudo check 2026-11-22`, pseudo split, earlier metrics) also showed the Teacher beating query-free `Shape+WhereAt` concave allocation.
6. **Clean path to the Student:** Because allocation plus per-trajectory recipe choice (`BUDGET_CHOICE`) already captures **93–100%** of the Teacher's gain, the right first Student is a lightweight **per-trajectory model** predicting value curves (or budgets) and the recipe from geometry and workload-density features, plugged straight into the existing C++ concave-envelope allocator (exact budget guarantee). Keep all six recipes: the greedy ones carry much of the recipe-choice gain, and the join gap is a budget effect rather than a recipe effect (Section 4). Designs and a go/no-go test are in Section 6.
7. **Small repo housekeeping left:** Update stale numbers/thresholds in `TEACHER_NEXT_STEPS.md` and `engine/src/teacher/README.md` so docs match `geolife.json`. Save the code behind the Range-shift check (Section 4); only its outputs were kept. Restore `BUDGET_LOOP_NO_UPDATE` from git history if a claim about the loop needs it (Section 5).
8. **Open gaps for a paper:** MLSimp and RL4QDTS run under this protocol, Bottom-Up (E/W) baselines, a second dataset (T-Drive), the Range-shift check on all 5 seeds, and the student.

---

## 9. Reproduction Command

```bash
./build-release/engine/teacher_eval \
  --config engine/configs/geolife.json \
  --cache cache \
  --seeds 7,99,123,2024,777 \
  --shares 0.0025,0.005,0.01,0.02 \
  --arms BUDGET_RDP,BUDGET_CHOICE,BUDGET_LOOP,BUDGET_LOOP_TRADE \
  --out "/home/krist/code/qdts-research/research_notes/Teacher final 2026-11-24"
```

That is the directory this run wrote to, at commit `ba16c22`. Use a new `--out` directory for any rerun: the CSVs are appended to.
