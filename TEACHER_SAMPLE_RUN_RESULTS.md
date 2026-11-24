# Teacher sample run: results

Run on 2026-11-20 with the code in `main` at `792231a` (PR #56 merged, including range F1 in J_T
and the `RDP_GLOBAL` baseline).

## Setup

- **Data:** a random 15 % of Geolife trajectories (sample seed 7): 2,970 of 19,800 trajectories
  and 2.94M of 19.3M points. After the 15 % query split, the database has 2,524 trajectories.
- **Config:** `engine/configs/geolife.json`, unchanged apart from the data path: held-out join,
  continuous Fréchet, ranking kNN, teacher trained on joins at 150 m plus 100, 200 and 300 m.
- **Seed:** 42, one seed only.
- **Runs:**
  - `teacher_eval` with all teacher arms and baselines, at budget shares 0.25 % and 1 %.
  - `run` with every configured baseline (`algorithms` × `allocation_alphas`), at 0.25 %, 0.5 %,
    1 % and 2 %.
- The two runners score the same baseline identically (`RDP_a0`: 0.521 J4 in both), so their
  numbers can be compared directly.

One seed on a 15 % sample: differences under about 0.01 J4 are not reliable.

## Teacher and baselines (`teacher_eval`)

Normalised scores on held-out queries. J4 is the mean of kNN, range, where-at and join.

### 0.25 %

| Arm | J4 | kNN | Range | Where-at | Join | J_T | Self join | J4-self |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BUDGET_LOOP | **0.685** | 0.755 | 0.910 | 0.456 | 0.618 | 0.773 | 0.801 | 0.698 |
| BUDGET_LOOP_NO_UPDATE | 0.682 | 0.750 | 0.910 | 0.455 | 0.612 | 0.769 | 0.800 | 0.695 |
| BUDGET_LOOP_REALLOCATE | 0.681 | 0.752 | 0.904 | 0.461 | 0.606 | 0.773 | 0.800 | 0.699 |
| BUDGET_CHOICE | 0.675 | 0.714 | 0.919 | 0.464 | 0.602 | 0.754 | 0.706 | 0.669 |
| BUDGET_LOOP_SHUFFLED | 0.643 | 0.673 | 0.920 | 0.472 | 0.508 | 0.692 | 0.407 | 0.587 |
| BUDGET_RDP | 0.632 | 0.769 | 0.685 | 0.366 | 0.709 | 0.682 | 0.727 | 0.621 |
| EQUAL_LOOP | 0.626 | 0.758 | 0.834 | 0.470 | 0.441 | 0.667 | 0.497 | 0.624 |
| EQUAL_CHOICE_RDP_PARTNERS | 0.615 | 0.703 | 0.842 | 0.475 | 0.438 | 0.640 | 0.458 | 0.591 |
| EQUAL_CHOICE | 0.612 | 0.705 | 0.832 | 0.477 | 0.434 | 0.640 | 0.467 | 0.594 |
| RECIPE_GREEDY_RDP2 | 0.592 | 0.717 | 0.808 | 0.464 | 0.377 | 0.593 | 0.276 | 0.561 |
| RECIPE_GREEDY_SED2 | 0.569 | 0.645 | 0.802 | 0.483 | 0.348 | 0.576 | 0.232 | 0.524 |
| RECIPE_GREEDY | 0.548 | 0.528 | 0.858 | 0.494 | 0.313 | 0.557 | 0.236 | 0.495 |
| PARTNER_l0.5 | 0.526 | 0.797 | 0.460 | 0.332 | 0.516 | 0.559 | 0.663 | 0.544 |
| PARTNER_l1 | 0.525 | 0.794 | 0.451 | 0.326 | 0.530 | 0.559 | 0.639 | 0.537 |
| RDP_a0 (= RECIPE_RDP) | 0.521 | 0.786 | 0.484 | 0.340 | 0.475 | 0.528 | 0.579 | 0.539 |
| RDP_a025 | 0.490 | 0.749 | 0.515 | 0.321 | 0.375 | 0.504 | 0.487 | 0.516 |
| SED_a0 (= RECIPE_SED) | 0.488 | 0.699 | 0.464 | 0.432 | 0.355 | 0.490 | 0.265 | 0.455 |
| UNI_a0 (= RECIPE_UNIFORM) | 0.407 | 0.518 | 0.484 | 0.382 | 0.245 | 0.418 | 0.165 | 0.374 |
| RDP_GLOBAL | 0.359 | 0.652 | 0.499 | 0.215 | 0.072 | 0.341 | 0.264 | 0.411 |

### 1 %

| Arm | J4 | kNN | Range | Where-at | Join | J_T | Self join | J4-self |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BUDGET_LOOP | **0.913** | 0.902 | 0.987 | 0.879 | 0.882 | 0.956 | 0.968 | 0.936 |
| BUDGET_LOOP_NO_UPDATE | 0.913 | 0.904 | 0.987 | 0.879 | 0.882 | 0.954 | 0.959 | 0.934 |
| BUDGET_LOOP_REALLOCATE | 0.911 | 0.902 | 0.987 | 0.880 | 0.876 | 0.957 | 0.966 | 0.936 |
| BUDGET_CHOICE | 0.909 | 0.891 | 0.988 | 0.884 | 0.874 | 0.944 | 0.920 | 0.917 |
| EQUAL_LOOP | 0.891 | 0.922 | 0.967 | 0.830 | 0.847 | 0.925 | 0.889 | 0.899 |
| EQUAL_CHOICE_RDP_PARTNERS | 0.890 | 0.906 | 0.970 | 0.831 | 0.851 | 0.913 | 0.865 | 0.885 |
| EQUAL_CHOICE | 0.890 | 0.907 | 0.970 | 0.831 | 0.851 | 0.913 | 0.859 | 0.884 |
| RECIPE_GREEDY_RDP2 | 0.885 | 0.927 | 0.971 | 0.825 | 0.817 | 0.886 | 0.767 | 0.869 |
| BUDGET_LOOP_SHUFFLED | 0.881 | 0.854 | 0.988 | 0.881 | 0.802 | 0.899 | 0.675 | 0.845 |
| BUDGET_RDP | 0.876 | 0.930 | 0.903 | 0.754 | 0.918 | 0.894 | 0.940 | 0.881 |
| RECIPE_GREEDY_SED2 | 0.861 | 0.874 | 0.970 | 0.839 | 0.760 | 0.861 | 0.658 | 0.836 |
| RDP_a025 | 0.846 | 0.943 | 0.790 | 0.713 | 0.939 | 0.837 | 0.891 | 0.829 |
| RDP_a0 (= RECIPE_RDP) | 0.836 | 0.945 | 0.758 | 0.698 | 0.943 | 0.830 | 0.898 | 0.820 |
| PARTNER_l0.5 | 0.832 | 0.946 | 0.749 | 0.686 | 0.948 | 0.830 | 0.921 | 0.820 |
| PARTNER_l1 | 0.825 | 0.949 | 0.741 | 0.669 | 0.940 | 0.825 | 0.894 | 0.812 |
| RECIPE_GREEDY | 0.824 | 0.824 | 0.978 | 0.819 | 0.675 | 0.825 | 0.522 | 0.773 |
| SED_a0 (= RECIPE_SED) | 0.823 | 0.894 | 0.807 | 0.809 | 0.781 | 0.812 | 0.720 | 0.804 |
| RDP_GLOBAL | 0.794 | 0.899 | 0.774 | 0.674 | 0.830 | 0.788 | 0.780 | 0.767 |
| UNI_a0 (= RECIPE_UNIFORM) | 0.689 | 0.721 | 0.790 | 0.730 | 0.514 | 0.699 | 0.397 | 0.644 |

## Training against test

Each term on the teacher's own training workloads, then on the held-out test queries.

| Share | Arm | kNN | Join | Range | Where-at |
|---|---|---|---|---|---|
| 0.25 % | BUDGET_LOOP | 0.821 → **0.755** | 0.900 → **0.618** | 0.914 → 0.910 | 0.456 → 0.456 |
| 0.25 % | BUDGET_CHOICE | 0.759 → 0.714 | 0.872 → 0.602 | 0.922 → 0.919 | 0.464 → 0.464 |
| 0.25 % | BUDGET_RDP | 0.800 → 0.769 | 0.888 → 0.709 | 0.676 → 0.685 | 0.366 → 0.366 |
| 0.25 % | RDP_a0 | 0.778 → 0.786 | 0.516 → 0.475 | 0.478 → 0.484 | 0.340 → 0.340 |
| 0.25 % | PARTNER_l0.5 | 0.799 → 0.797 | 0.651 → 0.516 | 0.454 → 0.460 | 0.332 → 0.332 |
| 1 % | BUDGET_LOOP | 0.964 → **0.902** | 0.989 → **0.882** | 0.991 → 0.987 | 0.882 → 0.879 |
| 1 % | BUDGET_CHOICE | 0.931 → 0.891 | 0.965 → 0.874 | 0.992 → 0.988 | 0.886 → 0.884 |
| 1 % | BUDGET_RDP | 0.947 → 0.930 | 0.975 → 0.918 | 0.900 → 0.903 | 0.754 → 0.754 |
| 1 % | RDP_a0 | 0.941 → 0.945 | 0.932 → 0.943 | 0.749 → 0.758 | 0.698 → 0.698 |
| 1 % | RDP_a025 | 0.937 → 0.943 | 0.917 → 0.939 | 0.783 → 0.790 | 0.713 → 0.713 |

## Baselines at every allocation (`run`)

J4 for every configured baseline. α sets each trajectory's budget in proportion to its length
raised to the power α: α = 1 is proportional to length, α = 0 gives every trajectory the same
budget. ✓ marks the baselines `teacher_eval` builds.

| Baseline | In `teacher_eval` | 0.25 % | 0.5 % | 1 % | 2 % |
|---|:---:|---:|---:|---:|---:|
| RDP α=0 | ✓ | **0.521** | 0.697 | 0.836 | 0.911 |
| RDP α=0.25 | ✓ | 0.490 | **0.698** | **0.846** | 0.917 |
| RDP α=1 | | 0.242 | 0.421 | 0.652 | 0.835 |
| SED α=0 | ✓ | 0.488 | 0.673 | 0.823 | 0.930 |
| SED α=0.25 | | 0.453 | 0.670 | 0.831 | **0.936** |
| SED α=1 | | 0.241 | 0.424 | 0.635 | 0.824 |
| UNIFORM α=0 | ✓ | 0.407 | 0.544 | 0.689 | 0.810 |
| UNIFORM α=0.25 | | 0.396 | 0.538 | 0.686 | 0.808 |
| UNIFORM α=1 | | 0.221 | 0.375 | 0.545 | 0.692 |
| RDP_GLOBAL | ✓ | 0.359 | 0.553 | 0.794 | 0.898 |

## Findings

1. **The teacher leads the baselines, and the lead comes from range and where-at.**
   - `BUDGET_LOOP` beats the best cheap baseline by +0.16 J4 at 0.25 % (0.685 against
     `PARTNER_l0.5`, 0.526) and by +0.07 at 1 % (0.913 against `RDP_a025`, 0.846).
   - Range alone is 0.91 against RDP's 0.48 at 0.25 %.
2. **The teacher's kNN and join gains do not carry over to new queries.**
   - Range and where-at score the same on training and test workloads.
   - The teacher's kNN drops 0.06 from training to test, and its join drops 0.11 to 0.28.
     The baselines' kNN and join barely change.
   - On held-out queries `BUDGET_LOOP` is below plain RDP on both: kNN 0.755 against 0.786 at
     0.25 %, and join 0.882 against 0.943 at 1 %.
   - It fits the particular neighbours and pairs in its training workloads, at the expense of
     what new queries need. This is a generalisation problem, not a code bug.
3. **Budget search is the step that carries over to the test.**
   - `BUDGET_RDP` beats `RDP_a0` by +0.11 J4 at 0.25 % and +0.04 at 1 %.
   - Recipe choice adds +0.04 at 0.25 % (`BUDGET_CHOICE` against `BUDGET_RDP`) and +0.03 at 1 %.
4. **The loop adds little J4.**
   - `BUDGET_LOOP` against `BUDGET_CHOICE`: +0.010 at 0.25 %, +0.004 at 1 %.
   - Reacting to other trajectories' choices, `BUDGET_LOOP` against `BUDGET_LOOP_NO_UPDATE`:
     +0.003 at 0.25 %, 0.000 at 1 %.
   - The loop's J_T gains sit mostly in kNN and join, the terms that do not carry over.
5. **Reallocation is not worth it.**
   - `BUDGET_LOOP_REALLOCATE` has the highest J_T but lower J4 than `BUDGET_LOOP` at both shares
     (0.681 against 0.685, 0.911 against 0.913).
   - It took 427 s at 0.25 % and most of the teacher's 1,962 s at 1 %.
6. **The shuffled control behaves as expected.** Wrong partners cost 0.04 J4 at 0.25 % and halve
   the self join (0.407 against 0.801).
7. **The loop never ended below its start.** `BUDGET_LOOP`'s J_T is above `BUDGET_CHOICE`'s at
   both shares (0.773 against 0.754, 0.956 against 0.944), so the watch-list gap in
   `TEACHER_CORRECTNESS_ISSUES.md` (item 1) did not show up here.
8. **Baselines `teacher_eval` does not build.**
   - α = 1 is the worst allocation for every algorithm at every share.
   - SED α = 0.25 is the best baseline at 2 % (0.936 against `SED_a0`, 0.930).
   - Otherwise the best baseline at each share is one `teacher_eval` already builds.

## Run time

`teacher_eval` took 51 minutes in total on 4 cores, using 1.3 GB of memory at peak:

| Step | Seconds |
|---|---:|
| Teacher setup (kNN cache) | 148 |
| Teacher at 0.25 % | 759 |
| Teacher at 1 % | 1,962 |
| Test workloads | 20 |
| Evaluating 44 arms | about 175 |

Evaluating an arm takes about 4 s on this sample, so 5 more baselines cost about 20 s per share
and seed, under 1 % of a run. `run` scored all 40 baselines at 4 shares in under 5 minutes.
