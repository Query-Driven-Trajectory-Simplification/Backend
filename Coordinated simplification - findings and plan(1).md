# Coordinated simplification for relational trajectory queries: findings and plan

*Status document for the group, 2026-11-05. It summarises the collective-search line of work: what we tried, what is now established, what is not, and what we propose next. Every number comes from a linked report; nothing here is new.*

---

## TL;DR

- **The question.** Query-driven trajectory simplification (QDTS) decides which points to keep under a global point budget. Current methods, including MLSimp, RL4QDTS and our own BKH, make that decision **one trajectory at a time**.
- **The observation.** Some queries are **relational**: the **self-join** ("which stored trajectories follow the same route?") and **route clustering**. Their answers depend on *pairs and groups* of simplified trajectories. Under these queries, representation choices interact. A join pair survives far more often when both sides are simplified the same way (DP on both: 85% match rate) than when they are mixed (58%). Mixed is worse than both-BKH (65%).
- **What works.** Re-simplifying whole **route groups** (clusters of trajectories on the same route, taken from the join graph) at once:
  - improves the held-out four-query score by **+0.055 at 0.25%** and **+0.030 at 0.5%**;
  - about ten times what point-level search achieved;
  - takes about 11 s and 252 evaluations.
  - Random groups of the same sizes achieve **nothing**. Followed by single-trajectory polishing, it is the best method we have.
- **What it is not (yet).**
  - Strong individual search catches up, at about 50× the cost, so the advantage is speed and a better starting point, not unreachable quality.
  - With the current BKH/DP menu the gain is mostly a **trade**: join improves, while range, where-at and kNN lose a little.
- **What we propose.** A **partner-aware learned simplifier**: MLSimp-style encoders with cross-trajectory attention and learned allocation and selection. It is trained from the coordinated search and makes coordinated choices in one pass. A first non-learned test shows coordination helps at point level (+0.040 over an identical independent control), but the hand-built rule is too blunt to beat BKH. That gap is what the model would learn.

---

## 1. Setup in one page

| | |
|---|---|
| Data | Geolife, **no-gap cut** (14,902 trajectories, not split at time gaps; matches MLSimp/RL4QDTS preprocessing). Test database: 12,666 trajectories, 16.5M points. Train, validation and query-pool splits are disjoint. |
| Budgets | 0.25%, 0.5% (also 1%, 2% in older work) of interior points, plus mandatory endpoints. At 0.25% that is about **3 interior points per trajectory**. |
| Queries | **Range** (2 km × 7 day boxes, F1); **where-at** (position at time t within 100 m); **kNN** (recall@10, discrete Fréchet, original held-out queries); **self-join** (all database pairs with discrete Fréchet ≤ 500 m, both sides simplified, micro F1). |
| Score | Each metric is normalised to its endpoints-only floor, (x − floor)/(1 − floor). **J4** = equal mean of the four. **J5** adds route clustering. |
| Workloads | Search uses a *search* workload, decisions use *validation*, and all reported numbers are on a held-out *test* workload. Self-join and route clustering are fixed global tasks, the same in search and test. |
| Starting point | **BKH**: our per-trajectory query-aware selector. It picks among SED, UNIFORM, RDP and greedy variants per trajectory using synthetic workloads, and never scores the join. Equal points per trajectory at 0.25% and 0.5% (allocation α = 0). |
| Seeds | 17/29/83. They vary workloads and search randomness and share database membership, so they are not independent datasets. |

### Relational vs per-trajectory queries
- **Per-trajectory:** range, where-at, and kNN against *original* queries. A trajectory's simplification affects the answer largely on its own (kNN is weakly coupled through competition between candidates).
- **Relational:** the self-join and route clustering. Both operands are simplified, so the answer depends on *how compatibly* related trajectories are simplified.

### What "route groups" are
On the original data we take the self-join pairs: start and end points within 500 m, discrete Fréchet ≤ 500 m. We then group greedily around medoids: a medoid plus all its unassigned partners, with at least 5 members.
- On Geolife that gives **252 groups covering 4,740 trajectories (37%)**, mostly repeated routes such as commutes.
- Members must be close to the medoid, so groups **cannot chain** into one giant cluster.
- **92% of all true join pairs lie inside a route group.**
- The route-clustering metric reruns the same grouping on the simplified data and scores agreement (symmetric macro best-match F1).

---

## 2. What we tried first, and why it did not work

These negative results matter. They show the controls are strict, and they explain why the positive result needed the right units.

| Attempt | Result | Report |
|---|---|---|
| Point-level collective search (single swaps, random group restorations; cheap or random ordering; LightGBM LambdaRank move ranker) | About +0.005 over BKH at 0.25%, and almost nothing at 1–2%. The ranker was the worst ordering (negative transfer from small training databases). Every declared gate failed. | Four query collective overnight gate |
| Current-error witness swaps (targeted repairs of failing join pairs) | Collective only +0.0008 over a matched individual control. | Current-error witness swaps |
| TRACLUS clustering as a 5th metric | No stable, informative setting on Geolife: one giant cluster at ε ≥ 25 m, chaotic below about 7 m, near-zero scores under compression. The MLSimp-style query-scoped variant is 96–99% dominated by one cluster. | Global clustering robustness gate; Clustering configuration and budget curves |
| Route groups with point-level witness repairs | The flat spots are real, but neutral-step individual search recovers 87% of the gain. The grouping-specific edge is +0.0017. | Route group repair and neutral controls |
| Small groups (4 trajectories) with anchor-synchronized menus | 0/24 groups needed a coordinated move. Diagnosis: a median of 6 internal vs about 412 external join relationships per group, so most interactions sat outside the group. | Small-group representation gate |

**Lesson:** coordination only paid off once (a) the move changed a trajectory's *whole representation* and (b) the group *enclosed most of its relationships*.

---

## 3. What is now established

### 3.1 The mechanism: pair compatibility
Match rate of true join pairs by simplifier on each side (mean of 3 seeds):

| Budget | BKH/BKH | DP/BKH | BKH/DP | DP/DP |
|---|---:|---:|---:|---:|
| 0.25% | 0.648 | 0.574 | 0.591 | **0.846** |
| 0.5% | 0.735 | 0.631 | 0.650 | **0.804** |

- **Mixed pairs are worse than both pure configurations.** A trajectory switched alone hurts its own matches; switching its partners with it pays.
- **DP's join advantage is mostly correspondence, not better individual shape.** Of the pairs DP/DP rescues, **66%** (0.25%) and **89%** (0.5%) need *both* sides to be DP.
- **False positives do not rise under DP/DP.**

*Caveat: measured with discrete Fréchet (kept vertices only). Continuous or resampled Fréchet is untested.*

### 3.2 Collective route-group switching
For each route group (all 252, random order, each visited once): re-simplify every member with DP at its own quota, and keep the switch if the full four-query score improves. The objective contains **no clustering term**.

| Test J4 | 0.25% | 0.5% |
|---|---:|---:|
| BKH | 0.497 | 0.657 |
| DP everywhere | 0.467 | 0.563 |
| Best of BKH or DP for the whole database | 0.497 (BKH) | 0.657 (BKH) |
| All route groups → DP (no search) | 0.538 | 0.642 |
| **Collective switching** (252 evaluations, ~11 s) | **0.552** | **0.686** |
| Collective + single-trajectory polish (~7 min) | **0.561** | **0.689** |

- **Order sensitivity:** five random orders land within 0.001 of each other.
- **Route clustering also improves without being optimised:** 0.515 → 0.652 at 0.25%. It reaches 0.756 when included in the objective.

### 3.3 Controls (does it survive scrutiny?)
| Control | Result |
|---|---|
| **Random groups**, same sizes, same 4,740 trajectories | 0.497 at 0.25% (= BKH); −0.055 vs route groups. **Structure is essential.** |
| Random groups drawn from the whole database | 0.496 at 0.25%. |
| **Reversible multi-pass individual switching** (up to 3 passes) | 0.556 (+0.004 vs collective) at 0.25%; 0.686 (−0.001) at 0.5%. It needs **~13,600 evaluations / 10–11 min** against 252 / 11 s. |
| Collective + polish vs multi-pass individual | +0.005 (0.25%), +0.004 (0.5%), with fewer total evaluations. |
| One-pass individual switching, strict or with neutral steps | Collective beats it by +0.012 to +0.017 in all 6 seed/budget cells. |

**Honest reading:** collective switching is a fast, structure-aware optimiser. It gets about 98% of the strong individual result with about 2% of the exact evaluations, and it gives the best starting point for polishing. It does *not* reach states individual search cannot.

### 3.4 It is a trade-off (with the current menu)
Changes at 0.25%, normalised: join **+0.28**, where-at −0.032, range −0.021, kNN −0.007. Limiting each classical metric to lose at most 0.01 normalised (a guardrail):

| | 0.25% | 0.5% |
|---|---:|---:|
| Unguarded collective | +0.055 | +0.030 |
| Guarded collective (δ = 0.01) | +0.020 (37%) | +0.017 (57%) |
| Guarded collective + polish | +0.038 | +0.023 |

- At 0.25%, each 0.01 of where-at given up buys about 0.1 of join.
- The guard set on the search workload holds on the test workload, so the losses are systematic properties of DP, not overfitting.
- *This was a sensitivity analysis, not a pre-registered test.*

### 3.5 Coordination also works at point level (first non-learned test)
At fixed quotas, a few kept points per route member are replaced. **Route-aligned** keeps the points that correspond, in order, to the medoid's kept points. The **independent control** uses the same candidates, the same removal rule and the same trajectories, but chooses by individual importance.

| 0.25% | Test J4 | Join | Where-at |
|---|---:|---:|---:|
| BKH | 0.4973 | 0.765 | 0.482 |
| Independent control | 0.4566 | 0.733 | 0.437 |
| **Route-aligned** | 0.4966 (**+0.040 vs control**) | **0.798** | 0.438 |
| Global spatial anchors | 0.4397 (+0.018 vs its control) | 0.775 | 0.402 |

Match rate of join pairs where both trajectories were changed:

| Changes | Match rate |
|---|---:|
| neither side | 0.663 |
| one side only | 0.655 / 0.649 |
| **both sides** | **0.716** |

- **Coordination signature:** changing both sides helps; changing one side doesn't. In the control, changing both hurts.
- **Not a product yet:** overall it ends level with BKH, because forcing a swap in every member costs where-at.
- **Route alignment beats spatial anchors:** being near the same place does not mean corresponding positions along the route.

### 3.6 Robustness (partial)
On the **gap30 cut** (split at gaps over 30 min; 20,165 test trajectories, 462 route groups; seed 17, 0.25% only), the same pattern holds:
- **Pair compatibility:** mixed 0.68 vs BKH 0.70 vs DP/DP 0.85.
- **Collective switching:** +0.055.
- **Random groups:** +0.002 to +0.008.

The individual and polish comparisons on gap30 were not run.

---

## 4. Proven vs not proven

| Claim | Status |
|---|---|
| Selector choices interact under the self-join (mixed pairs are worse than both pure configurations) | **Established** (3 seeds, 2 budgets; replicated on gap30, 1 seed) |
| Route groups from the join graph are the right units; random groups fail | **Established** |
| Collective switching reaches strong individual quality with about 2% of the evaluations | **Established** (no-gap) |
| Collective + polish is the best method found | **Established** (no-gap, BKH/DP menu) |
| Collective reaches quality individual search cannot | **Not supported:** individual search catches up |
| The join vs classical trade-off is fundamental | **Not established:** observed with the BKH/DP menu only |
| Partner information improves point selection over independent selection | **Established in one hand-built test** |
| A first pass with partner information can beat BKH or DP overall | **Open:** the hand rule did not |
| Any ML component beats the non-learned ladder | **Open:** nothing learned has been tested in this line |
| Results hold beyond Geolife, or for continuous Fréchet | **Open** |

---

## 5. The thesis story (agreed framing)

1. **Strong baselines:** geometric selectors (DP, SED, UNIFORM), allocation rules (size-proportional, size^α), and per-trajectory query-aware selection (BKH; MLSimp and RL4QDTS in the literature).
2. **Measured interactions:** under relational queries, a trajectory's useful representation depends on its partners' (§3.1).
3. **Coordinated refinement:** compared against meaningful vs random groups and against repeated reversible individual search, using the same candidates (§3.2–3.3).
4. **Quality against runtime:** all four metrics, setup costs, and the bounded-loss variant (§3.4).
5. **Learning, if justified:** the partner-aware simplifier (§6), positioned as extending MLSimp's point importance from *within one trajectory* to *across related trajectories*.

Framing notes:
- The precise contrast is **individual vs coordinated moves**. Individual search on the full objective *does* see the join; it just changes one trajectory at a time.
- Collective search is a legitimate contribution on its own (comparable quality, far faster). ML is the extension, not the rescue.
- **Report the join vs classical split openly.** It is the motivation for the next step, not something to hide.

---

## 6. What we want to build: a partner-aware learned simplifier

**Goal:** replace the hand rules (pick DP or BKH, size^α, align to the medoid) with a model that decides **allocation and selection jointly, in one pass, using related trajectories.**

### Architecture sketch
1. **Relational graph** (built once, cost counted): join candidates or route groups, via an endpoint prefilter plus Fréchet on originals. A cheaper approximate graph is an option to study.
2. **Point encoder:** a transformer over each trajectory's points (position, time, speed, heading, local geometry), chunked for long trajectories as in MLSimp.
3. **Cross-trajectory attention (the new part):** each trajectory's points attend to its partners' points. This is the learned version of the route alignment shown to matter in §3.5.
4. **Heads:**
   - a **selection head** giving a keep-score per point;
   - an **allocation head** giving each trajectory's share of the budget, conditioned on relational context. The target is the marginal benefit of extra points, not join degree.
5. **Budget layer:** exact integer quotas summing to the budget (water-filling), then top-q points per trajectory, endpoints always kept.

**Inference:** one forward pass plus top-k. No search, no exact evaluations.

### Training
- **Teacher:** coordinated search (collective + polish, plus point-level alternatives) on *development databases*, producing several good alternatives per group in different partner states and at different budgets. Switching labels alone would only teach "use DP".
- **Fine-tuning** through differentiable query surrogates. Range and where-at exist. A **soft discrete-Fréchet join** surrogate would be new.
- **Evaluation** on separate databases, against the full non-learned ladder.

### Why this is different from earlier ML attempts that failed
The per-trajectory imitation scorer, the surrogate MLP, the gain-curve allocation and the LambdaRank move ranker were all **blind to partners** and targeted classical queries. The new model sees partners, targets the relational queries where we have measured headroom, and has a teacher that found real gains.

### Risks
- **Little relational training data:** about 250 groups per database. It needs several development databases.
- **The soft join surrogate is unproven.**
- **The bar is high:** to count, the model must beat collective switching (+0.055 in 11 s) or match it on new data without search.
- **Short budgets:** with about 3 interior points per trajectory, every decision is costly.

### Variants, from safest to most ambitious
| Variant | What the model decides | Difficulty |
|---|---|---|
| Group-aware selector choice | which simplifier each trajectory or group uses | low |
| Group-aware allocation | point quotas across related trajectories | medium |
| **Partner-aware point selection** (preferred) | the actual points, compatible with partners | high |

Recommended order: fixed-quota point selection first, allocation second, never both untested at once.

---

## 7. Proposed next steps (each with a go/no-go)

1. **Present this angle** to the supervisor and group. Agree on the framing and on which variant to target.
2. **Backend:**
   - add a **self-join mode** (both sides simplified) next to the one-sided join;
   - use **one primary score per query type**: kNN recall@k, range F1, where-at average-hit or within-radius, join F1. Keep the rest as diagnostics.
3. **Incremental group evaluator:** re-score only the join pairs and queries a changed group touches. Needed for richer teacher data and faster search; the speedup is still to be measured.
4. **Smarter non-learned first pass** (cheap): decide *which* members change and *which* point to give up, protecting where-at. Go if it beats BKH and whole-group DP at fixed quotas.
5. **Teacher data:** point-level alternatives per group, evaluated in database context, across development databases and budgets.
6. **Model v1:** partner-aware selection head with fixed quotas. Go if it beats the independent-encoder ablation (same model without cross-attention) and matches collective switching on a separate database with far fewer evaluations.
7. **Model v2:** add the allocation head and the soft join surrogate.
8. **Robustness:** finish gap30 (individual and polish arms, all seeds), then the 5-min cut. Check continuous or resampled Fréchet for the compatibility result.

---

## 8. Things that would be cool, but are not planned yet

- **Ranked kNN or database kNN** (a stored trajectory as the query) as a third relational query. It would test whether the mechanism generalises beyond the join's 500 m threshold.
- **Co-movement or convoy queries** (genuinely group-level) for datasets with many trajectories at the same time, such as Porto.
- **Correlated-sampling theory:** the link to join synopses in approximate query processing, where both sides keep the same keys, could give a principled account of why coordination helps.
- **A certified upper bound** on achievable score, via an LP relaxation over per-trajectory candidate simplifications with pair rewards. It would replace optimistic headroom estimates.

---

## 9. Open questions for the group

1. Which **join semantics** is primary for the thesis: the self-join of the simplified database, or the one-sided query join, or both?
2. Do we keep **route clustering** as a reported metric, given it shares the join's similarity (r ≈ 0.88 with join F1)?
3. Is a **join vs classical trade-off** acceptable as a result, or do we require bounded classical loss, such as the δ = 0.01 guardrail?
4. Which **ML variant** do we commit to: selection first (recommended) or allocation first?
5. Which **datasets and cuts** are in scope: no-gap primary, with gap30 and 5-min as robustness?

---

## Glossary

- **BKH:** our per-trajectory selector gate. It chooses each trajectory's simplifier from SED, UNIFORM, RDP and greedy variants using synthetic workloads; it never scores the join.
- **DP:** Douglas–Peucker (RDP) at a fixed point quota.
- **Collective switching:** accept or reject re-simplifying a whole route group at once.
- **Polish:** single-trajectory BKH↔DP switches after collective switching.
- **Normalised score:** (x − endpoints-only floor) / (1 − floor).
- **J4 / J5:** the four-query score / J4 plus route clustering.

## Report index (`~/code/qdts-research/reports/`)

- Four query collective overnight gate, Current-error witness swaps: point-level collective search.
- Global clustering robustness gate, Clustering configuration and budget curves: TRACLUS.
- Coordination sensitive clustering gate, Route group repair and neutral controls: route groups and plateaus.
- Small-group representation gate: small groups with anchor menus.
- **Collective selector switching**, **Collective switching controls**, **Classical-loss guardrail**: the main positive result and its checks.
- **Coordinated first-pass selection**: coordination at point level.
- Rethinking collective search with ML: earlier literature review and ML directions.
- `research_notes/Gap30 robustness partial/`: the partial gap30 replication.
