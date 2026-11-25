# Proposed section: Evaluation framework

Outline for a new section that defines how every method is evaluated, before any method is
introduced. Each item gives the **motivation** (why the framework needs it) and the **method** (how
our framework does it). **Status** says whether it already exists in the code or still needs work.

**Placement.**

- **Thesis:** its own chapter, after the problem definition and before the theory and method
  chapters.
- **ICDE paper:** §3 becomes "Problem and evaluation framework" (about 1.25 pages). Engine and
  adapter details go to the technical report.

**One rule keeps it from contradicting the experimental setup:** this section holds definitions and
rules, with no parameter values. The setup section holds the values (datasets, box sizes, thresholds,
seeds, hardware) and introduces no new definitions.

---

## 1. Requirements

The principles the framework must satisfy, stated up front so every later choice can be traced to
one of them.

- **R1. Held-out queries**
  - *Motivation:* a simplification fitted to known queries can overfit them. If test queries come
    from the training distribution or from the database itself, as in prior QDTS evaluations, that
    overfitting is invisible.
  - *Method:* no method sees the test queries. Training workloads are built from the database only;
    test workloads are built from a held-out pool, after every method has produced its
    simplification.
- **R2. Coverage of query types**
  - *Motivation:* methods can trade one query type against another (the teacher's recipe step at 1%
    trades join and kNN for range and where-at), so a single query type hides trade-offs.
  - *Method:* four query types with different semantics (similarity, range, temporal, pairwise),
    each reported separately, plus one untrained query (clustering) as a transfer check.
- **R3. Comparable budgets**
  - *Motivation:* prior papers count endpoints differently (RL4QDTS inside the budget, MLSimp on
    top), so "0.25%" means different things across papers.
  - *Method:* one budget definition for every method, and every result reported as the share of all
    points kept (B/P).
- **R4. Equal information access**
  - *Motivation:* the teacher sees sample workloads and error-driven baselines do not. Without an
    explicit statement, every comparison invites the objection that it is unfair.
  - *Method:* a table stating, per method, what it sees (database, training-query types and counts,
    held-out queries: never) and how it is tuned.
- **R5. Normalised aggregation**
  - *Motivation:* raw scores sit on different scales and have different floors. Endpoints-only Range
    already scores about 0.50, Where-at about 0.15, so raw averages over-weight the easy terms.
  - *Method:* every score normalised against its endpoints-only floor, then averaged into one
    combined score, with per-term scores always reported next to it.
- **R6. Reproducibility and a fixed protocol**
  - *Motivation:* the metric tolerances were tightened after earlier results had been seen, so the
    protocol has to be frozen and disclosed to stay credible.
  - *Method:* fixed seeds, a dated metric-freeze commit, tuning only on a development seed or a
    pseudo split, and a public artifact.

---

## 2. Data split and budget

- **Held-out split of trajectories**
  - *Motivation:* query trajectories must not also be database trajectories, or kNN and join become
    self-queries.
  - *Method:* each dataset is split into a database part, which is simplified, and a query pool,
    which is never simplified and never seen during training. Several random splits; state that they
    overlap and are therefore not independent replications.
  - *Status:* exists (`split`, 85/15 in the current config).
- **Budget definition**
  - *Motivation:* every trajectory needs its endpoints to remain a trajectory; R3.
  - *Method:* B = 2T + round(s · (P − 2T)): endpoints reserved, the share s applied to interior
    points. Report B/P.
  - *Status:* exists (`allocation::total_budget`).
- **User-disjoint splits for learned models**
  - *Motivation:* GeoLife users repeat routes, so a random split leaks near-duplicates between the
    student's training and test data and inflates its results.
  - *Method:* split by GeoLife user for every learned model.
  - *Status:* **to do** (planned for the student).

---

## 3. Query suite

One table in the text: query type, what is compared with what, semantics, score.

- **Held-out kNN**
  - *Motivation:* similarity search is the main use of trajectory databases and the query type where
    trajectories compete with each other.
  - *Method:* each original query trajectory from the held-out pool retrieves its top k from the
    simplified database under continuous Fréchet distance; score = recall@k against the answer on
    the original database. Ties are ranked identically in the evaluator and the teacher.
  - *Status:* exists.
- **Range**
  - *Motivation:* the standard QDTS query in all prior papers (Zhang et al., RL4QDTS, MLSimp).
  - *Method:* space-time boxes; macro F1 between the trajectories found on the simplified and the
    original database. Primary semantics: point-in-box, as in prior work, for comparability.
  - *Status:* exists.
- **Where-at**
  - *Motivation:* temporal accuracy: where was the object at time t? Not covered by the other three.
  - *Method:* probes at sampled times on each database trajectory, position interpolated on the
    simplified trajectory, scored by a distance ramp.
  - *Status:* exists.
- **Held-out join**
  - *Motivation:* pairwise similarity (which trajectories travel together) is the query type most
    sensitive to how the budget is split across trajectories.
  - *Method:* original held-out queries against the simplified database, matched when the continuous
    Fréchet distance is within a threshold; macro F1 averaged over several thresholds.
  - *Status:* exists.
- **Clustering (untrained)**
  - *Motivation:* checks whether gains transfer to a task no method optimises.
  - *Method:* connected-component route clusters on the simplified database against those on the
    original; reported, but kept out of the combined score.
  - *Status:* exists.
- **Self-join and self-kNN (diagnostics)**
  - *Motivation:* queries between two simplified trajectories are where coordination between
    trajectories can matter (`QDTS_THEORY.md`, P3).
  - *Method:* reported as a separate combined score (J4-self), never in the headline, because the
    teacher's training workloads are drawn from the database itself.
  - *Status:* exists.

---

## 4. Scoring

- **Floor normalisation**
  - *Motivation:* R5.
  - *Method:* norm(x) = (x − floor) / (1 − floor), floor = the score with endpoints only. 0 means no
    better than endpoints, 1 means perfect, and below 0 is possible.
  - *Status:* exists.
- **Combined score J4**
  - *Motivation:* one number to rank methods, without hiding trade-offs.
  - *Method:* the mean of the four normalised terms. The arithmetic mean is the headline; geometric
    and harmonic means are reported as checks. Equal weights are a stated design choice, not a tuned
    one.
  - *Status:* exists; the weight-simplex check is **to do**.
- **Recovered share R for learned methods**
  - *Motivation:* the teacher's score is a reference, not an upper bound, so a learned method is
    judged by how much of it it reproduces.
  - *Method:* R = (J_method − J_base) / (J_teacher − J_base). R > 1 is allowed.
  - *Status:* definition only; **to do** with the student.
- **Statistics**
  - *Motivation:* overlapping splits understate variance.
  - *Method:* mean ± std over splits, paired differences with bootstrap confidence intervals over
    queries, wins per split.
  - *Status:* means, std and paired t-intervals exist (`teacher_stats.py`); bootstrap over queries
    **to do**.

---

## 5. Training and test separation

- **Separate workloads for training and testing**
  - *Motivation:* R1. This is the main difference from prior QDTS evaluations.
  - *Method:* the teacher's training workloads (kNN lists, join pairs, range boxes, where-at probes)
    are built from the database alone and cached. The test workloads are built from the held-out
    pool only after every method is done. Training and test range boxes come from the same generator
    but are disjoint samples.
  - *Status:* exists (`teacher_eval`).
- **Tuning without the test data**
  - *Motivation:* tuning on the reported splits leaks the test set into the method.
  - *Method:* every design choice is tuned on a development seed (excluded from the reported splits)
    or on a pseudo split, which splits the database again and never reads the query pool.
  - *Status:* exists (`--pseudo-split`).

---

## 6. Integrating external methods

- **Adapters, not modified code**
  - *Motivation:* a reviewer must be able to see that RL4QDTS and MLSimp were run faithfully,
    especially since MLSimp scores below uniform sampling under our protocol.
  - *Method:* the authors' code stays untouched as a submodule; all changes live in an adapter, and
    every change is listed with its reason (`rl4qdts/CHANGES.md`, `mlsimp/CHANGES.md`).
  - *Status:* exists on the `Adding-RL4QDTS` and `mlsimp` branches.
- **Two settings per method**
  - *Motivation:* corrections (e.g. MLSimp's range queries were its test queries) change results;
    showing only the corrected version invites the claim that it was set up to lose.
  - *Method:* report the authors' setting and the corrected one, plus the method's strongest variant
    under our protocol.
  - *Status:* MLSimp seed 7 only; RL4QDTS **to do**.
- **Both protocols**
  - *Motivation:* a method can lose under our protocol only because the protocol differs from its own.
  - *Method:* every method, ours included, also evaluated under the authors' native protocol, with a
    sanity check that our reproduction matches their published numbers.
  - *Status:* **to do**.

---

## 7. Comparison with prior protocols

- **Protocol table**
  - *Motivation:* makes "larger and stricter" a checkable statement, and explains in advance why
    results under our protocol differ from published ones.
  - *Method:* one table, columns RL4QDTS / MLSimp / ours; rows: database size, number of test
    queries, test queries held out or drawn from training, range box size, similarity measure (EDR or
    continuous Fréchet), kNN k, join thresholds, endpoints in or on top of the budget.
  - *Status:* **to do** (facts are in the writing plan and the two papers; verify each cell).

---

## 8. Robustness variants and threats to validity

- **Robustness variants**
  - *Motivation:* the headline should not depend on one setting.
  - *Method:* unseen join thresholds (between the training ones); segment-in-box Range; shifted range
    workloads (box size, spatial distribution); alternative aggregations of J4.
  - *Status:* unseen thresholds and aggregations exist; segment and shift exist for seed 7 only, and
    the code was not saved: **to do** on all splits.
- **Threats to validity, each with the experiment that addresses it**
  - *Point-in-box Range rewards vertex placement, not geometry* (proven in `QDTS_THEORY.md`, S4) →
    segment-in-box variant.
  - *Overlapping splits* → user-disjoint folds and bootstrap intervals.
  - *Metrics fixed after earlier results* → freeze commit, plus the old configuration rerun on the
    same splits.
  - *Equal weights are a choice* → weight-simplex sweep and per-term wins/ties/losses.
  - *Workload knowledge differs between methods* → information-access table and a workload-size
    sweep.
  - *One dataset* → T-Drive (and a third dataset in Phase 2).

---

## What it changes elsewhere

- **Writing plan:** §3 renamed and marked as the framework section; the information-access table
  (T2) moves from §6 here; a new protocol table (T12) in the main text; contribution bullet 5 ("open
  protocol and artifact") points to this section.
- **Introduction draft:** the protocol contribution bullet already matches; only the outline
  sentence changes.
- **Setup section:** keeps only the values this section defers to it.
