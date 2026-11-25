# Theoretical backing for the QDTS principles

Draft. This note goes through the three principles we use to explain the teacher's results
(**slack**, **separability**, **marginality**). For each step it states what can be proven, what
can only be argued, and what rests on the experiments. The goal is a thesis chapter where the
theory explains *why* the results look the way they do, and the experiments measure *how much*.

Numbers quoted are from `TEACHER_RESULTS.md` (branch `origin/results`, 5 seeds, 2026-11-24).
Code references are to `main` at `bb6ea56`.

**Key to the labels used below:**

| Label | Meaning |
|---|---|
| **Proven** | A short, complete proof exists (given here or sketched precisely enough to write out) |
| **Argued** | A sound argument, but it rests on an approximation or an assumption that is not proven |
| **Empirical** | Only the experiments support it; theory at most explains the direction |
| **Not provable** | A design choice or a normative claim: it can be motivated but not proven |

---

## Summary

| # | Claim | Status | Section |
|---|---|---|---|
| L1 | Simplifying a trajectory moves its distance to any query by at most its deviation | **Proven** | 1 |
| L2 | `simplification_bound` is a valid bound on that deviation | **Proven** (by construction) | 1 |
| S1 | A join answer cannot change while δ_i is below the pair's slack | **Proven** | 2 |
| S2 | A kNN top-k set (original queries) only changes near ties | **Proven** | 2 |
| S3 | A segment-in-box Range answer only changes near the box boundary | **Proven** in space; **open** for space-time boxes | 2 |
| S4 | Point-in-box Range does **not** obey slack | **Proven** (counterexample) | 2 |
| P1 | With original queries, each outcome depends on one trajectory only | **Proven** | 3 |
| P2 | J4 is additive across trajectories up to F1 curvature and kNN ties | **Argued** | 3 |
| P3 | With compressed queries, outcomes depend on pairs of trajectories | **Proven** | 3 |
| P4 | The leftover coupling is small in practice | **Empirical** | 3 |
| M1 | Given separability and concave value curves, greedy marginal allocation is optimal | **Proven** (classic) | 4 |
| M2 | Without concavity, the allocator is optimal at envelope vertices, with a bounded gap | **Proven** at vertices; gap bound **to write out** for this allocator | 4 |
| M3 | The value curves are close to concave | **Empirical** (not guaranteed) | 4 |
| M4 | Size-proportional budgets are suboptimal in general | **Proven** | 4 |
| T1 | The loop terminates at a local optimum | **Proven** (when it stops before its pass cap) | 5 |
| T2 | The teacher is a global optimum / an upper bound on any method | **Not provable**, and false as stated | 5 |
| G1 | Fitting each trajectory's budget to its training pairs overfits | **Argued** (optimiser's curse) | 6 |
| G2 | `flip_probability` is the expected flip rate under its model | **Proven** | 6 |
| G3 | That model (uniform error, uniform slack spread) fits the data | **Empirical**, not yet tested | 6 |

Section 7 lists the conclusions that rest on experiments alone; section 8 the ones that are design
choices.

---

## 0. Setting

- Database trajectories T_1, …, T_n. A simplification T_i′ keeps a subsequence of T_i's points,
  always including both endpoints.
- δ_i is any upper bound on the Fréchet distance d_F(T_i, T_i′); the teacher uses
  `simplification_bound`.
- An **original query** Q is an unsimplified trajectory (held-out kNN and held-out join). A
  **compressed query** Q′ is itself simplified (self join, clustering, self kNN).
- d_F is the continuous Fréchet distance unless stated otherwise. Everything in sections 1–3 also
  holds for the discrete variant, with δ_i measured in that variant.

---

## 1. The foundation: the stability lemma

**L1 (Proven).** For any query Q:

```
|d_F(Q, T_i′) − d_F(Q, T_i)| ≤ d_F(T_i, T_i′) ≤ δ_i
```

*Proof.* d_F is a pseudometric on curves (Alt & Godau 1995), so it satisfies the triangle
inequality: d_F(Q, T_i′) ≤ d_F(Q, T_i) + d_F(T_i, T_i′), and symmetrically. ∎

**L2 (Proven, by construction).** `simplification_bound` ([objective.cpp](engine/src/teacher/objective.cpp))
builds an explicit matching: each kept segment walks its stretch of the original at the same share
of arc length. Any monotone matching of the two curves gives an upper bound on the Fréchet distance.
Along each original edge both points move linearly, so their distance is convex in the parameter
and its maximum sits at a vertex, which is where the code evaluates it. ∎

> To write out: a short proof that the arc-length walk is monotone and covers both curves, and the
> convexity step (distance between two linearly moving points is convex).

**Why it matters.** Every later step uses one consequence: with original queries, **only δ_i
matters** for how far a distance can move, and **only one trajectory's choice** enters each
distance.

---

## 2. Slack: answers only change near their decision boundaries

Define the **slack** of an answer as the distance from the query's value to its decision boundary.
L1 gives a sufficient condition for each query type: the answer is unchanged while δ_i is smaller
than the slack.

### S1. Join (Proven)

A held-out join pair (Q, T_i) at threshold ε is matched iff d_F(Q, T_i′) ≤ ε. With
s = |d_F(Q, T_i) − ε|: if δ_i < s, L1 keeps d_F(Q, T_i′) on the same side of ε. ∎

### S2. kNN with original queries (Proven)

Only the database side moves: each distance d_j moves to d_j′ with |d_j′ − d_j| ≤ δ_j. Let i be in
the top k and j outside it. i stays ahead of j if d_i + δ_i < d_j − δ_j, that is, if
δ_i + δ_j < d_j − d_i. So:

- **Sufficient for the whole top-k set:** every δ below half the gap between the k-th and
  (k + 1)-th distances.
- **Per trajectory:** i can only lose or gain its place against competitors whose distance is
  within δ_i + δ_j of its own.

Ties are broken by database index in both the evaluator and the teacher, so equal distances do not
change the argument. ∎

This is exactly what the kNN margin term prices ([objective.cpp](engine/src/teacher/objective.cpp),
`raw_separable`): a hit survives with probability 1 while its deviation is below its gap.

### S3. Range with segment-in-box semantics (Proven in space; open in space-time)

The Fréchet distance bounds the Hausdorff distance from above, so every point of T_i is within δ_i
of T_i′ and vice versa. For a box B:

- If T_i reaches more than δ_i inside B, some point of T_i′ lies in B: the answer stays "hit".
- If T_i stays more than δ_i outside B, no point of T_i′ can be in B: the answer stays "miss". ∎

**Open:** the Range boxes are space-time boxes (1 km × 1 km × 7 days). The spatial Fréchet distance
does not bound time. A point on a kept chord has an interpolated timestamp that can differ from the
original point it is matched to. A space-time version needs a deviation bound in (x, y, t), for
example a time-synchronised (SED-style) bound. Until that is written, S3 holds for the spatial part
only.

### S4. Point-in-box Range does not obey slack (Proven, by counterexample)

Take a trajectory with a single vertex inside B, and simplify by dropping that vertex while the
chord still passes through B. The deviation can be arbitrarily small, yet the point-in-box answer
changes from "hit" to "miss". The answer depends on **which vertices are kept**, not on geometry. ∎

**Consequence, and a check against the experiments.** Theory predicts that point-in-box Range can be
improved by vertex placement without any gain in geometric fidelity, and that this advantage should
shrink under segment semantics. The Range-shift check (seed 7) shows exactly that: under `seg1km`
the teacher's Range advantage mostly disappears, while its J4 lead stays through the other terms.
This is a case where theory explains an experimental result.

### Where-at

Where-at compares T_i′ with T_i at sampled times. It is stable under a time-synchronised error
bound, not under Fréchet, and it depends on T_i′ alone. It needs no slack argument: it is per
trajectory by definition.

---

## 3. Separability: per-trajectory decisions are enough

### P1. Each outcome depends on one trajectory (Proven)

With original queries:

- the held-out join outcome of (Q, T_i) is a function of T_i′ only;
- a Range outcome (box, T_i) is a function of T_i′ only;
- the where-at score of T_i is a function of T_i′ only;
- a kNN outcome depends on all T_j′, but by S2 only through competitors near the k-th place.

Each claim follows from the definitions of the queries plus L1. ∎

### P2. J4 is additive up to two sources of coupling (Argued)

Writing x_i for trajectory i's choice (budget and recipe) and o_{q,i}(x_i) for its outcome on query
q, the objective has the form:

```
J = Σ_q w_q · F1_q( Σ_i o_{q,i}(x_i) )   +   Σ_i where_at_i(x_i)
```

The floor normalisation is affine, so it keeps this form. Two things stop J from being a plain sum
over trajectories:

1. **F1 is not linear in the counts.** F1 = 2H / (F + T), with H hits, F found, T true. A first-order
   expansion makes each trajectory's contribution additive. The error is second order, with
   curvature ∂²F1/∂F² = 4H / (F + T)³. It is small when a query has many true partners and **not**
   small when it has few, which is common for join queries.
2. **kNN ranking.** Coupled only near ties (S2).

So separability is a first-order statement with known residuals, not a theorem. The teacher reflects
this: the loop tracks the join exactly (`JoinState`), while the value curves use marginal gains
against fixed partners.

> To strengthen: bound the first-order error as a function of the smallest per-query true count, or
> report the distribution of per-query true counts so the reader can judge it.

### P3. Compressed queries are genuinely coupled (Proven)

With Q′ simplified as well, the triangle inequality gives:

```
|d_F(Q′, T_i′) − d_F(Q, T_i)| ≤ δ_Q + δ_i
```

Each outcome now depends on **two** trajectories' choices. Moreover, the two errors can cancel:
near-duplicate routes simplified the same way can stay close even when each has a large deviation.
So coordination can pay, and only here. ∎

**Consequence.** Theory predicts that coordination matters for the self join, clustering and self
kNN, and much less for held-out queries. That matches the history in the results doc: coordination
mattered under discrete Fréchet and compressed queries, and largely disappeared once the headline
queries became originals.

### P4. The leftover coupling is small in practice (Empirical)

How small the residuals in P2 are cannot be derived; it depends on the data. The evidence:

- Reacting to other trajectories' choices (full loop vs `BUDGET_LOOP_NO_UPDATE`) was worth about
  +0.001 J4 (earlier runs, earlier metrics; that control was removed 2026-11-22).
- Allocation plus per-trajectory recipe choice captures 93–100% of the teacher's gain.

> To strengthen: restore `BUDGET_LOOP_NO_UPDATE` and rerun it under the current metrics, since the
> claim currently rests on runs under the old configuration.

---

## 4. Marginality: price points against one global budget

### M1. Optimality of greedy marginal allocation (Proven, classic)

Given separability, choosing budgets is a **separable resource-allocation problem**:

```
maximise Σ_i v_i(b_i)   subject to   Σ_i b_i = B,   2 ≤ b_i ≤ cap_i,  b_i integer
```

where v_i(b) is trajectory i's best value at budget b. If every v_i is concave in b, giving each next
point to the largest marginal gain is optimal (Gross 1956; Fox 1966; Ibaraki & Katoh 1988). ∎

### M2. Without concavity (Proven at envelope vertices; the gap bound needs writing out)

The allocator ([curves.cpp](engine/src/teacher/curves.cpp)) repeatedly takes the step, of any
length, with the best gain per point from a trajectory's current budget. That is a walk along each
curve's **upper concave envelope**.

- **Everett's theorem (1963):** a solution of the Lagrangian relaxation
  `max Σ_i v_i(b_i) − λ Σ_i b_i` is optimal for the total budget it uses. Whenever every trajectory
  sits at a vertex of its envelope, the allocation is such a solution, so it is optimal for that
  total. ∎
- **The tail:** near the end, the remaining budget can be smaller than the best step, so the last
  steps can stop between envelope vertices. The gap to the optimum is then bounded by how far those
  curves lie below their envelopes. The standard argument bounds it by the largest single
  envelope-to-curve gap; this needs writing out for this allocator, whose final steps are truncated
  to the budget left.

The existing tests check the two exact cases: `TeacherAllocate.IsOptimalOnConcaveCurves` and
`TeacherAllocate.IsOptimalOnEnvelopeVertices`.

### M3. The value curves are close to concave (Empirical, not guaranteed)

There is no general reason for v_i to be concave:

- **RDP's error is not monotone in its split order.** Splitting a segment at its farthest point can
  leave another point, which sat close to the old chord, far from both new chords. So even the RDP
  curve can have bumps.
- **v_i is the best of six recipes at each budget**, and the greedy recipes are not nested. The
  maximum of several curves is generally not concave.

Diminishing returns are plausible on average (more points, smaller deviation, fewer flips), but this
is an empirical property.

> To strengthen: report, per budget share, how far the curves are from their envelopes (for
> example, the total envelope-to-curve gap at the allocated budgets). That quantifies M2's gap
> directly.

### M4. Size-proportional budgets are suboptimal in general (Proven)

At an interior optimum of M1, the marginal value per point is equal across trajectories (the KKT
condition). Size-proportional budgets satisfy this only if every value curve is a rescaled copy of
the others, which fails as soon as trajectories differ in query density or shape. ∎

How much is gained is empirical: allocation alone (`BUDGET_RDP`) gives +0.041 to +0.083 J4 over
`RDP_a0`.

### What marginality does not cover

M1 and M2 optimise **J_T on the training workloads through the value curves**, and the curves use a
per-trajectory kNN check rather than the exact ranking. So the allocation is optimal for a proxy of
a proxy: curves stand in for J_T, and J_T stands in for J4.

---

## 5. What the teacher's search does and does not guarantee

### T1. The loop terminates at a local optimum (Proven, conditionally)

The loop ([teacher.cpp](engine/src/teacher/teacher.cpp), `Search::run`) visits trajectories in
random order and switches a trajectory's recipe only if J_T rises by more than
`kMinImprovement = 1e-12`. J_T is tracked exactly, so every switch strictly increases it. The set of
recipe assignments is finite, so no assignment repeats and the loop must stop. If it stops because a
full pass made no switch, no single trajectory can improve J_T by switching recipe: a local optimum,
equivalently a pure Nash equilibrium of the game where every trajectory maximises the shared J_T. ∎

**Condition:** the loop also stops after `kMaxResponsePasses = 10` passes. If it hits that cap, the
local-optimum guarantee does not hold. The logs show which case occurred.

The same argument applies to budget trading with pairwise moves (strict gain, finite states), with
the local optimum taken with respect to the tried trades.

### T2. The teacher is not a global optimum (Not provable; false as stated)

- The loop finds a local optimum, not a global one.
- The search space is restricted: six fixed recipes per trajectory and budgets up to a cap of about
  4× the mean interior budget. The teacher never selects individual points.
- It optimises J_T, not J4 (section 6).

So "teacher = ceiling" cannot be claimed. The defensible claim is: *the best mix of classic
simplifiers under one budget that coordinate ascent finds, under exact training trackers.* This is
also why the results doc avoids calling `BUDGET_LOOP_TRADE` a universal ceiling.

---

## 6. Generalisation, and the join fix

### G1. Per-trajectory budgets fitted to training pairs overfit (Argued)

The teacher sets one parameter per trajectory (its budget) from a few samples (its training pairs).
It then picks the cheapest budget whose training outcomes pass. Choosing the best option by noisy
estimates systematically overstates its value out of sample: the **optimiser's curse** (Smith &
Winkler 2006). The predictions:

- a train/held-out gap that grows as each trajectory has fewer training pairs;
- the teacher's deviations piling up just below each trajectory's smallest training slack.

The gap itself is observed (join, at 1%: training 0.952, held out 0.791). The two predictions above
are not yet measured; they are the step 0 diagnostics in `JOIN_FLIP_PROBABILITY.md`.

### G2. `flip_probability` is the expected flip rate under its model (Proven)

Assume (a) the simplification moves the query distance by e uniform on [−δ, δ], which L1 allows, and
(b) a nearby query's slack is uniform on [max(0, s − h), s + h]. A query with slack x flips with
probability max(0, δ − x)/(2δ). Integrating over x gives the closed form implemented in
`flip_probability`. ∎ The tests check it against a numerical average of the definition.

Its properties follow from the formula: 0 at δ = 0, at most ½, the kNN margin rule at h = 0, and a
charge starting at δ = s − h instead of δ = s. The bandwidth h is a standard bias-variance trade-off
of a kernel-smoothed estimate.

### G3. The model fits the data (Empirical, not yet tested)

Neither assumption is proven:

- (a) δ_i is a worst-case bound. The real error e is probably concentrated well inside ±δ_i, in
  which case the uniform model overcharges.
- (b) The spread of real query slacks around a training pair is unknown.

Both are measurable on the pseudo split: the distribution of e/δ on training pairs, and training vs
held-out slacks per trajectory.

---

## 7. Conclusions that rest on experiments

Theory gives the direction for most of these; the magnitudes, and some of the claims themselves,
come only from the runs.

| Conclusion | Theory's part | Experimental part | How to strengthen |
|---|---|---|---|
| Allocation is the largest single step (44–64% of the gain) | M1/M4: allocation must help | The size | More datasets |
| Recipe choice adds 29–55% | None: depends on recipe behaviour on this data | All of it | Ablate recipes one at a time |
| Coordination adds little (~+0.001) | P1–P3 bound where it can matter | The size | Rerun `BUDGET_LOOP_NO_UPDATE` under current metrics |
| Allocation + recipe choice captures 93–100% | P2 makes it plausible | The size | Same as above |
| Value curves are near concave | M3: not guaranteed | Not yet measured | Report envelope gaps |
| The join gap is a budget effect, not a shape effect | G1 predicts budget overfitting | Diagnosis on seed 42 under earlier metrics | Step 0 diagnostics under current metrics |
| About two thirds of the training gain transfers | None | All of it | Transfer depends on G1/G3 |
| Range gains come from vertex placement | S4 predicts it can happen | `seg1km` check, seed 7 only | All 5 seeds; save the code |
| Results hold under Range distribution shift | None | Seed 7 only | All 5 seeds |
| Partners assumed RDP in the value curves barely matter | Partly: only the Range price depends on partners | Seeds 7 and 99 | Keep the patch |
| The 5 seeds replicate the result | None | Low variance across seeds | Seeds share most trajectories, so this shows stability, not independent replication |

---

## 8. Design choices that cannot be proven

These can be motivated but not derived:

- **Equal weights in J4**, and its arithmetic mean. The results doc also reports geometric and
  harmonic means; which aggregate is "right" is a normative choice.
- **Metric tolerances** (1 km boxes, 25–200 m where-at ramp, 50–200 m join thresholds). They were
  tightened after earlier results had been seen, which should be stated wherever results are
  reported.
- **Continuous Fréchet as the distance.** The whole slack and separability argument relies on it
  being a metric and on original queries. Under vertex-based measures (EDR, synchronised Euclidean),
  L1 can fail, so P1–P3 can fail, and coordination may matter more. That is a limit on how far the
  conclusions transfer, especially when comparing with MLSimp and RL4QDTS.
- **The recipe set and the budget cap**, which define what the teacher can find (T2).

---

## 9. Suggested structure for the thesis chapter

1. Setting and the stability lemma (L1, L2).
2. Slack per query type (S1–S4), including the point-in-box counterexample and its experimental
   confirmation.
3. The separability proposition (P1, P2), contrasted with compressed queries (P3).
4. Allocation optimality (M1, M2, M4), with the envelope gap measured (M3).
5. What the teacher guarantees (T1) and what it does not (T2).
6. Generalisation (G1–G3) as motivation for the expected-outcome join term.
7. Experiments, framed as measuring the residuals the theory leaves open: P4, M3, G3 and section 7.

---

## References (to verify before citing)

These are cited from memory; check years, venues and exact theorem statements.

- H. Alt, M. Godau. Computing the Fréchet distance between two polygonal curves. *Int. J.
  Computational Geometry & Applications*, 1995.
- H. Everett III. Generalized Lagrange multiplier method for solving problems of optimum allocation
  of resources. *Operations Research*, 1963.
- B. Fox. Discrete optimization via marginal analysis. *Management Science*, 1966.
- O. Gross. A class of discrete-type minimization problems. RAND report, 1956.
- T. Ibaraki, N. Katoh. *Resource Allocation Problems: Algorithmic Approaches*. MIT Press, 1988.
- J. E. Smith, R. L. Winkler. The optimizer's curse: skepticism and postdecision surprise in
  decision analysis. *Management Science*, 2006.
- H. Imai, M. Iri. Polygonal approximations of a curve: formulations and algorithms. 1988.
- P. K. Agarwal, S. Har-Peled, N. H. Mustafa, Y. Wang. Near-linear time approximation algorithms for
  curve simplification. *Algorithmica*, 2005.
