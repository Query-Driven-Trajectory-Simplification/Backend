# Teacher: open problems

A running record of what is wrong or limiting in the teacher, what each of its steps is worth,
and what has been tried. Started 2026-11-20.

Related files:
- [TEACHER_CODE_WALKTHROUGH.md](TEACHER_CODE_WALKTHROUGH.md): how the code works, the current
  correctness issues (section 9) and next steps (section 10).
- [TEACHER_SAMPLE_RUN_RESULTS.md](TEACHER_SAMPLE_RUN_RESULTS.md): the full numbers from the
  sample run used below.

All numbers are from a 15 % Geolife sample (2,524 database trajectories), seed 42, held-out J4
unless stated otherwise. One seed on a sample: differences under about 0.01 J4 are not reliable.

## 1. Which steps are worth it

The teacher is a chain of steps. Each arm adds one step to the one before, so the difference
between neighbouring arms is what that step adds.

| Step | Compared | J4 gain at 0.25 % | J4 gain at 1 % | J_T gain at 0.25 % | Time at 0.25 % |
|---|---|---:|---:|---:|---:|
| Budget search (allocation) | BUDGET_RDP − RDP_a0 | **+0.111** | **+0.040** | +0.154 | 37 s (curves) |
| Recipe choice, equal budgets | EQUAL_CHOICE − RDP_a0 | **+0.091** | **+0.054** | +0.112 | 2 s |
| Recipe choice, searched budgets | BUDGET_CHOICE − BUDGET_RDP | **+0.043** | **+0.033** | +0.072 | (in curves) |
| Loop at searched budgets | BUDGET_LOOP − BUDGET_CHOICE | +0.010 | +0.004 | +0.019 | 91 s |
| Reacting, within the loop | BUDGET_LOOP − BUDGET_LOOP_NO_UPDATE | +0.003 | +0.000 | +0.004 | – |
| Loop at equal budgets | EQUAL_LOOP − EQUAL_CHOICE_RDP_PARTNERS | +0.011 | +0.001 | +0.027 | 95 s |
| Reallocation | BUDGET_LOOP_REALLOCATE − BUDGET_LOOP | **−0.004** | **−0.002** | +0.000 | 427 s |
| Trading | BUDGET_LOOP_TRADE − BUDGET_LOOP | +0.003 | +0.000 | +0.019 | 114 s |
| Interleaved trade + loop | BUDGET_LOOP_TRADE_INTERLEAVED − BUDGET_LOOP_TRADE | −0.001 | −0.001 | +0.001 | 618 s |

Trading and interleaving come from a second run on branch `loop-trade`, which changes the kNN
ranking; within that run `BUDGET_CHOICE` and `BUDGET_LOOP` scored 0.6748 and 0.6850 J4 at
0.25 %, against 0.675 and 0.685 here, so the change barely moves the numbers.

What this says:

- **Allocation and selection carry the teacher.** Budget search and recipe choice together take
  J4 from 0.521 to 0.675 at 0.25 % and from 0.836 to 0.909 at 1 %. Both are cheap.
- **The loop adds little held-out J4.** It adds 0.004–0.010 J4, but 0.012–0.019 J_T. Its gains
  sit mostly in kNN and join, the terms that do not carry over to new queries (section 3).
  Reacting to other trajectories' switches adds almost nothing on top of looking once.
- **Reallocation loses J4.** It has the highest J_T, but lower J4 than the loop it starts from,
  and it is the most expensive step: 427 s at 0.25 % and most of the teacher's 1,962 s at 1 %.
- **Trading raises J_T, not held-out J4.** +0.019 J_T but +0.003 J4 at 0.25 % and +0.000 at 1 %.
  It moves held-out kNN up and held-out join down (section 4).
- **Interleaving adds nothing.** A little more J_T, slightly less J4, at 5–6 times trading's cost.

The steps that change **budgets** using a score of each trajectory alone (allocation), and the
step that changes **recipes** once (selection), pay. Every step that iterates on interactions
between trajectories (loop, reallocation, trading, interleaving) raises J_T but adds at most
0.01 held-out J4, because its gains sit in kNN and join, which do not carry over (section 3).

## 2. The candidate space limits what the teacher can reach

### The problem

For each trajectory the teacher decides two things only: how many points it keeps, and which of
six recipes picks them (RDP, SED, UNIFORM, GREEDY, GREEDY_SED2, GREEDY_RDP2). It never decides
on individual points. If the best point set for a trajectory is not one of the six, no search,
loop or trade can reach it.

### Why kNN and join are missing from point selection

The three greedy recipes are the only query-aware ones. `Greedy::sequence` keeps, one at a time,
the point with the largest gain in:

- **range:** the weight of every uncovered box the point lies in;
- **where-at:** how much the probes' ramp scores improve on the segment the point splits.

kNN and join enter only later: when choosing between the six finished candidates (`pick`, the
loop), and when choosing budgets (curves, allocation, trading). A point that would keep a join
pair matched or keep a trajectory in a query's top k is kept only if one of the six recipes
keeps it for its own reasons.

They are left out of greedy for a reason. Range and where-at are local: a point's gain depends
on nearby boxes and probes, so greedy can update it cheaply. kNN and join depend on Fréchet
distance, a bottleneck measure: a new point changes the distance only if it fixes the one
stretch where the two curves are furthest apart. Finding that point means a Fréchet
computation (O(n·m)) per candidate point per query or partner, which is too slow inside
greedy's inner loop.

### Evidence

| 0.25 % | BUDGET_LOOP | RDP_a0 |
|---|---:|---:|
| Range | **0.910** | 0.484 |
| Where-at | **0.456** | 0.340 |
| kNN | 0.755 | **0.786** |
| Join (held out) | 0.618 | 0.475 |

The teacher mostly picks greedy candidates, which win range and where-at. RDP and SED keep
shape, so they do well on Fréchet distance and therefore on kNN and join. Per trajectory the
teacher gets one or the other, or a fixed half-and-half mix, never a point set that keeps what
a particular join pair or kNN query needs and spends the rest on boxes and probes. That is
consistent with the teacher's held-out kNN falling below plain RDP even though J_T rewards kNN.

### Options, cheapest first

1. **A slack-constrained recipe.** Fréchet distance satisfies the triangle inequality, so kNN
   and join become an error limit per trajectory:
   - held-out join pair (i, j): if simplifying i moves it at most δ from its original, then
     `F(simplified i, j) ≤ F(i, j) + δ`, so the pair stays matched if `δ ≤ ε − F(i, j)`;
   - kNN: true neighbour i of query e stays within e's cutoff if `δ ≤ cutoff(e) − F(e, i)`;
   - a non-pair stays unmatched if `δ ≤ F(i, j) − ε`;
   - self join: both sides move, so `δ_i + δ_j ≤ slack`.

   Each trajectory gets a tolerance, the smallest slack over its relations. The recipe keeps the
   fewest points that meet it (Fréchet-error RDP, or greedy on the worst-deviating point), then
   spends the rest of the budget on range and where-at with the existing greedy. The conditions
   are sufficient, not necessary, so the recipe is conservative, but it costs about one RDP pass
   per trajectory. It would be a seventh candidate; the loop can still prefer the others.
2. **Point swaps scored exactly.** For one trajectory, try removing one kept point and adding
   another, score with the trackers, keep improving swaps. Reaches any point set; kNN needs a
   distance per watching query per swap, which `KnnState::propose_bounded` (branch `loop-trade`)
   keeps cheap. Best run on the trajectories with the most kNN and join relations.
3. **More seed mixes.** Seed a grid of fractions (0, ¼, ½, ¾) of the interior budget from RDP
   instead of only half, and let the loop pick per trajectory. Check that the variants differ at
   the budgets used; a quarter-SED variant was removed earlier because it matched unseeded greedy.
4. **Fréchet error as a greedy term.** Penalise the trajectory's maximum deviation in greedy's
   score, weighted by how many kNN and join relations it has. Simpler than option 1, but
   heuristic, with no guarantee.

Recommendation: option 1 first. It targets the gap the sample shows, at little cost. The next
subsection works it out in detail.

### The slack-constrained recipe in detail

**Why it could carry over to new queries.** Continuous Fréchet distance is a metric. For a
trajectory t, its simplified version t′ and **any** query q:

```
|F(q, t′) − F(q, t)|  ≤  F(t, t′)  =:  ε_t
```

Simplifying t moves its distance to every query by at most ε_t, its own Fréchet error, including
queries the teacher never saw. The current kNN and join terms reward particular ranks and pairs
in the training workloads, which is what does not carry over (section 3). A bound on ε_t protects
every query whose margin is larger than ε_t:

- held-out join (query original, t compressed): only pairs with `|F(q, t) − ε| < ε_t` can flip;
- self join (both compressed): the distance moves by at most `ε_i + ε_j`;
- kNN: a true neighbour t stays in the top k while `F(q, t) + ε_t` stays below every outsider's
  distance minus that outsider's error. Queries with a wide gap between their k-th and (k+1)-th
  neighbour are safe whatever the simplification does.

**A cheap certificate.** The maximum synchronised error (max SED) is never below the Fréchet
distance, since the time-synchronous matching is one valid matching. So a simplification with
max SED ≤ s has Fréchet error ≤ s. The exact min-max SED DP planned for the where-at baseline
gives the tightest Fréchet bound SED can certify at a given budget.

**The recipe** (a seventh candidate in `CandidateMaker`, built like `GREEDY_SED2` and
`GREEDY_RDP2`, which seed `Greedy` with a fixed half of the budget):

1. *Slack s_t* from t's training relations, the smallest margin it must keep:
   `cutoff(e) − F(e, t)` as a kNN neighbour of e, `|ε − F(i, t)|` in a join pair at threshold ε.
   Clamp between a floor and a cap, so trajectories without relations get a default.
2. *Seed:* the shortest prefix of a low-error order (SED or RDP, later the min-max SED DP) whose
   Fréchet error is ≤ s_t. One Fréchet decision per check.
3. *Fill:* the rest of the budget with `Greedy`, for range and where-at.
4. If the seed alone exceeds the budget, the candidate is the minimum-error simplification there.

Selection, the loop and trading then decide per trajectory whether it beats the other six.

**Risks.**
- The bound is worst-case. Real shifts are often much smaller than ε_t, so the recipe can
  overspend on safety.
- Held-out margins are unknown, and slack estimated from training margins may be too tight or too
  loose.
- J_T may rate it below the rank-fitting candidates on training data, so selection may rarely
  pick it even where it would do better on held-out queries. This interacts with the term weights
  and with a fit/check split of the training workloads (TEACHER_CODE_WALKTHROUGH.md 10.7).

**Plan, cheapest first.**
1. *Check whether ε_t explains the current held-out losses* (below). If missed neighbours and
   pairs have large ε_t against their margins, the recipe targets the right thing; if misses
   happen at small ε_t, the problem is elsewhere.
2. *A simple version:* seed with the SED prefix that reaches one global slack (for example 25 m).
   Does selection pick it, and do held-out kNN and join move?
3. *Per-trajectory slack* from training margins, and the min-max SED DP as the seed order if SED's
   greedy order is not tight enough.
4. Report how often each recipe is picked, and held-out kNN and join separately, not only J4.

### Check: does ε_t explain the held-out losses?

Run 2026-11-20 on the 15 % sample, seed 42, with a diagnostic dump in `teacher_eval` (branch
`eps-check`, `EPS_CHECK_DIR`). For each arm: every trajectory's ε_t = F(t, t′); for every held-out
kNN query its true neighbours and returned answers, with original and compressed distances and the
(k+1)-th cutoff; for every held-out join candidate (150 m) both distances. A true neighbour's margin
is `cutoff − F(q, t)`; a join pair's is `ε − F(q, t)`. "At risk" means ε_t ≥ margin. Analysis:
`scratchpad/scripts/eps_analysis.py`.

**0.25 %:**

| | RDP_a0 | BUDGET_CHOICE | BUDGET_LOOP | BUDGET_LOOP_TRADE |
|---|---:|---:|---:|---:|
| Median ε_t, all trajectories | 292 m | 310 m | 309 m | 304 m |
| Median ε_t, held-out kNN neighbours | 225 m | 221 m | 222 m | 221 m |
| kNN recall (raw) | **0.929** | 0.905 | 0.919 | 0.928 |
| Neighbours at risk | 49 % | 55 % | 55 % | 54 % |
| Miss rate, at risk / safe | 14.1 % / 0.3 % | 16.6 % / 0.8 % | 14.3 % / 0.6 % | 12.9 % / 0.4 % |
| Outsiders pulled into a top k | **316** | 423 | 362 | 319 |
| Median ε_t of those outsiders | **282 m** | 596 m | 473 m | 336 m |
| Join recall (raw) | 0.643 | 0.887 | **0.883** | 0.853 |
| Join pairs at risk | 70 % | 65 % | 64 % | 61 % |
| Pairs lost, at risk / safe | 51 % / 0 % | 17 % / 0 % | 18 % / 0 % | 24 % / 0 % |
| Median actual shift / ε_t, kNN neighbours | 0.005 | 0.013 | 0.013 | 0.012 |

Miss rate by ε_t / margin, `BUDGET_LOOP` at 0.25 % (`RDP_a0` in brackets):

| ε_t / margin | < 0.25 | 0.25–0.5 | 0.5–1 | 1–2 | > 2 |
|---|---:|---:|---:|---:|---:|
| kNN neighbours missed | 0.3 % (0.0) | 0.7 % (0.5) | 1.0 % (0.4) | 4.5 % (2.2) | 18.8 % (19.3) |
| Join pairs lost | – (0.0) | 0 % (0) | 0 % (0) | 8 % (13) | 23 % (65) |

At 1 % the picture is the same with smaller numbers: `RDP_a0` has the lowest ε_t (median 60 m
against 117 m for `BUDGET_LOOP`), kNN recall 0.982 against 0.967, and 81 against 145 outsiders
pulled in (median ε_t 62 m against 94 m).

What it shows:

- **ε_t is the right risk measure.** Below the bound almost nothing is lost: 0.3–0.8 % of safe
  kNN neighbours and exactly 0 join pairs (the inequality guarantees it). Misses climb steeply
  once ε_t passes the margin.
- **But the bound is very loose.** Actual distance shifts are about 1 % of ε_t (median 0.005–0.013
  at 0.25 %, 0 at 1 %), and only 13–17 % of at-risk neighbours are actually missed. At 0.25 % the
  median ε_t of a kNN neighbour (≈ 220 m) is above its margin for half of them, so a strict
  guarantee would cost far more points than the budget has.
- **The teacher's held-out kNN loss is outsiders pushing in, not neighbours falling out.**
  At-risk miss rates for true neighbours are the same as RDP's (14.3 % against 14.1 %). The
  difference is the outsiders pulled into a top k: 362 against 316 for `BUDGET_LOOP`, 423 for
  `BUDGET_CHOICE`, and those outsiders are compressed much harder (median ε_t 473–596 m against
  282 m). Raw recall is exactly 1 − outsiders / (queries · k), so this accounts for the whole gap.
  Trading closes it (319 outsiders, ε_t 336 m), which is why trading brings held-out kNN back to
  RDP's level.
- **The teacher's join is better than RDP's on held-out pairs** (recall 0.88 against 0.64), and
  not through ε_t: at a similar share of at-risk pairs it loses far fewer of them (18 % against
  51 %). Its join fitting does carry over in part; the drop from training (0.900 → 0.618
  normalised) is relative to an inflated training score, not below the baseline.

What it means for the recipe:

- A per-neighbour slack is aimed at the wrong trajectories. The losses come from trajectories
  that are nobody's training neighbour, compressed to almost nothing, whose simplified shapes
  then land close to new queries.
- The useful form is a **cap on ε_t for every trajectory**: a global slack, or a budget floor
  defined by Fréchet error rather than by points. Step 2 of the plan (one global slack) tests
  exactly this; R1's budget floor is the same idea from the allocation side.
- Because the bound is loose, the slack should be set empirically (for example the ε_t at which
  intruders appear), not from the worst case.

One seed on a sample; the dump does not include the interleaved arm.

### Test: three ways to cap ε_t

Three variants were run on the same sample (seed 42, 15 %, held-out workloads, ε_t dump). They
were built in a scratch copy of loop-trade and are not committed to any branch. All three use the
slack S = the 75th percentile of equal-budget RDP Fréchet error at the given share: 780 m at
0.25 % and 146 m at 1 %.

- **global**: a 7th candidate recipe, `SLACK`. It keeps the shortest RDP prefix with Fréchet
  error ≤ S, then fills the rest of the budget with `Greedy`. If that prefix doesn't fit the
  budget, the candidate falls back to plain RDP.
- **traj**: the same candidate, but each trajectory's slack is its smallest training margin
  (kNN cutoff − d, join |ε − d|, at least 10 m), capped at S.
- **floor**: no new recipe. Allocation and trading never let a trajectory fall below the shortest
  RDP prefix within S. S was raised in steps of ×1.25 until the floors fit in 80 % of the budget.
  That ended at 1219 m / 183 m, with the floors using 74 % / 78 % of all points.

Held-out J4 (kNN / join in brackets), against the baseline run:

| Arm, share | Baseline | global | traj | floor |
|---|---:|---:|---:|---:|
| `BUDGET_LOOP` 0.25 % | 0.6850 (0.755 / 0.618) | 0.6848 | 0.6850 | **0.6448** (0.741 / 0.552) |
| `BUDGET_LOOP_TRADE` 0.25 % | 0.6876 (0.785 / 0.598) | 0.6856 | 0.6892 | **0.6531** (0.756 / 0.569) |
| `BUDGET_LOOP` 1 % | 0.9126 (0.902 / 0.882) | 0.9125 | 0.9128 | 0.8982 (0.908 / 0.832) |
| `BUDGET_LOOP_TRADE` 1 % | 0.9127 (0.903 / 0.877) | 0.9133 | 0.9117 | 0.9004 (0.906 / 0.839) |

How many trajectories the teacher assigned `SLACK`, out of 2524: global 38–40 at 0.25 % and
163–180 at 1 %; traj 1 and 21–23.

Outsiders pulled into a held-out top k (median ε_t in brackets), `BUDGET_LOOP`:

| Share | Baseline | global | traj | floor |
|---|---:|---:|---:|---:|
| 0.25 % | 362 (473 m) | 366 (467 m) | 362 (473 m) | 383 (355 m) |
| 1 % | 145 (94 m) | 147 (92 m) | 145 (94 m) | 136 (112 m) |

Results:

- **global and traj change nothing.** All differences are within ±0.002, which is noise for one
  seed. The candidate is picked rarely, and only where it is close to what was already picked.
  traj's slacks are tiny (median 14.6 m), so for almost every trajectory the prefix doesn't fit
  and the candidate is just RDP again. Under J_T, a recipe that protects queries the teacher
  can't see never wins the per-trajectory choice.
- **floor costs 0.01–0.04 J4.** It pushes the intruders' ε_t down (473 → 355 m at 0.25 %), but
  there are more intruders, not fewer, and join and where-at fall sharply. Any floor that fits the
  budget takes three quarters of the points away from query-driven allocation. At 1 % held-out
  kNN gains a little (+0.006) and join loses 0.05.
- **So capping ε_t is not the fix.** The safe point budget is too large: the cap that fits is
  1.5× the 75th percentile, and lowering the outsiders' ε_t by a quarter did not reduce their
  number. Together with the loose bound (actual shift ≈ 1 % of ε_t), this means the held-out
  kNN loss depends on *where* the error goes relative to other trajectories, not on its size.
  This closes options 1–2 of the plan above; option 3 (per-neighbour slack, min-max SED seed)
  is not expected to do better, because traj was already that idea in its cheap form.

## 3. kNN and join do not carry over to new queries

Training terms against held-out terms for `BUDGET_LOOP` at 0.25 %:

| Term | Training | Test | RDP_a0 test |
|---|---:|---:|---:|
| Range | 0.914 | 0.910 | 0.484 |
| Where-at | 0.456 | 0.456 | 0.340 |
| kNN | 0.821 | **0.755** | 0.786 |
| Join | 0.900 | **0.618** | 0.475 |

The teacher fits the particular neighbours and pairs in its training workloads. This is the main
limit on the held-out score, and the reason the loop's J_T gains turn into little J4. Section 2's
candidate space is one likely cause; the steps in TEACHER_CODE_WALKTHROUGH.md section 10.7 are
others to try. The ε_t check in section 2 measured where the held-out kNN loss comes from:
heavily compressed outsiders pushing into new queries' top k, not true neighbours falling out.
Capping ε_t (globally, per trajectory, or as a budget floor) did not reduce that loss; see
"Test: three ways to cap ε_t".

### Test: a margin term (branch `experiment/margin-term`, local, not committed)

The objective is changed so that the loop and trading see near misses, not only crossings. Both
parts are off unless an environment variable is set, and a new test checks that the tracked
values equal a full recompute.

- `TEACHER_KNN_MARGIN=β`: each training query scores hits plus *safe* hits, out of 2k. A true
  neighbour is safe when its compressed distance × (1 + β) is still at or below the nearest
  non-neighbour's in the query's watch list.
- `TEACHER_JOIN_MARGIN=β`: every join term gets a twin, with half the join weight. In the twin, a
  positive pair counts only within ε(1 − β), and a negative pair already counts as a false match
  within ε(1 + β).

Held-out J4 on the sample at β = 0.1. The baseline is the same code without a margin.

| Arm, share | Baseline | kNN margin | join margin | both |
|---|---:|---:|---:|---:|
| `BUDGET_LOOP` 0.25 % | 0.6850 | 0.6843 | 0.6859 | 0.6842 |
| `BUDGET_LOOP_TRADE` 0.25 % | 0.6876 | 0.6958 | 0.6909 | **0.6998** |
| `BUDGET_LOOP` 1 % | 0.9126 | 0.9120 | 0.9093 | 0.9067 |
| `BUDGET_LOOP_TRADE` 1 % | 0.9127 | 0.9093 | 0.9046 | **0.8993** |

- **At 0.25 % trading gains 0.012.** The gain is in held-out join, which rises from 0.598 to
  0.643, while held-out kNN dips from 0.785 to 0.772. Before, trading traded held-out join for
  held-out kNN; with the margin it does much less of that.
- **Part of this may be re-weighting.** The kNN margin halves the weight of each hit, which moves
  trading's balance towards join. Even the kNN-only run raised held-out join to 0.628.
- **At 1 % it loses 0.013, through join** (0.877 → 0.830). The training pairs are already
  about 99 % matched, so the twin threshold of ±15 m mostly spends points on pairs near the
  boundary, and that does not carry over.
- **Trading is 6–7 times slower with the kNN margin** (757 s against 114 s at 0.25 %), because
  the bounded proposal falls back to exact distances.
- Tracked and exact J_T matched in every run.

Still to do:
1. Let β depend on the share, for example 0.1 at 0.25 % and about 0.02 at 1 %.
2. Run a control with kNN weight 0.5 and no margin (`--term-weights 0.5,1,1,1`).
3. Make the bounded proposal margin-aware.

Only then is a full-data run of the margin worth it.

## 4. Trading and interleaved trade + loop

### What trading does (branch `loop-trade`, `BUDGET_LOOP_TRADE`)

Starting from `BUDGET_LOOP`, up to 3 rounds. Each round scores every trajectory's best recipe at
b ± δ (δ ∈ {1, 2, 4}) against the current state, extends the kNN watch lists to the offered
versions, then pairs the best buyers with the cheapest sellers. Each pair is applied exactly,
seller first, and kept only if the combined exact gain is positive. At the end it returns the
starting state if exact J_T fell.

### Interleaved trade + loop (experiment, branch `loop-trade-interleaved`)

`BUDGET_LOOP_TRADE_INTERLEAVED`, on branch `loop-trade-interleaved` on top of `loop-trade` (not for
merging): each cycle
runs trading, rebuilds the six candidates at the new budgets, and reruns the loop so every
trajectory re-picks its recipe given its partners' new budgets. A cycle is kept only if exact J_T
rises; at most 4 cycles. If the loop ends below the trade, the trade's result is kept.

**Unit-test databases** (all 26 teacher tests pass with the arm included), J_T per cycle:

| Start | Trade | Loop | 2nd trade | 2nd loop |
|---:|---:|---:|---:|---:|
| 0.670782 | 0.734056 | 0.734145 | 0.735773 | 0.735773 |
| 0.523870 | 0.628736 | 0.628736 | 0.635124 | 0.635124 |
| 0.591772 | 0.630447 | 0.630963 | – | – |
| 0.226532 | 0.335063 | 0.338558 | 0.378745 | 0.378745 |

On these tiny databases the second trade cycle adds more than the reloop (reloop +0.0000 to
+0.0035, second trade up to +0.04). That points at trading's 3-round limit as the bottleneck
rather than stale recipes.

**Sample run** (branch `loop-trade` plus the experimental arm, 2026-11-20, 50 min in total):

| Arm | J4 0.25 % | J4 1 % | J_T 0.25 % | J_T 1 % | Teacher time 0.25 % / 1 % |
|---|---:|---:|---:|---:|---:|
| BUDGET_CHOICE | 0.6748 | 0.9093 | 0.7545 | 0.9438 | – |
| BUDGET_LOOP | 0.6850 | 0.9126 | 0.7728 | 0.9564 | 59 s / 114 s |
| BUDGET_LOOP_TRADE | **0.6876** | **0.9127** | 0.7915 | 0.9614 | +114 s / +207 s |
| BUDGET_LOOP_TRADE_INTERLEAVED | 0.6871 | 0.9121 | 0.7926 | 0.9623 | +618 s / +1,303 s |

Per term at 0.25 %, training → held-out test:

| Arm | kNN | Join | Range | Where-at | Self join | J4-self |
|---|---|---|---:|---:|---:|---:|
| BUDGET_LOOP | 0.821 → 0.755 | 0.900 → 0.618 | 0.910 | 0.456 | 0.801 | 0.699 |
| BUDGET_LOOP_TRADE | 0.875 → **0.785** | 0.920 → **0.598** | 0.905 | 0.463 | 0.824 | 0.721 |
| BUDGET_LOOP_TRADE_INTERLEAVED | 0.878 → 0.784 | 0.920 → 0.596 | 0.905 | 0.464 | 0.824 | 0.721 |

What it shows:

- **Trading improves the in-database setting, not held-out J4.** It raises J_T by +0.019 and
  +0.005, the self join and J4-self by +0.022 and +0.004, but held-out J4 by only +0.003 and
  +0.000. It trades held-out join (−0.020 at 0.25 %) for held-out kNN (+0.029), which brings
  held-out kNN back to about plain RDP's level.
- **Trading was still accepting trades when it stopped.** Round 3 kept 77 trades at 0.25 % and
  126 at 1 %; the interleaved arm's later trade rounds kept accepting at 1 % (51, 29, 13 per
  round). But those extra trades added only +0.001 J_T and no J4, so lifting the 3-round limit
  (problem 3 below) is not worth it for J4.
- **The reloop adds almost nothing.** After the first trade, the loop made 34 switches at 0.25 %
  (+0.0002 J_T) and 10 at 1 % (+0.0002); later cycles made 4, 1 or 0. Recipes barely go stale
  when budgets move.
- **Interleaving is not better.** J4 −0.0005 and −0.0006 against `BUDGET_LOOP_TRADE`, for 5–6
  times the time. Do not open a PR for it.
- **The tracked J_T matched the exact value** after every trading run (for example 0.791499 /
  0.791499), and the kNN list correction was +0.000000 in every round, so problem 1 below did not
  occur on this sample either.

### Problems found in the trading code

From a review of `loop-trade` at `ee1fa18`. Its 60 teacher and kNN tests pass.

1. **The tracked J_T goes out of sync after `watch()` with original queries.** Adding a member
   that is already inside a query's top k changes the tracker's hits, but `value` is corrected
   only on the compressed-query path. Only the logs are affected; the returned value is exact.
   *Fixed* on `loop-trade` in `135c2a5`: the correction is now taken around both paths, and the
   log says "kNN list correction". 72 tests pass.
2. **Offers are scored before the watch, so some profitable trades are never tried.** Accepted
   trades are rescored exactly, but pairing and the stopping rule use the stale estimates.
   *Fix:* rescore the kNN part of the offers after `watch()`.
3. **At most 3 trades per trajectory, and no loop afterwards.** A budget moves at most 12 points.
   *Fix:* trade until a round keeps nothing; the interleave experiment tests the loop half.
4. **Trades cannot pass the curve cap** (about 4× the mean interior budget, at least 6), which
   was only there to keep curve-building cheap. *Fix:* a separate, higher limit for trading.
5. **The kNN cutoff is not on the new 1 mm grid** (`build_knn` still uses
   `query::frechet_distance`). Only the curves' per-trajectory check uses it.
6. **Earlier results are not comparable.** The new kNN ranking changes the evaluator's kNN
   ground truth, so all earlier kNN and J4 numbers shift slightly.

Fixed by this branch: the continuous-Fréchet tie mismatch between `KnnState` and `query::knn`
(issue 2 in TEACHER_CODE_WALKTHROUGH.md), now one 1 mm-grid ranking in both.

## 5. Full-data run

Full Geolife (16,830 database trajectories, 16.3 M points), branch `bugfix/watch_knn`, seeds 42
and 17. Each run took about 4 h 25 min. Building the kNN workload took 68 min for seed 42 and 51 min
for seed 17; both sit in a V6 cache in the scratchpad.

| Arm | 0.25 % s42 | s17 | mean | 1 % s42 | s17 | mean |
|---|---:|---:|---:|---:|---:|---:|
| `BUDGET_CHOICE` | 0.5938 | 0.5994 | 0.5966 | 0.8813 | 0.8835 | 0.8824 |
| `BUDGET_LOOP` | 0.5979 | 0.6031 | 0.6005 | 0.8828 | 0.8888 | 0.8858 |
| `BUDGET_LOOP_TRADE` | 0.6082 | 0.6117 | 0.6100 | 0.8850 | 0.8905 | 0.8878 |
| Best single recipe (GREEDY_RDP2) | 0.4911 | 0.4902 | 0.4907 | 0.8343 | 0.8389 | 0.8366 |
| Best allocation baseline | 0.4788 | 0.4772 | 0.4780 | 0.8078 | 0.8112 | 0.8095 |
| `RDP_a0` | 0.4609 | 0.4603 | 0.4606 | 0.7965 | 0.7996 | 0.7981 |

The best allocation baseline is `PARTNER_l0.5` at 0.25 % and `RDP_a025` at 1 %, on both seeds.

- **The two seeds agree:** every arm moves by at most 0.006, and the order is the same.
- **Scores are lower than on the sample,** for the teacher and the baselines alike. At 0.25 % the
  teacher is at 0.608 against 0.688, and RDP at 0.461 against 0.521.
- **The lead holds:** +0.12 at 0.25 % and +0.05 at 1 % over the best single recipe, on both seeds.
- **Each step adds on both seeds:** the loop +0.004 / +0.005 at 0.25 % / 1 % on seed 17 (seed 42:
  +0.004 / +0.002), trading +0.009 / +0.002 (seed 42: +0.010 / +0.002).
- **The loop's exact check (C1) did not fire,** and trading's tracked J_T matched the exact value.
- **At 1 % held-out kNN and join are still below RDP on both seeds:** kNN 0.894 / 0.899 against
  0.914 / 0.916, join 0.825 / 0.843 against 0.871 / 0.876. Range and where-at carry the lead
  (section 3).

Time per share (seed 42): value curves 11 / 32 min, loop about 22 min, trading 35 / 27 min.

## 6. Correctness issues still open

Summary; details and fixes in TEACHER_CODE_WALKTHROUGH.md section 9.

| Issue | Where | Status |
|---|---|---|
| The loop can lower J_T, never checked against its start | `Teacher::run`, `KnnState` | fixed on `bugfix/watch_knn` (`57fa6f9`); never fired on the sample or the full data |
| kNN tracker ranks near-ties differently from `query::knn` | `KnnState::hits` | fixed on `loop-trade` |
| Shuffled control keeps the real range boxes | `shuffled_partners` | fixed on `bugfix/shuffled-knn` (`6ee591c`, local); control range 0.920 → 0.849 at 0.25 %, J4 0.6436 → 0.6304 and 0.8805 → 0.8774 |
| Shuffled kNN lists can be shorter than k | `shuffled_partners` | fixed on `bugfix/shuffled-knn` (`a97b68e`, local); shortened lists marked invalid, 257 on the sample |
| `teacher_eval` builds 5 of the 10 configured baselines | `baseline_arms`, `CHEAP` | open |
| Summary shows the held-out join as the self join | `teacher_stats.py` | merged into `main` (PR #57) |
| Trading problem 1 (tracked J_T after `watch()`) | `trade_budgets` | fixed on `loop-trade` (`135c2a5`) |
| Seller matching skipped a buyer's own seller entry | `trade_budgets` | fixed on `loop-trade` (`61cce4d`) |
| Trading problems 2–6 above | `trade_budgets` | open |
