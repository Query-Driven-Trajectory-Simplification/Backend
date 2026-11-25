# Introduction: draft (IEEE conference style)

Working draft for Section I. It follows the flow of MLSimp's introduction (Song et al., PVLDB 18(2)),
rewritten around our method. Part 1 maps the two flows, Part 2 is the draft, and Part 3 lists what
each claim rests on.

**Markers used in the draft:**

| Marker | Meaning | Before submission |
|---|---|---|
| `[TODO: …]` | A placeholder or a fact still to check | Fill in |
| `[PLANNED: …]` | Describes the student, which is under development. The surrounding text is written as it would read if the student works | Replace with results, or switch to the fallback text in Part 3 |

Nothing marked `[PLANNED]` may stay in a submitted version. Part 3 tracks the student's status and
has a fallback version of each affected paragraph in case the student does not work out.

---

## Part 1: The flow, paragraph by paragraph

| # | MLSimp's paragraph | Ours |
|---|---|---|
| 1 | Motivation: GPS volumes, simplification | Same |
| 2 | Error-driven methods (EDTS) ignore queries; survey numbers | Same, shorter |
| 3 | QDTS proposed; RL4QDTS is the state of the art | QDTS; RL4QDTS and MLSimp as the state of the art, both scoring individual points |
| — | Example 1 + Fig. 1: RL4QDTS step by step | Example 1 + Fig. 1: where point scoring puts the budget, and what that costs |
| 4 | "Three major challenges", numbered | Three challenges: (1) pricing points across trajectories, (2) knowing when an answer flips, (3) learning without targets |
| 5 | MLSimp, with **globality** and **uniqueness** in one sentence each | A teacher–student framework, with **marginality** and **slack** in one sentence each |
| 6 | Each challenge answered by a mechanism | Challenges 1 and 2 answered by the teacher's stages, challenge 3 by the student |
| — | Example 2: MLSimp on the same Fig. 1 | Example 2: the teacher on the same Fig. 1 |
| 7 | Four contribution bullets, numbers in the last | Five bullets: teacher, principles, student `[PLANNED]`, protocol, experiments |
| 8 | Outline | Same, with a section for the student |

MLSimp's two models also learn from each other (GNN-TS and Diff-TS in a mutual-learning loop). Our
pair is asymmetric: the teacher computes, the student imitates. That difference is worth stating in
Section II.

**IEEE conventions** (MLSimp uses PVLDB's author-year style): numbered citations in order of first
appearance, "Fig. 1", sections numbered in Roman numerals, contributions as a bulleted list. For an
IEEE conference paper, keep the introduction to roughly one page, about 1,000–1,200 words; MLSimp's
runs about 1,600.

---

## Part 2: Draft

### I. Introduction

The proliferation of GPS-enabled devices generates trajectory data at a rate that makes it costly to
store, transmit and query [1], [2]. Trajectory simplification reduces this cost by retaining only a
subset of each trajectory's points while keeping the information that applications need [3], [4].

Classical methods are error-driven: they retain the points whose removal would cause the largest
geometric error under perpendicular, synchronised or direction-based distances [5]–[7]. Such methods
treat every trajectory alike and ignore how the data will be used. An experimental study [8] reports
that error-driven simplification loses up to half of its accuracy on range queries and up to a third
on similarity queries.

Query-driven trajectory simplification (QDTS) addresses this limitation by retaining the points that
queries depend on. RL4QDTS [9] uses reinforcement learning to add points in regions selected by
sample range queries, and MLSimp [10] scores each point by its correlation with its trajectory and
its difference from its neighbours, then samples points across the whole database according to these
scores. Both methods decide at the level of individual points, and both let the distribution of the
budget across trajectories follow from those point-level decisions.

*Example 1:* Fig. 1(a) shows four trajectories under a shared budget. T1 and T2 follow the same road
and each has a held-out query close to its join threshold. T3 crosses a range query box. T4 lies away
from all queries. A point-scoring method assigns T3 many points inside the box, because the box raises
their scores, and gives T4 only its endpoints. On T1, its highest-scoring points lie next to each
other at a single turn, so the rest of the trajectory is represented by one long segment. T1's
simplified shape then deviates from the original by more than the distance between the query and the
join threshold, and the join answer changes from "match" to "no match".

[TODO: Fig. 1(a) — draw this example: four trajectories, one range box, two held-out queries with
their join thresholds as dashed bands, coloured kept points.]

This example illustrates three challenges that existing QDTS methods do not address.

1) *Pricing points across trajectories.* The budget is shared by the whole database, so a point
should go to the trajectory where it improves query accuracy most. When the distribution of points
follows from point scores, it is decided implicitly: under our evaluation, MLSimp places 49% of its
points inside 100 range boxes that cover 3% of the data, and keeps only the endpoints of 14% of the
trajectories [TODO: seed 7; confirm on all seeds].

2) *Knowing when a query answer changes.* Query accuracy is measured on sets: whether a pair still
joins, whether a neighbour stays in the top k. An answer changes only when the simplified trajectory
deviates by more than the answer's distance to its decision boundary. A point's importance score does
not express this distance, and points chosen independently by score can leave long gaps whose
deviation flips many answers.

3) *Learning without targets.* Learned QDTS methods are trained without knowing what a good
simplification looks like: RL4QDTS learns from rewards alone, and MLSimp from two models that teach
each other. Without targets, a model can neither be trained to match the best achievable decisions nor
be measured against them. Moreover, a simplification fitted to known queries can overfit them, and
evaluations whose queries resemble the training data do not reveal this.

To address these challenges, we propose a **teacher–student framework** for QDTS. The **teacher**
decides, offline, how many points each trajectory keeps and which simplification algorithm selects
them, by optimising against exact trackers of the query metrics. The **student** learns to predict
these per-trajectory decisions from trajectory and workload features, and is what we deploy
[PLANNED: student design and features to be confirmed]. Both rest on two principles.
**Marginality** captures that every point is priced against one budget shared by the whole database,
so it goes where its marginal gain in query accuracy is highest. **Slack** captures that a query
answer changes only when a trajectory's deviation exceeds the answer's distance to its decision
boundary, so points matter where answers are close to changing. Both follow from one property of the
Fréchet distance: simplifying a trajectory moves its distance to any original query by at most the
trajectory's own deviation. As a result, each query outcome depends on a single trajectory, and the
problem becomes nearly separable across trajectories. This has a direct consequence for learning: a
student only needs to predict a budget and an algorithm per trajectory, not the importance of every
point, and the expensive point-level modelling of prior work is unnecessary.

The framework answers each challenge in turn. For the first challenge, the teacher builds for every
trajectory a value curve of query accuracy against budget and allocates the global budget by marginal
value along the curves' concave envelopes, which is optimal when the curves are concave. For the
second, it scores every decision against exact trackers of the query metrics (kNN recall, range F1,
where-at accuracy and join F1), so a choice counts only if it keeps answers on the right side of their
boundaries. It then refines the result by coordinate ascent over the algorithm choices and by trading
points between trajectories. For the third, the teacher's decisions provide the per-trajectory
targets that learned methods lack. The student is trained on them and hands its predictions to the
same global allocator, so the budget is met exactly, and its quality is measured as the share of the
teacher's gain it recovers [PLANNED]. We evaluate every method on held-out queries that neither the
teacher nor the student sees, including join thresholds outside the training set.

*Example 2:* In Fig. 1(b), the teacher moves points from T3, whose range answers are already correct
with fewer points, to T1 and T2, whose join answers are close to their thresholds, until the marginal
value per point is equal across trajectories. It selects T1's points with an algorithm that bounds the
deviation along the whole trajectory, so both join answers are preserved within the same total
budget.

[TODO: Fig. 1(b) — same four trajectories with the teacher's kept points; shade each trajectory's
deviation band against each answer's distance to its threshold.]

In summary, we make the following contributions.

- We propose a **teacher for query-driven trajectory simplification** that jointly chooses
  per-trajectory budgets and simplification algorithms against exact trackers of kNN, range,
  where-at and join accuracy. To the best of our knowledge, it is the first QDTS method that
  allocates the budget explicitly by marginal query value and the first to provide per-trajectory
  targets for learned methods. [TODO: verify the novelty claims against recent QDTS work.]
- We introduce the principles of **marginality** and **slack**. We show that, with original queries
  and continuous Fréchet distance, QDTS is nearly separable across trajectories, give conditions under
  which greedy budget allocation is optimal, and identify compressed queries as the case where
  coordination between trajectories can pay.
- We propose a **student** that learns the teacher's per-trajectory decisions and plugs into the
  same global allocator. It recovers [PLANNED: R]% of the teacher's gain over the strongest
  baseline at [PLANNED: speed-up] lower cost, without building query workloads at deployment.
- We propose an **evaluation protocol on held-out queries**, including unseen join thresholds and
  shifted range workloads, which exposes overfitting to the training queries that conventional
  evaluations hide.
- We report **experiments** on GeoLife with five held-out splits and four budgets against error-driven
  baselines, RL4QDTS and MLSimp. The teacher improves the combined query score over the strongest
  baseline by 0.077 to 0.110 at every budget, and at 1% of the interior points it scores 0.820 against
  0.350 for MLSimp [TODO: RL4QDTS; MLSimp and RL4QDTS on all five splits; second dataset]. We report
  every learned method as its share of the teacher's gain, R = (J_method − J_baseline) /
  (J_teacher − J_baseline) [PLANNED: student results].

The rest of this paper is organised as follows. Section II reviews related work. Section III defines
the problem and the query metrics. Section IV develops marginality and slack. Section V presents the
teacher and Section VI the student [PLANNED]. Section VII reports the experiments, and Section VIII
concludes the paper.

### References (placeholders)

[1] [TODO: trajectory data volume / management survey]
[2] [TODO]
[3] [TODO: trajectory simplification survey]
[4] [TODO]
[5] D. H. Douglas and T. K. Peucker, 1973 (RDP, perpendicular distance). [TODO: full entry]
[6] [TODO: synchronised Euclidean distance, e.g. Meratnia and de By, 2004]
[7] [TODO: direction-based distances (DAD/SAD)]
[8] D. Zhang et al., experimental evaluation of trajectory simplification, 2018. [TODO: full entry]
[9] Z. Wang, C. Long, G. Cong, C. S. Jensen, "Collectively simplifying trajectories in a database: A
    query accuracy driven approach," in *Proc. IEEE ICDE*, 2024. [TODO: pages]
[10] Y. Song, Y. Gu, T. Li, Y. Li, C. S. Jensen, G. Yu, "Quantifying point contributions: A
     lightweight framework for efficient and effective query-driven trajectory simplification,"
     *PVLDB*, vol. 18, no. 2. [TODO: year and pages; arXiv says VLDB 2025, PVLDB metadata 2024]

Not yet cited in the text, but likely needed for the student in Section II: learning from an expert
or oracle in combinatorial optimisation, e.g. M. Gasse et al., learning to branch, NeurIPS 2019; M. B.
Paulus et al., learning to cut by looking ahead, ICML 2022; and knowledge distillation, G. Hinton et
al., 2015. [TODO: verify all three.]

---

## Part 3: What each claim rests on

### Numbers used in the draft

| Claim | Source | Status |
|---|---|---|
| Teacher vs strongest baseline: +0.077 to +0.110 J4 | `TEACHER_RESULTS.md`, 5 seeds | Ready; MLSimp and RL4QDTS not yet in that table |
| Teacher 0.820 vs MLSimp 0.350 J4 at 1% | `mlsimp/RESULTS_seed7.md` on `origin/mlsimp` ("wip results"), seed 7; MLSimp with the authors' selection (`MLSimp-l2`) | **Seed 7 only** |
| MLSimp: 49% of points in 100 boxes covering 3% of points; 14% of trajectories endpoints only | Same file, "Why MLSimp scores low" | Seed 7, 1% budget |
| RL4QDTS performs worse than the teacher | Experiments on `origin/Adding-RL4QDTS` | **No results file in the branch.** Add the numbers or a results file before quoting |
| Error-driven methods lose up to half / a third | [8], as cited by MLSimp | Check against [8] itself |

### Fairness: state it in the paper, not only in an appendix

MLSimp scores below uniform sampling under our protocol (J4 0.15–0.48 with its authors' selection,
against 0.30–0.71 for `UNI_a0`). A reviewer will question that unless the paper shows it is fair:

- **Different evaluation settings.** MLSimp's paper evaluates with 2 km range boxes and EDR at 2 km.
  Ours uses 1 km boxes and join thresholds of 50–200 m under continuous Fréchet. Say in Section VI
  that the gap is under this protocol, and why the protocol is tighter.
- **It is not our training.** The authors' pretrained model gives the same J4 under the same
  selection (0.233–0.590 against our 0.234–0.602).
- **The fixes are documented.** `mlsimp/CHANGES.md` lists every deviation from the authors' code
  (their range queries were their test queries; endpoints were kept on top of the budget). Report the
  authors' setting as well as the corrected one.
- **Report the strongest MLSimp variant too.** With the baselines' per-trajectory budgets
  (`MLSimp-pt`), MLSimp reaches 0.234–0.602. Quoting only the weakest variant invites the criticism
  that it was set up to lose.
- **Workload knowledge.** The teacher sees sample workloads. MLSimp sees 100 range queries; with all
  256,000 it improves (`MLSimp-allboxes`, 0.184–0.513). Mention this when comparing.

The same applies to RL4QDTS once its numbers are in: `rl4qdts/CHANGES.md` already documents the
changes (training on our practice range queries, never the test workload).

### Student: status and what the text depends on

The draft describes the student as it would read if it works. Each row is a `[PLANNED]` claim and
what has to happen before it can stay.

| Claim in the draft | Needed | Status |
|---|---|---|
| A per-trajectory student is enough | Go/no-go: train a budget predictor on `budgets.csv`, keep RDP points, compare J4 with `BUDGET_RDP` on trajectories held out by GeoLife user | Not started |
| It predicts budget **and** algorithm | Export the teacher's chosen recipe per trajectory; `budgets.csv` holds only budgets | Not started |
| It plugs into the same allocator, so the budget is exact | Student predicts value curves (or budgets turned into curves) that `allocate` accepts | Design open |
| It recovers R% of the teacher's gain | Full runs on the 5 splits, with R per budget | Not started |
| It is cheaper and needs no query workloads at deployment | Timing against the teacher including its 366–701 s setup; confirm which features need workloads | Not started |
| Features capture slack (query density near thresholds, kNN near-ties, box density) | Feature set and ablation | Untested; the teacher's budgets depend on database context, so trajectory-only features may not suffice |

**Pitfalls that affect the text:**

- **Leakage.** The seeds share most of their trajectories. Split the student's data by GeoLife user,
  never by seed, or R is inflated.
- **Inherited overfitting.** Plain imitation copies the teacher's join overfitting (held-out join
  0.791 against 0.835 for allocation alone at 1%). Either fix the teacher first (the expected-outcome
  join term, `JOIN_FLIP_PROBABILITY.md`) or train the student through the allocator with
  held-out-style losses. If the student then beats the teacher on held-out join, that becomes a
  headline result and the introduction should say so.
- **Workload knowledge.** If the student's features use query workloads, it is workload-aware like
  MLSimp, and the comparison must say so.

**Fallback if the student does not work out** (framing A: an analysis paper; the rest of the
introduction stays as it is):

- *Challenge 3:* keep the first sentence about learning without targets, and add the generalisation
  point as the main issue: "No method computes what a good query-driven simplification looks like,
  so learned methods are trained without targets and evaluated without a reference. At the same time,
  a simplification fitted to known queries can overfit them, and evaluations whose queries resemble
  the training data do not reveal this."
- *Approach paragraph:* "we propose a **teacher** for QDTS that decides, for every trajectory, how
  many points it keeps and which simplification algorithm selects them", without the student
  sentences.
- *Answers paragraph:* end with "its decisions provide per-trajectory targets that learned methods
  lack", and keep the held-out evaluation sentence.
- *Contributions:* drop the student bullet and the R sentence; mention the student as future work in
  the conclusion.
- *Outline:* the teacher in Section V, experiments in Section VI, conclusion in Section VII.

### Claims to keep careful

- **"Teacher", not "optimal".** The teacher reaches a local optimum within six algorithms and a
  budget cap (`QDTS_THEORY.md`, T1–T2).
- **"Nearly separable".** Proven structurally, approximate in practice (F1 with few partners, kNN
  near ties).
- **Join transfer.** The teacher's held-out join is its weakest term (0.791 against 0.835 for budget
  allocation alone at 1%). The draft does not hide this, but does not mention it either; decide
  whether it belongs in the introduction or in Section VI.
- **Point-in-box range.** The teacher's range advantage largely disappears under segment semantics
  (seed 7). Mention it in Section VI.
