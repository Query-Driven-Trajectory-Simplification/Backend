# Is the Learned Allocator Really ML?

## Short answer

Mostly, the gain can be had without ML, and the point of the model is not speed.

## What the model is

The model is three numbers (weights on trajectory size, density and curve). They are fitted to query workloads by trial and error against the Evaluator's score. That sits on the line between machine learning and parameter tuning.

A hand rule found the same way gets most of the gain:

Gain in the mean of kNN, range and where-at scores over size-proportional, on test workloads:

| Allocation | Gain |
|---|---|
| size^0.25 (hand rule, one number) | +2.7 to +4.3 points |
| Learned 3-number model | +3.3 to +5.1 points, which is a further +0.6 to +1.0 over the hand rule |

## Why it is not about speed

At inference, the model and the hand rule cost the same: compute a weight per trajectory, share out the budget, run SED.

## Why it still counts as ML

- **Learned from data.** It is a parametric policy learned from query workloads, optimized against an objective, and checked for generalization on unseen test queries. This is the same principle as policy search in reinforcement learning.
- **Trained once.** It is reused on new datasets without queries. A non-learned method that adapts to the data would need a query workload and many Evaluator runs for every upload.
- **Balances several signals.** It weighs size, density and curve, and finds the trade-off between kNN, range and where-at. A single hand rule cannot easily do that.

## Expected objection

"That is just tuning three parameters."

**Answer:** more capacity was tried (9 features, and an MLP) and did not improve test accuracy. The evidence says a small learned policy is enough, and that is a result in itself.

## How to present it

- Call it a *learned allocation policy*. Be upfront that it is small, and say why.
- Keep **size^0.25 as a baseline**, so the reader sees exactly what the learning adds.
- Use **MLSimp** as the comparison with a deep-learning method, evaluated on the same queries.
- If a larger ML component is required, a learned model that decides *which* points to keep is the place for it. It is riskier given the remaining time.
