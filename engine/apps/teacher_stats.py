"""Summarises a teacher_eval run: python3 teacher_stats.py <out dir> [summary.md]

Per share: means and SDs over seeds, paired J4 differences with t intervals, the gate against
the best cheap baseline, the coordination checks, each step's gain in J_T (training) against J4
and J4-self (test), and the self join at 100, 150 and 300 m.

norm_join is J4's join term (held-out when join.mode is HeldOut). J4_self scores the in-database
setting: the database's own kNN and self join (compressed on both sides), with J4's range and
where-at. Its kNN and join queries are the ones the teacher trains on, by definition of the setting.
"""
import csv
import math
import statistics
import sys
from pathlib import Path
from collections import defaultdict

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
METRICS = ["J4", "norm_knn", "norm_range", "norm_where_at", "norm_join", "J4_self", "norm_self_knn",
           "norm_self_join_mean", "norm_clustering"]
# Each step of the teacher, from the arm before it to the arm after it
STEPS = [("budget search", "RDP_a0", "BUDGET_RDP"),
         ("recipe choice", "RDP_a0", "EQUAL_CHOICE"),
         ("recipe choice at searched budgets", "BUDGET_RDP", "BUDGET_CHOICE"),
         ("loop, equal budgets", "EQUAL_CHOICE_RDP_PARTNERS", "EQUAL_LOOP"),
         ("loop", "BUDGET_CHOICE", "BUDGET_LOOP"),
         ("looking once", "BUDGET_CHOICE", "BUDGET_LOOP_NO_UPDATE"),
         ("reacting", "BUDGET_LOOP_NO_UPDATE", "BUDGET_LOOP")]
CHEAP = ["RDP_a0", "RDP_a025", "PARTNER_l0.5", "PARTNER_l1"]


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def mean_sd(xs):
    if not xs:
        return "–"
    m = statistics.fmean(xs)
    return f"{m:.4f}" if len(xs) < 2 else f"{m:.4f} ± {statistics.stdev(xs):.4f}"


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    if len(d) < 2:
        return (statistics.fmean(d), None, None, len(d)) if d else None
    m, sd = statistics.fmean(d), statistics.stdev(d)
    h = T95.get(len(d) - 1, 1.96) * sd / math.sqrt(len(d))
    return m, m - h, m + h, len(d)


def fmt(p):
    if p is None:
        return "–"
    m, lo, hi, n = p
    return f"{m:+.4f}" if lo is None else f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={n})"


def main():
    run = Path(sys.argv[1])
    rows = list(csv.DictReader(open(run / "results.csv")))
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        # Runs from before clustering became a diagnostic have no J4 column: derive it
        if num(r.get("J4", "")) is None:
            parts = [num(r.get(c, "")) for c in ["norm_knn", "norm_range", "norm_where_at", "norm_join"]]
            parts = [x for x in parts if x is not None]
            r["J4"] = str(statistics.fmean(parts)) if parts else ""
        for m in METRICS + ["J_T"]:
            v = num(r.get(m, ""))
            if v is not None:
                data[r["share"]][r["arm"]][m][int(r["seed"])] = v
    other = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # share, arm, eps, seed
    if (run / "thresholds.csv").exists():
        for r in csv.DictReader(open(run / "thresholds.csv")):
            if r.get("join", "Self") != "Self":
                continue
            v = num(r["norm_join"])
            if v is not None:
                other[r["share"]][r["arm"]][float(r["eps_m"])][int(r["seed"])] = v
    out = []
    for share in sorted(data, key=float):
        arms = data[share]
        out.append(f"## Share {float(share) * 100:g}%\n")
        out.append("| Arm | " + " | ".join(METRICS) + " |")
        out.append("|---|" + "---:|" * len(METRICS))
        for arm in sorted(arms):
            cells = [mean_sd(list(arms[arm][m].values())) for m in METRICS]
            out.append(f"| {arm} | " + " | ".join(cells) + " |")
        out.append("")
        cheap = [a for a in CHEAP if a in arms and arms[a]["J4"]]
        best = max(cheap, key=lambda a: statistics.fmean(arms[a]["J4"].values())) if cheap else None
        out.append(f"Best cheap baseline by mean J4: **{best}**\n")
        out.append("| Paired J4 difference | mean [95% CI] |")
        out.append("|---|---|")
        for arm in sorted(arms):
            if arm != "RDP_a0" and "RDP_a0" in arms:
                out.append(f"| {arm} − RDP_a0 | {fmt(paired(arms[arm]['J4'], arms['RDP_a0']['J4']))} |")
        if best and "BUDGET_CHOICE" in arms:
            gate = paired(arms["BUDGET_CHOICE"]["J4"], arms[best]["J4"])
            out.append(f"| **BUDGET_CHOICE − {best} (gate: ≥ .03, CI above 0)** | {fmt(gate)} |")
        if all(a in arms for a in ["RDP_a0", "EQUAL_CHOICE_RDP_PARTNERS", "BUDGET_RDP", "BUDGET_CHOICE"]):
            j = {a: arms[a]["J4"] for a in ["RDP_a0", "EQUAL_CHOICE_RDP_PARTNERS", "BUDGET_RDP", "BUDGET_CHOICE"]}
            seeds = set.intersection(*(set(v) for v in j.values()))
            inter = {s: j["BUDGET_CHOICE"][s] - j["EQUAL_CHOICE_RDP_PARTNERS"][s] - j["BUDGET_RDP"][s] + j["RDP_a0"][s] for s in seeds}
            out.append(f"| Interaction (BUDGET_CHOICE − EQUAL_CHOICE_RDP_PARTNERS − BUDGET_RDP + RDP_a0) | {fmt(paired(inter, {s: 0.0 for s in seeds}))} |")
        for a, b in [("BUDGET_LOOP", "BUDGET_CHOICE"), ("BUDGET_LOOP_REALLOCATE", "BUDGET_LOOP"),
                     ("EQUAL_LOOP", "EQUAL_CHOICE_RDP_PARTNERS"), ("BUDGET_LOOP", "BUDGET_LOOP_SHUFFLED")]:
            if a in arms and b in arms:
                out.append(f"| {a} − {b} | {fmt(paired(arms[a]['J4'], arms[b]['J4']))} |")
        out.append("")
        steps = [st for st in STEPS if st[1] in arms and st[2] in arms and arms[st[1]]["J_T"]]
        if steps:
            out.append("Gain of each step in training (J_T) against test (J4, J4-self), "
                       "mean over seeds. Kept = test gain / training gain:\n")
            out.append("| Step | ΔJ_T | ΔJ4 | kept | ΔJ4_self | kept |")
            out.append("|---|---:|---:|---:|---:|---:|")
            for name, before, after in steps:
                row = [f"{name} ({before} → {after})"]
                train = paired(arms[after]["J_T"], arms[before]["J_T"])
                row.append(fmt(train))
                for m in ["J4", "J4_self"]:
                    test = paired(arms[after][m], arms[before][m])
                    row.append(fmt(test))
                    kept = test[0] / train[0] if test and train and abs(train[0]) > 1e-9 else None
                    row.append("–" if kept is None else f"{kept:.0%}")
                out.append("| " + " | ".join(row) + " |")
            out.append("")
        if share in other:
            out.append("Normalised self-join F1 at the teacher's 150 m and at thresholds it does not optimise:\n")
            out.append("| Arm | 100 m | 150 m | 300 m |")
            out.append("|---|---:|---:|---:|")
            for arm in sorted(arms):
                # Runs from before the held-out join have the self join in norm_join
                at_150 = arms[arm]["norm_self_join"] or arms[arm]["norm_join"]
                cells = [mean_sd(list(other[share][arm].get(100.0, {}).values())),
                         mean_sd(list(at_150.values())),
                         mean_sd(list(other[share][arm].get(300.0, {}).values()))]
                out.append(f"| {arm} | " + " | ".join(cells) + " |")
            out.append("")
    text = "\n".join(out)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write("# Teacher summary\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    main()
