"""Summarises a teacher_eval run: python3 teacher_stats.py <out dir> [summary.md]

Per share: means and SDs over seeds, paired J4 differences with t intervals, the gate against
the best cheap baseline, the coordination checks, and the join at 100 and 300 m.
"""
import csv
import math
import statistics
import sys
from pathlib import Path
from collections import defaultdict

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
METRICS = ["J4", "J5", "J_fam", "norm_knn", "norm_range", "norm_where_at", "norm_join", "norm_clustering"]
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
        for m in METRICS + ["J_T", "join_recall_seen", "join_recall_unseen"]:
            v = num(r.get(m, ""))
            if v is not None:
                data[r["share"]][r["arm"]][m][int(r["seed"])] = v
    other = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # share, arm, eps, seed
    if (run / "thresholds.csv").exists():
        for r in csv.DictReader(open(run / "thresholds.csv")):
            v = num(r["norm_join"])
            if v is not None:
                other[r["share"]][r["arm"]][float(r["eps_m"])][int(r["seed"])] = v
    out = []
    for share in sorted(data, key=float):
        arms = data[share]
        out.append(f"## Share {float(share) * 100:g}%\n")
        out.append("| Arm | " + " | ".join(METRICS) + " | seen / unseen join recall |")
        out.append("|---|" + "---:|" * (len(METRICS) + 1))
        for arm in sorted(arms):
            cells = [mean_sd(list(arms[arm][m].values())) for m in METRICS]
            seen = list(arms[arm]["join_recall_seen"].values())
            unseen = list(arms[arm]["join_recall_unseen"].values())
            su = f"{statistics.fmean(seen):.4f} / {statistics.fmean(unseen):.4f}" if seen and unseen else "–"
            out.append(f"| {arm} | " + " | ".join(cells) + f" | {su} |")
        out.append("")
        cheap = [a for a in CHEAP if a in arms and arms[a]["J4"]]
        best = max(cheap, key=lambda a: statistics.fmean(arms[a]["J4"].values())) if cheap else None
        out.append(f"Best cheap baseline by mean J4: **{best}**\n")
        out.append("| Paired J4 difference | mean [95% CI] |")
        out.append("|---|---|")
        for arm in sorted(arms):
            if arm != "RDP_a0" and "RDP_a0" in arms:
                out.append(f"| {arm} − RDP_a0 | {fmt(paired(arms[arm]['J4'], arms['RDP_a0']['J4']))} |")
        if best and "TEACHER" in arms:
            gate = paired(arms["TEACHER"]["J4"], arms[best]["J4"])
            out.append(f"| **TEACHER − {best} (gate: ≥ .03, CI above 0)** | {fmt(gate)} |")
        if all(a in arms for a in ["RDP_a0", "GATE_a0", "TEACHER_RDP", "TEACHER"]):
            j = {a: arms[a]["J4"] for a in ["RDP_a0", "GATE_a0", "TEACHER_RDP", "TEACHER"]}
            seeds = set.intersection(*(set(v) for v in j.values()))
            inter = {s: j["TEACHER"][s] - j["GATE_a0"][s] - j["TEACHER_RDP"][s] + j["RDP_a0"][s] for s in seeds}
            out.append(f"| Interaction (TEACHER − GATE_a0 − TEACHER_RDP + RDP_a0) | {fmt(paired(inter, {s: 0.0 for s in seeds}))} |")
        for a, b in [("BR_TEACHER", "TEACHER"), ("ITER_TEACHER", "BR_TEACHER"), ("ITER_TEACHER", "TEACHER"),
                     ("BR_a0", "GATE_a0"), ("BR_TEACHER", "BR_SHUFFLED")]:
            if a in arms and b in arms:
                out.append(f"| {a} − {b} | {fmt(paired(arms[a]['J4'], arms[b]['J4']))} |")
        out.append("")
        if share in other:
            out.append("Normalised join F1 at the test's 150 m and at thresholds the teacher does not optimise:\n")
            out.append("| Arm | 100 m | 150 m | 300 m |")
            out.append("|---|---:|---:|---:|")
            for arm in sorted(arms):
                cells = [mean_sd(list(other[share][arm].get(100.0, {}).values())),
                         mean_sd(list(arms[arm]["norm_join"].values())),
                         mean_sd(list(other[share][arm].get(300.0, {}).values()))]
                out.append(f"| {arm} | " + " | ".join(cells) + " |")
            out.append("")
        pairs = [(arms[a]["J_T"][s], arms[a]["J4"][s]) for a in arms for s in arms[a]["J_T"] if s in arms[a]["J4"]]
        if len(pairs) >= 3:
            xs, ys = zip(*pairs)
            out.append(f"Surrogate check: Pearson r(J_T, J4) = {statistics.correlation(xs, ys):.3f} over {len(pairs)} arm-seeds\n")
    text = "\n".join(out)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write("# Teacher summary\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    main()
