import csv
import math
import statistics
import sys
from collections import defaultdict

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
METRICS = ["J5", "J_fam", "norm_knn", "norm_range", "norm_where_at", "norm_join", "norm_clustering"]
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
    rows = list(csv.DictReader(open(sys.argv[1])))
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        for m in METRICS + ["J_T", "join_recall_seen", "join_recall_unseen"]:
            v = num(r.get(m, ""))
            if v is not None:
                data[r["share"]][r["arm"]][m][int(r["seed"])] = v
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
        cheap = [a for a in CHEAP if a in arms and arms[a]["J5"]]
        best = max(cheap, key=lambda a: statistics.fmean(arms[a]["J5"].values())) if cheap else None
        out.append(f"Best cheap baseline by mean J5: **{best}**\n")
        out.append("| Paired J5 difference | mean [95% CI] |")
        out.append("|---|---|")
        for arm in sorted(arms):
            if arm != "RDP_a0" and "RDP_a0" in arms:
                out.append(f"| {arm} − RDP_a0 | {fmt(paired(arms[arm]['J5'], arms['RDP_a0']['J5']))} |")
        if best and "TEACHER" in arms:
            gate = paired(arms["TEACHER"]["J5"], arms[best]["J5"])
            out.append(f"| **TEACHER − {best} (gate: ≥ .03, CI above 0)** | {fmt(gate)} |")
        if all(a in arms for a in ["RDP_a0", "GATE_a0", "TEACHER_RDP", "TEACHER"]):
            j = {a: arms[a]["J5"] for a in ["RDP_a0", "GATE_a0", "TEACHER_RDP", "TEACHER"]}
            seeds = set.intersection(*(set(v) for v in j.values()))
            inter = {s: j["TEACHER"][s] - j["GATE_a0"][s] - j["TEACHER_RDP"][s] + j["RDP_a0"][s] for s in seeds}
            out.append(f"| Interaction (TEACHER − GATE_a0 − TEACHER_RDP + RDP_a0) | {fmt(paired(inter, {s: 0.0 for s in seeds}))} |")
        out.append("")
        pairs = [(arms[a]["J_T"][s], arms[a]["J5"][s]) for a in arms for s in arms[a]["J_T"] if s in arms[a]["J5"]]
        if len(pairs) >= 3:
            xs, ys = zip(*pairs)
            out.append(f"Surrogate check: Pearson r(J_T, J5) = {statistics.correlation(xs, ys):.3f} over {len(pairs)} arm-seeds\n")
    text = "\n".join(out)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write("# Teacher summary\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    main()
