"""Summarises a teacher_eval run: python3 teacher_stats.py <out dir> [summary.md]

Per share: means and SDs over seeds, paired J4 differences with t intervals, the gate against
the best cheap baseline, the loop and trade gains, each step's gain in J_T (training) against J4
and J4-self (test), and the self and held-out joins at join.eps_m and at every threshold in
thresholds.csv (teacher_eval --eval-eps), with the thresholds the teacher trained on marked.

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
         ("recipe choice at searched budgets", "BUDGET_RDP", "BUDGET_CHOICE"),
         ("loop", "BUDGET_CHOICE", "BUDGET_LOOP"),
         ("trades", "BUDGET_LOOP", "BUDGET_LOOP_TRADE")]
CHEAP = ["RDP_a0", "RDP_a025", "RDP_GLOBAL", "SED_a0", "UNI_a0", "BU_PED_a0", "BU_DAD_a0",
         "RDP_a1", "SED_a1", "BU_PED_a1", "BU_DAD_a1", "PARTNER_l0.5", "PARTNER_l1"]
EXTRA = ["J_T", "J_T4", "norm_self_join", "norm_join_main"]
JOINS = [("Self", "Normalised self-join F1"), ("HeldOut", "Normalised held-out join F1")]


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
    reader = csv.DictReader(open(run / "results.csv"))
    columns = set(reader.fieldnames or [])
    rows = list(reader)
    # scores[share][arm][metric][seed] = value, one entry per number in results.csv
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        # Runs from before clustering became a diagnostic have no J4 column: derive it
        if num(r.get("J4", "")) is None:
            parts = [num(r.get(c, "")) for c in ["norm_knn", "norm_range", "norm_where_at", "norm_join"]]
            parts = [x for x in parts if x is not None]
            r["J4"] = str(statistics.fmean(parts)) if parts else ""
        for m in METRICS + EXTRA:
            v = num(r.get(m, ""))
            if v is not None:
                scores[r["share"]][r["arm"]][m][int(r["seed"])] = v
    # The join at join.eps_m alone, per join kind. Runs from before the held-out join have no
    # norm_self_join column; their norm_join is the self join and they have no held-out join
    main_join = {"Self": "norm_self_join" if "norm_self_join" in columns else "norm_join",
                 "HeldOut": "norm_join_main" if "norm_self_join" in columns else None}
    # thresholds[join kind][share][arm][eps_m][seed] = normalised join F1, from thresholds.csv,
    # and per join kind the thresholds the teacher trained on
    thresholds = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    trained = defaultdict(set)
    if (run / "thresholds.csv").exists():
        for r in csv.DictReader(open(run / "thresholds.csv")):
            kind = r.get("join", "Self")
            if kind not in dict(JOINS):
                continue  # route clustering rows
            v = num(r["norm_join"])
            if v is None:
                continue
            eps = float(r["eps_m"])
            thresholds[kind][r["share"]][r["arm"]][eps][int(r["seed"])] = v
            if r.get("teacher_threshold") == "1":
                trained[kind].add(eps)
    out = []
    for share in sorted(scores, key=float):
        arms = scores[share]  # arms[arm][metric][seed]
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
        for a, b in [("BUDGET_LOOP", "BUDGET_CHOICE"), ("BUDGET_LOOP_TRADE", "BUDGET_LOOP")]:
            if a in arms and b in arms:
                out.append(f"| {a} − {b} | {fmt(paired(arms[a]['J4'], arms[b]['J4']))} |")
        out.append("")
        # J_T4 leaves out the query-free shape and margin terms; older runs have only J_T
        train_key = "J_T4" if any(arms[a].get("J_T4") for a in arms) else "J_T"
        steps = [st for st in STEPS if st[1] in arms and st[2] in arms and arms[st[1]][train_key]]
        if steps:
            out.append(f"Gain of each step in training ({train_key}) against test (J4, J4-self), "
                       "mean over seeds. Kept = test gain / training gain:\n")
            out.append(f"| Step | Δ{train_key} | ΔJ4 | kept | ΔJ4_self | kept |")
            out.append("|---|---:|---:|---:|---:|---:|")
            for name, before, after in steps:
                row = [f"{name} ({before} → {after})"]
                train = paired(arms[after][train_key], arms[before][train_key])
                row.append(fmt(train))
                for m in ["J4", "J4_self"]:
                    test = paired(arms[after][m], arms[before][m])
                    row.append(fmt(test))
                    kept = test[0] / train[0] if test and train and abs(train[0]) > 1e-9 else None
                    row.append("–" if kept is None else f"{kept:.0%}")
                out.append("| " + " | ".join(row) + " |")
            out.append("")
        for kind, title in JOINS:
            metric = main_join[kind]
            has_main = metric is not None and any(arms[arm][metric] for arm in arms)
            eps_list = sorted({eps for arm in thresholds[kind][share] for eps in thresholds[kind][share][arm]})
            if not has_main and not eps_list:
                continue
            out.append(f"{title} at join.eps_m and at the --eval-eps thresholds. "
                       "* = a threshold the teacher trained on:\n")
            header = (["join.eps_m"] if has_main else []) + [
                f"{eps:g} m{' *' if eps in trained[kind] else ''}" for eps in eps_list]
            out.append("| Arm | " + " | ".join(header) + " |")
            out.append("|---|" + "---:|" * len(header))
            for arm in sorted(arms):
                cells = [mean_sd(list(arms[arm][metric].values()))] if has_main else []
                cells += [mean_sd(list(thresholds[kind][share][arm].get(eps, {}).values()))
                          for eps in eps_list]
                out.append(f"| {arm} | " + " | ".join(cells) + " |")
            out.append("")
    text = "\n".join(out)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write("# Teacher summary\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    main()
