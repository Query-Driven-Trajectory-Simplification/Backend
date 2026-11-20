"""Shows the progress of a train.py run, refreshing every few seconds."""

import argparse
import json
import os
import time
from pathlib import Path

WIDTH = 30


def duration(seconds):
    seconds = int(max(seconds, 0))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days} d {hours} h"
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def bar(fraction):
    fraction = min(max(fraction, 0.0), 1.0)
    filled = round(fraction * WIDTH)
    return "[" + "#" * filled + "-" * (WIDTH - filled) + f"] {fraction * 100:5.1f}%"


def remaining(done, total, since, done_at_since, now):
    if since is None or done_at_since is None or done <= done_at_since or now <= since:
        return None
    rate = (done - done_at_since) / (now - since)
    return (total - done) / rate


def render(data, now):
    lines = [f"MLSimp training  {data.get('title', '')}",
             f"updated {time.strftime('%H:%M:%S', time.localtime(data.get('updated', now)))}"
             f" ({duration(now - data.get('updated', now))} ago)", ""]
    current = data.get("current", {})
    for name, info in data.get("stages", {}).items():
        status = info.get("status")
        if status == "done":
            lines.append(f"  done     {name:<14} {duration(info.get('seconds', 0))}")
        elif status == "running":
            lines.append(f"  running  {name:<14} for {duration(now - info.get('started', now))}")
            if current.get("total"):
                left = remaining(current["done"], current["total"], current.get("since"),
                                 current.get("done_at_since"), now)
                lines.append(f"           {current.get('detail', '')} {current['done']}/{current['total']}")
                lines.append(f"           {bar(current['done'] / current['total'])}"
                             + (f"  ~{duration(left)} left" if left is not None else ""))
            if current.get("stage_total"):
                left = remaining(current["stage_done"], current["stage_total"], current.get("stage_since"),
                                 current.get("stage_done_at_since"), now)
                lines.append(f"  stage    {bar(current['stage_done'] / current['stage_total'])}"
                             + (f"  ~{duration(left)} left" if left is not None else ""))
        else:
            lines.append(f"  waiting  {name}")
    if all(info.get("status") == "done" for info in data.get("stages", {}).values()):
        lines += ["", "Training finished."]
    elif now - data.get("updated", now) > 900:
        lines += ["", "No update for 15 minutes: training may have stopped. Run the same train.py command to resume."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the --out folder of the train.py run")
    parser.add_argument("--every", type=float, default=3, help="seconds between refreshes")
    parser.add_argument("--once", action="store_true", help="print once and exit")
    args = parser.parse_args()
    path = args.out / "progress.json"
    while True:
        try:
            text = render(json.loads(path.read_text()), time.time())
        except (OSError, ValueError):
            text = f"Waiting for {path} ..."
        if args.once:
            print(text)
            return
        os.system("cls" if os.name == "nt" else "clear")
        print(text, flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
