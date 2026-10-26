"""Evaluation harness.

  python eval/run_eval.py --synthetic            # generated fixtures (pipeline check only)
  python eval/run_eval.py --data data            # your recorded clips (data/labels.csv)
  python eval/run_eval.py --data data --write-config --export-trace

Outputs Markdown + JSON + PNG charts into docs/results/ (or --out).

Metrics: per-challenge detection rate and false-positive rate, session-level
detection / FPR / inconclusive rate, average challenges to verdict, per-clip
latency, and an ablation of (a) passive only, (b) fixed challenge sequence,
(c) agent-chosen sequence, all by offline replay of the same recorded clips.
Thresholds are fitted leave-one-participant-out, so each participant is scored
with a config that never saw their clips.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unmask.agent.session import replay  # noqa: E402
from unmask.scoring import load_config, session_risk  # noqa: E402
from unmask.scoring.tune import fit_config  # noqa: E402
from unmask.vision import analyze_clip  # noqa: E402
from unmask.vision.analyze import ANALYSIS_VERSION  # noqa: E402

CHALLENGES = ("head_turn", "hand_pass", "light_change")
FIXED = ["head_turn", "hand_pass", "light_change"]
FLAGGED = ("SUSPICIOUS", "HIGH_RISK")


def read_labels(root: Path) -> list[dict]:
    rows = list(csv.DictReader(open(root / "labels.csv")))
    for r in rows:
        r["abspath"] = str((root / r["path"]).resolve())
        r.setdefault("condition", "default")
        r["condition"] = r.get("condition") or "default"
    return rows


def analyse_all(rows, cache_dir: Path, ev_dir: Path, detector_factory=None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in enumerate(rows):
        st = os.stat(r["abspath"])
        key = hashlib.sha1(f"{ANALYSIS_VERSION}|{r['abspath']}|{st.st_size}|{st.st_mtime}|{r['challenge']}".encode()).hexdigest()[:16]
        cp = cache_dir / f"{key}.json"
        if cp.exists():
            res = json.loads(cp.read_text())
        else:
            t0 = time.perf_counter()
            det = detector_factory(r) if detector_factory else None
            res = analyze_clip(r["abspath"], r["challenge"], detector=det)
            res["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            evs = []
            for j, (label, jpg) in enumerate(res.pop("evidence", [])):
                p = ev_dir / f"{r['participant']}_{r['label']}_{r['condition']}_{r['challenge']}_{j}.jpg"
                p.write_bytes(jpg)
                evs.append({"label": label, "path": str(p)})
            res["evidence"] = evs
            res.pop("series", None)
            cp.write_text(json.dumps(res))
        res.update(participant=r["participant"], label=r["label"], condition=r["condition"], path=r["path"])
        out.append(res)
        print(f"\r  analysed {i + 1}/{len(rows)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return out


def sessions_from(clips):
    groups = defaultdict(dict)
    for c in clips:
        groups[(c["participant"], c["label"], c["condition"])][c["challenge"]] = c
    return groups


def run_strategies(clips_by_ch, cfg, use_bedrock=False):
    def analyze(ch):
        return clips_by_ch.get(ch)
    present = {ch: True for ch in clips_by_ch}
    return {
        "passive_only": replay(present, analyze, cfg, use_model=False, fixed_order=[]),
        "fixed_sequence": replay(present, analyze, cfg, use_model=False, fixed_order=[c for c in FIXED if c in clips_by_ch]),
        "agent": replay(present, analyze, cfg, use_model=use_bedrock),
    }


def pct(n, d):
    return None if d == 0 else round(100.0 * n / d, 1)


def fmt(v, suffix="%"):
    return "n/a" if v is None else f"{v}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--synthetic", action="store_true", help="generate and use synthetic fixtures")
    ap.add_argument("--n-synth", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "docs" / "results"))
    ap.add_argument("--bedrock", action="store_true", help="agent strategy calls Bedrock (falls back to rules)")
    ap.add_argument("--write-config", action="store_true", help="fit on all data and write scoring_vN.json")
    ap.add_argument("--export-trace", action="store_true", help="write docs/AGENT_TRACE_EXAMPLE.json")
    ap.add_argument("--base-config", default=str(ROOT / "unmask" / "scoring" / "config" / "scoring_v0.json"),
                    help="prior thresholds the cross-validation folds start from. Fixed to the hand-set v0 priors "
                         "(NOT scoring_current.json) so that re-running after --write-config reproduces the same numbers")
    ap.add_argument("--public-prefix", default="divine_",
                    help="only sessions whose data/<label>/<folder> starts with this may appear in public outputs "
                         "(failure frames, example trace); other sessions only feed aggregate numbers")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        from unmask.synth import make_dataset
        data = ROOT / "fixtures" / "generated"
        print(f"generating synthetic fixtures in {data}", file=sys.stderr)
        make_dataset(str(data), a.n_synth)
        provenance = "SYNTHETIC fixtures (procedurally drawn cartoons, not real people or real face swaps)"
    else:
        data = Path(a.data)
        if not (data / "labels.csv").exists():
            sys.exit(f"{data}/labels.csv not found. See docs/DATA_COLLECTION.md or run with --synthetic.")
        provenance = f"recorded clips in {data}"
    rows = read_labels(data)
    work = ROOT / "eval" / "out" / ("synthetic" if a.synthetic else "real")
    clips = analyse_all(rows, work / "cache", work / "evidence")
    base = load_config(a.base_config)
    participants = sorted({c["participant"] for c in clips})
    sess = sessions_from(clips)
    public_prefix = None if a.synthetic else a.public_prefix

    # ---------------------------------------------------- cross-validated replay
    # Leave-one-participant-out when there are several participants. With a single participant
    # that is impossible, so fall back to leave-one-session-out (weaker: see the limitation note).
    by_participant = len(participants) > 1
    unit_of_clip = (lambda c: c["participant"]) if by_participant else (lambda c: c["condition"])
    unit_of_key = (lambda k: k[0]) if by_participant else (lambda k: k[2])
    units = sorted({unit_of_clip(c) for c in clips})
    cv_scheme = "leave-one-participant-out" if by_participant else "leave-one-session-out"
    results = []
    fold_cfgs = {}
    for u in units:
        train = [c for c in clips if unit_of_clip(c) != u]
        cfg = fit_config(train, base, version=f"cv-{u}") if len(units) > 1 else base
        fold_cfgs[u] = cfg
        for key, by_ch in sess.items():
            if unit_of_key(key) != u:
                continue
            strat = run_strategies(by_ch, cfg, a.bedrock)
            per_ch = {}
            for ch in CHALLENGES:
                if ch in by_ch and "passive" in by_ch:
                    per_ch[ch] = session_risk([by_ch["passive"], by_ch[ch]], cfg)
            results.append({"participant": key[0], "label": key[1], "condition": key[2], "strategies": strat,
                            "per_challenge": per_ch, "clips": by_ch,
                            "public": session_is_public(by_ch, public_prefix)})

    # ---------------------------------------------------- metrics
    metrics = {"provenance": provenance, "n_clips": len(clips), "n_participants": len(participants),
               "n_sessions": len(results), "config_base": base.get("version"), "generated_at": time.strftime("%Y-%m-%d %H:%M"),
               "lopo": by_participant, "cv_scheme": cv_scheme, "n_cv_folds": len(units),
               "single_participant": not by_participant, "public_prefix": public_prefix}
    abl = {}
    for sname in ("passive_only", "fixed_sequence", "agent"):
        real = [r["strategies"][sname]["result"] for r in results if r["label"] == "real"]
        fake = [r["strategies"][sname]["result"] for r in results if r["label"] == "swapped"]
        allr = real + fake
        abl[sname] = {
            "detection_rate": pct(sum(x["verdict"] in FLAGGED for x in fake), len(fake)),
            "high_risk_rate": pct(sum(x["verdict"] == "HIGH_RISK" for x in fake), len(fake)),
            "false_positive_rate": pct(sum(x["verdict"] in FLAGGED for x in real), len(real)),
            "inconclusive_rate": pct(sum(x["verdict"] == "INCONCLUSIVE" for x in allr), len(allr)),
            "missed_as_low_risk": pct(sum(x["verdict"] == "LOW_RISK" for x in fake), len(fake)),
            "avg_challenges": round(statistics.mean(len(x["challenges_done"]) for x in allr), 2) if allr else None,
            "n_real": len(real), "n_swapped": len(fake),
        }
    metrics["ablation"] = abl
    pc = {}
    for ch in CHALLENGES:
        real = [r["per_challenge"][ch] for r in results if r["label"] == "real" and ch in r["per_challenge"]]
        fake = [r["per_challenge"][ch] for r in results if r["label"] == "swapped" and ch in r["per_challenge"]]
        pc[ch] = {"detection_rate": pct(sum(x["verdict"] in FLAGGED for x in fake), len(fake)),
                  "false_positive_rate": pct(sum(x["verdict"] in FLAGGED for x in real), len(real)),
                  "n_real": len(real), "n_swapped": len(fake)}
    metrics["per_challenge"] = pc
    lat = defaultdict(list)
    for c in clips:
        lat[c["challenge"]].append(c["timing_ms"]["total"])
    metrics["latency_local_ms"] = {ch: {"mean": round(statistics.mean(v), 1), "p50": round(statistics.median(v), 1),
                                        "p95": round(sorted(v)[max(0, int(0.95 * len(v)) - 1)], 1),
                                        "max": round(max(v), 1), "n": len(v)} for ch, v in lat.items()}
    lam = ROOT / "docs" / "results" / "lambda_latency.json"
    metrics["latency_lambda"] = json.loads(lam.read_text()) if lam.exists() else None
    metrics["unusable_reasons"] = dict(Counter(c["reason"] for c in clips if not c["usable"]))
    orders = Counter(tuple(st["decision"]["challenge"] for st in r["strategies"]["agent"]["trace"]
                           if st["decision"]["action"] == "issue_challenge") for r in results)
    metrics["agent_orders"] = {" > ".join(k) if k else "(none)": v for k, v in orders.most_common()}
    policies = Counter(st["decision"].get("policy") for r in results for st in r["strategies"]["agent"]["trace"])
    metrics["agent_policies"] = dict(policies)

    # ---------------------------------------------------- failures
    fail_dir = out / "failures"
    if fail_dir.exists():
        shutil.rmtree(fail_dir)
    fail_dir.mkdir()
    failures = []
    withheld = 0
    for r in results:
        res = r["strategies"]["agent"]["result"]
        wrong = (r["label"] == "swapped" and res["verdict"] in ("LOW_RISK", "INCONCLUSIVE")) or \
                (r["label"] == "real" and res["verdict"] in FLAGGED)
        if not wrong:
            continue
        if not r["public"]:   # privacy: other people's sessions are aggregate-only
            withheld += 1
            continue
        frames = []
        for ch, c in r["clips"].items():
            for e in c.get("evidence", [])[:1]:
                if os.path.exists(e["path"]):
                    dst = fail_dir / Path(e["path"]).name
                    shutil.copyfile(e["path"], dst)
                    frames.append(str(dst.relative_to(out)))
        failures.append({"participant": r["participant"], "label": r["label"], "condition": r["condition"],
                         "verdict": res["verdict"], "risk": res["risk"], "reasons": res["top_reasons"],
                         "unusable": {ch: c["reason"] for ch, c in r["clips"].items() if not c["usable"]},
                         "frames": frames[:3]})
    metrics["failures"] = failures
    metrics["failures_withheld"] = withheld

    # ---------------------------------------------------- charts
    charts = make_charts(results, metrics, out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "results.md").write_text(render_md(metrics, charts))
    print((out / "results.md").read_text())

    if a.write_config and not a.synthetic:
        cfg_dir = ROOT / "unmask" / "scoring" / "config"
        n = 1 + max([int(p.stem.split("_v")[1]) for p in cfg_dir.glob("scoring_v*.json") if p.stem.split("_v")[1].isdigit()] or [0])
        final = fit_config(clips, base, version=f"v{n}-{time.strftime('%Y%m%d')}")
        final["notes"] = f"Fitted by eval/run_eval.py on {len(clips)} clips from {len(participants)} participants."
        (cfg_dir / f"scoring_v{n}.json").write_text(json.dumps(final, indent=2))
        (cfg_dir / "scoring_current.json").write_text(json.dumps({"extends": f"scoring_v{n}.json"}) + "\n")
        print(f"wrote scoring_v{n}.json and pointed scoring_current.json at it", file=sys.stderr)
    elif a.write_config:
        print("--write-config ignored for synthetic data (never ship thresholds fitted on cartoons)", file=sys.stderr)

    if a.export_trace:
        public = [r for r in results if r["public"]]
        if public:
            export_trace(public, ROOT / "docs" / "AGENT_TRACE_EXAMPLE.json", provenance)
        else:
            print("no public sessions: example trace not exported", file=sys.stderr)


def session_is_public(by_ch: dict, prefix: str | None) -> bool:
    """A session may appear in public outputs only if every clip comes from data/<label>/<prefix...>."""
    if prefix is None:
        return True
    return all(len(Path(c["path"]).parts) >= 3 and Path(c["path"]).parts[1].startswith(prefix) for c in by_ch.values())


def export_trace(results, path: Path, provenance: str):
    """Pick an agent session whose challenge order differs from the fixed default
    (i.e. the vision output visibly steered the agent), preferring a swapped one."""
    def score(r):
        tr = r["strategies"]["agent"]["trace"]
        order = [st["decision"].get("challenge") for st in tr if st["decision"]["action"] == "issue_challenge"]
        adaptive = order and order != FIXED[:len(order)]
        return (bool(adaptive), r["label"] == "swapped", len(tr))
    best = max(results, key=score)
    s = best["strategies"]["agent"]
    doc = {"_about": ("Real exported decision trace from eval/run_eval.py offline replay. Each step lists what OpenCV "
                      "measured (event.features), the per-signal scores before/after, the state the agent saw "
                      "(agent_input), and what it chose and why (decision). Compare signals_after with the next "
                      "decision to see vision output changing the agent's next action."),
           "_provenance": provenance, "participant": best["participant"], "label": best["label"],
           "condition": best["condition"], **{k: s[k] for k in ("session_id", "status", "result", "trace")}}
    path.write_text(json.dumps(doc, indent=2, default=str))
    print(f"wrote {path}", file=sys.stderr)


def make_charts(results, metrics, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    charts = {}
    abl = metrics["ablation"]
    names = ["passive_only", "fixed_sequence", "agent"]
    labels = ["Passive only", "Fixed sequence", "Agent-chosen"]
    keys = [("detection_rate", "Detection rate", "#2b6cb0"), ("false_positive_rate", "False-positive rate", "#c05621"),
            ("inconclusive_rate", "Inconclusive rate", "#718096")]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=130)
    w = 0.26
    for i, (k, lab, col) in enumerate(keys):
        vals = [abl[n][k] or 0 for n in names]
        bars = ax.bar([x + (i - 1) * w for x in range(3)], vals, w, label=lab, color=col)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("% of sessions")
    ax.set_ylim(0, 110)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.set_title(f"Ablation ({metrics['n_sessions']} sessions)", fontsize=10, pad=24)
    fig.tight_layout()
    fig.savefig(out / "ablation.png")
    plt.close(fig)
    charts["ablation"] = "ablation.png"

    lat = metrics["latency_local_ms"]
    chs = [c for c in ("passive", "head_turn", "hand_pass", "light_change") if c in lat]
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=130)
    ax.bar(chs, [lat[c]["p50"] for c in chs], color="#2b6cb0", label="p50")
    ax.scatter(chs, [lat[c]["p95"] for c in chs], color="#c05621", zorder=3, label="p95")
    ax.axhline(30000, color="#a0aec0", ls="--", lw=1)
    ax.text(len(chs) - 0.5, 30000, "API Gateway 30 s limit", fontsize=7, va="bottom", ha="right", color="#4a5568")
    ax.set_ylabel("ms per clip (local)")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "latency.png")
    plt.close(fig)
    charts["latency"] = "latency.png"

    sig_names = ["seam", "texture", "temporal", "pose", "occlusion", "lighting"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=130)
    for j, lab, col, dx in (("real", "Real", "#2f855a", -0.12), ("swapped", "Swapped", "#c53030", 0.12)):
        for i, sname in enumerate(sig_names):
            ys = [r["strategies"]["fixed_sequence"]["result"]["signals"][sname]["score"] for r in results if r["label"] == j]
            ys = [y for y in ys if y is not None]
            ax.scatter([i + dx] * len(ys), ys, s=14, color=col, alpha=0.75, label=lab if i == 0 else None)
    ax.axhline(0.5, color="#a0aec0", lw=1, ls=":")
    ax.set_xticks(range(len(sig_names)), sig_names)
    ax.set_ylabel("signal suspicion (cross-validated)")
    ax.set_ylim(-0.05, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "signals.png")
    plt.close(fig)
    charts["signals"] = "signals.png"
    return charts


def render_md(m, charts):
    L = [f"# Evaluation results", "",
         f"> **Data:** {m['provenance']}  ", f"> {m['n_clips']} clips · {m['n_participants']} participants · "
         f"{m['n_sessions']} sessions · generated {m['generated_at']} · thresholds fitted "
         f"{m['cv_scheme']} ({m['n_cv_folds']} folds)", ""]
    if m.get("single_participant") and not m["provenance"].startswith("SYNTHETIC"):
        L += ["> ⚠️ **Limitation: single participant.** Leave-one-participant-out is impossible with one person, so "
              "thresholds were fitted **leave-one-session-out**. The sessions share the same face, camera and swap source, "
              "so this check is weaker and overstates how well the thresholds will generalise to new people. "
              "More participants are needed for a real generalisation estimate.", ""]
    if m["provenance"].startswith("SYNTHETIC"):
        L += ["> ⚠️ These numbers come from synthetic cartoons that were *designed* to contain swap-like artefacts. "
              "They only show that the pipeline runs end to end; they are **not** evidence of real-world detection performance.", ""]
    L += ["## Ablation: does choosing challenges adaptively help?", "",
          "| Strategy | Detection rate (swapped flagged) | High-risk on swapped | False-positive rate (real flagged) | Inconclusive | Swaps missed as LOW_RISK | Avg challenges |",
          "|---|---|---|---|---|---|---|"]
    for n, lab in (("passive_only", "(a) Passive only"), ("fixed_sequence", "(b) Fixed sequence"), ("agent", "(c) Agent-chosen")):
        x = m["ablation"][n]
        L.append(f"| {lab} | {fmt(x['detection_rate'])} | {fmt(x['high_risk_rate'])} | {fmt(x['false_positive_rate'])} | "
                 f"{fmt(x['inconclusive_rate'])} | {fmt(x['missed_as_low_risk'])} | {x['avg_challenges']} |")
    if "ablation" in charts:
        L += ["", f"![ablation]({charts['ablation']})"]
    L += ["", "## Per-challenge detection (passive + that one challenge)", "",
          "| Challenge | Detection rate | False-positive rate | n real | n swapped |", "|---|---|---|---|---|"]
    for ch, x in m["per_challenge"].items():
        L.append(f"| {ch} | {fmt(x['detection_rate'])} | {fmt(x['false_positive_rate'])} | {x['n_real']} | {x['n_swapped']} |")
    if "signals" in charts:
        L += ["", f"![signals]({charts['signals']})"]
    L += ["", "## Latency per clip", "", "| Clip | Local p50 (ms) | Local p95 (ms) | Local max (ms) | n |", "|---|---|---|---|---|"]
    for ch, x in m["latency_local_ms"].items():
        L.append(f"| {ch} | {x['p50']} | {x['p95']} | {x['max']} | {x['n']} |")
    if m["latency_lambda"]:
        L += ["", "Lambda (from `make smoke`):", "", "```json", json.dumps(m["latency_lambda"], indent=2), "```"]
    else:
        L += ["", "_Lambda latency: run `make smoke` after deploying to fill this in._"]
    if "latency" in charts:
        L += ["", f"![latency]({charts['latency']})"]
    L += ["", "## Agent behaviour", "", "Challenge orders the agent chose:", ""]
    L += [f"- `{k}`: {v} sessions" for k, v in m["agent_orders"].items()]
    L += ["", "Decision sources: " + ", ".join(f"{k}: {v}" for k, v in m["agent_policies"].items()), ""]
    L += ["## Unusable clips", ""]
    L += [f"- {k}: {v}" for k, v in m["unusable_reasons"].items()] or ["- none"]
    L += ["", f"## Failure cases ({len(m['failures'])})", ""]
    if not m["failures"]:
        L.append("None among public sessions in this run (agent strategy).")
    if m.get("failures_withheld"):
        L.append(f"{m['failures_withheld']} further failure(s) on non-public sessions are counted in the aggregate "
                 "numbers above but not shown (privacy: only sessions from the `" + str(m["public_prefix"]) + "` folders are shown).")
    for f in m["failures"]:
        L.append(f"### {f['participant']} · {f['label']} · {f['condition']} → {f['verdict']} (risk {f['risk']})")
        L += [f"- {t}" for t in f["reasons"]]
        if f["unusable"]:
            L.append(f"- unusable clips: {f['unusable']}")
        L += [f"![]({p})" for p in f["frames"]]
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
