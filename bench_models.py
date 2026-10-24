"""Model benchmark: latency + answer quality across Jarvis's pool.

Usage:
  ./jarvis/Scripts/python bench_models.py            # full run (~10 min)
  ./jarvis/Scripts/python bench_models.py --quick    # 2 tasks x all models
  ./jarvis/Scripts/python bench_models.py --models gemini-3.5-flash-lite,laguna

Measures per (model, task): time-to-first-token, total seconds, output
tokens. Then judges each answer 1-5 with gemini-3.5-flash. Writes
bench_results.md (tables) + bench_results.json (full answers).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import config
import router

TASKS = [
    ("factual", "Why is the sky blue? Answer in one sentence."),
    ("coding", "Write a Python function fib(n) that returns a list of the "
               "first n Fibonacci numbers. Code with one short comment."),
    ("reasoning", "A bat and ball cost $1.10 in total. The bat costs $1.00 "
                  "more than the ball. How much does the ball cost? Answer "
                  "in one short sentence."),
    ("summarize", "Summarize this in one sentence: Octopuses have three "
                  "hearts, blue blood, and the ability to change both color "
                  "and texture in a fraction of a second. Two-thirds of their "
                  "neurons are in their arms, allowing each arm to taste, "
                  "touch, and make decisions semi-independently. They can "
                  "unscrew jars from the inside, escape through holes barely "
                  "larger than their beaks, and some species can survive "
                  "briefly outside water by absorbing oxygen through skin."),
    ("voice-style", "Explain recursion like I'm five, in exactly two short "
                    "sentences."),
]

# (label, provider, model, pace_seconds_between_tasks, max_tokens)
# Reasoning models get a big budget: their trace eats tokens before the answer.
POOL = [
    ("flash-lite", "gemini", "gemini-3.5-flash-lite", 3, 400),
    ("flash", "gemini", "gemini-3.5-flash", 3, 400),
    ("laguna-s-2.1:free", "openrouter", "poolside/laguna-s-2.1:free", 5, 1500),
    ("nemotron-lightning:free", "openrouter",
     "nvidia/nemotron-3.5-lightning:free", 5, 1500),
    ("nemotron-ultra:free", "openrouter",
     "nvidia/nemotron-3-ultra-550b-a55b:free", 5, 1500),
]

JUDGE_MODEL = "gemini-3.5-flash"
JUDGE_PROMPT = (
    "Score this assistant answer 1-5 (5=best) on: correctness, following the "
    "format/length instruction, clarity for spoken delivery. Reply with ONLY "
    "'N: one-line reason'.\nTask: {task}\nAnswer: {answer}"
)


def run_once(client, model: str, prompt: str, max_tokens: int = 400):
    """Stream one completion. Returns dict with text/ttft/total/usage/error."""
    t0 = time.time()
    ttft, text, reasoning = None, "", ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=max_tokens, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = delta.content or ""
            think = getattr(delta, "reasoning", "") or ""
            if think == piece:
                think = ""  # some providers mirror content into reasoning
            if (piece or think) and ttft is None:
                ttft = time.time() - t0
            text += piece
            reasoning += think
        try:
            stream.close()
        except Exception:
            pass
        total = time.time() - t0
        return {"text": text.strip(), "ttft": round(ttft or total, 2),
                "total": round(total, 2),
                "out_tokens": max(1, len(text) // 4),
                "reasoning_tokens": max(0, len(reasoning) // 4),
                "error": ""}
    except Exception as e:
        return {"text": "", "ttft": None, "total": round(time.time() - t0, 2),
                "out_tokens": 0, "error": str(e)[:200]}


def judge(client, task_name: str, task_prompt: str, answer: str) -> str:
    if not answer:
        return "0: no answer (call failed or reasoning-only)"
    last = "?"
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    task=f"{task_name}: {task_prompt}", answer=answer[:1500])}],
                temperature=0.0, max_tokens=60)
            out = (r.choices[0].message.content or "").strip()[:120]
            if out:
                return out
            last = "empty judgment"
        except Exception as e:
            last = str(e)[:100]
            if "429" in last:
                print("      judge 429, backing off 60s...")
                time.sleep(60)
            else:
                time.sleep(5)
    return f"?: judge failed ({last})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--judge-only", action="store_true",
                    help="re-score bench_results.json with the judge; no model calls")
    args = ap.parse_args(argv)

    tasks = TASKS[:2] if args.quick else TASKS
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    pool = [p for p in POOL if not wanted or p[0] in wanted or p[2] in wanted]
    if not pool:
        print("no models match --models; options:",
              ", ".join(p[0] for p in POOL))
        return 2

    judge_client = router.make_client("gemini")
    if args.judge_only:
        with open("bench_results.json", encoding="utf-8") as f:
            saved = json.load(f)
        results = saved["results"]
        tasks = [t for t in TASKS if t[0] in saved.get("tasks", [])]
        for label, r in results.items():
            if "tasks" not in r:
                continue
            for tname, prompt in tasks:
                tr = r["tasks"].get(tname)
                if not tr:
                    continue
                print(f"judging {label}/{tname}...", flush=True)
                tr["score"] = judge(judge_client, tname, prompt,
                                    tr.get("text", ""))
                time.sleep(8)  # judge pacing: flash quota is tight
        with open("bench_results.json", "w", encoding="utf-8") as f:
            json.dump({"tasks": [t[0] for t in tasks], "results": results},
                      f, indent=1)
        _write_md(tasks, results)
        print("\nre-scored bench_results.md + bench_results.json")
        return 0

    results: dict = {}
    for label, provider, model, pace, maxtok in pool:
        print(f"\n=== {label} ({model}) ===")
        try:
            client = router.make_client(provider)
        except Exception as e:
            print(f"  client failed: {e}")
            results[label] = {"model": model, "client_error": str(e)}
            continue
        results[label] = {"model": model, "tasks": {}}
        for i, (tname, prompt) in enumerate(tasks):
            print(f"  [{tname}]...", flush=True)
            for attempt in (1, 2):
                res = run_once(client, model, prompt, max_tokens=maxtok)
                if not res["error"] or "429" not in res["error"] or attempt == 2:
                    break
                print("    429, backing off 30s...")
                time.sleep(30)
            res["score"] = judge(judge_client, tname, prompt, res["text"])
            results[label]["tasks"][tname] = res
            print(f"    ttft={res['ttft']}s total={res['total']}s "
                  f"tok~{res['out_tokens']}+{res.get('reasoning_tokens', 0)}r "
                  f"score={res['score']} "
                  f"{('[ERR] ' + res['error'][:80]) if res['error'] else ''}")
            time.sleep(pace)
    _write_md(tasks, results)
    print("\nwrote bench_results.md + bench_results.json")
    return 0


def _write_md(tasks, results):
    with open("bench_results.json", "w", encoding="utf-8") as f:
        json.dump({"tasks": [t[0] for t in tasks], "results": results}, f,
                  indent=1)
    lines = ["# Bench results", "",
             "| model | " + " | ".join(t[0] for t in tasks) + " | avg total |",
             "| --- | " + " | ".join("---" for _ in tasks) + " | --- |"]
    for label, r in results.items():
        if "tasks" not in r:
            lines.append(f"| {label} | CLIENT FAILED | | |")
            continue
        cells, tot, n = [], 0.0, 0
        for tname, _ in tasks:
            tr = r["tasks"].get(tname, {})
            cells.append(f"{tr.get('score', '?').split(':')[0]} "
                         f"({tr.get('total', '?')}s)")
            if tr.get("total"):
                tot += tr["total"]
                n += 1
        lines.append(f"| {label} | " + " | ".join(cells) +
                     f" | {round(tot / max(n, 1), 1)}s |")
    lines += ["", "## Answers"]
    for label, r in results.items():
        lines.append(f"\n### {label} ({r.get('model', '?')})")
        for tname, _ in (tasks if "tasks" in r else []):
            tr = r["tasks"][tname]
            lines.append(f"\n**{tname}** ttft={tr.get('ttft')}s "
                         f"total={tr.get('total')}s score={tr.get('score')}")
            lines.append(f"> {(tr.get('text') or '[ERR] ' + tr.get('error', ''))[:600]}")
    with open("bench_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote bench_results.md + bench_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
