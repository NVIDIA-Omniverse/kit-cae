#!/usr/bin/env python3
"""
Kit-CAE Eval Runner

CLI tool for managing Kit-CAE agent skill evaluations.

Usage:
    python run_eval.py --list
    python run_eval.py --eval 01_import_inspect --prompt
    python run_eval.py --eval 01_import_inspect --prompt --harness claude-code --model opus-4.7
    python run_eval.py --eval 02_volume_render --validate --harness claude-code --model opus-4.7
    python run_eval.py --all --validate --harness claude-code --model opus-4.7
    python run_eval.py --results
"""

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")
VALIDATORS_DIR = os.path.join(SCRIPT_DIR, "validators")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Placeholder shown when harness/model aren't specified (e.g. --prompt inspection).
FALLBACK_RENDER_DIR = "data/renders"


def load_prompts():
    prompts = {}
    for path in sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.yaml"))):
        with open(path) as f:
            data = yaml.safe_load(f)
        key = os.path.splitext(os.path.basename(path))[0]
        data["_file"] = path
        data["_key"] = key
        prompts[key] = data
    return prompts


def resolve_eval_key(prompts, eval_key):
    if eval_key in prompts:
        return eval_key
    matches = [k for k in prompts if k.startswith(eval_key)]
    if len(matches) == 1:
        return matches[0]
    print(f"Error: eval '{eval_key}' not found. Use --list to see available evals.")
    sys.exit(1)


def combo_dir(harness, model):
    return os.path.join(RESULTS_DIR, harness, model)


def render_dir_for(harness, model):
    return os.path.join(combo_dir(harness, model), "renders")


def substitute_render_dir(text, render_dir):
    """Replace the literal token '{render_dir}' in text (no .format — braces elsewhere must survive)."""
    return text.replace("{render_dir}", render_dir)


def substituted_prompt(data, render_dir):
    """Return the prompt text and expected_outputs with {render_dir} substituted."""
    prompt = substitute_render_dir(data.get("prompt", ""), render_dir)
    outputs = []
    for out in data.get("expected_outputs", []) or []:
        sub = dict(out)
        if isinstance(sub.get("path"), str):
            sub["path"] = substitute_render_dir(sub["path"], render_dir)
        outputs.append(sub)
    return prompt, outputs


def list_evals(prompts):
    print(f"\nAvailable evals ({len(prompts)}):\n")
    print(f"  {'#':<4} {'Name':<25} {'Graded':<8} Description")
    print(f"  {'─'*4} {'─'*25} {'─'*8} {'─'*50}")
    for key, data in prompts.items():
        num = key.split("_")[0]
        graded = "Yes" if data.get("graded", True) else "No"
        desc = data.get("description", "").strip()[:50]
        print(f"  {num:<4} {data['name']:<25} {graded:<8} {desc}")
    print()


def show_prompt(prompts, eval_key, harness=None, model=None):
    eval_key = resolve_eval_key(prompts, eval_key)
    data = prompts[eval_key]

    if harness and model:
        render_dir = render_dir_for(harness, model)
    else:
        render_dir = FALLBACK_RENDER_DIR

    prompt_text, outputs = substituted_prompt(data, render_dir)

    print(f"\n{'='*70}")
    print(f"EVAL: {data['name']} ({eval_key})")
    print(f"Graded: {'Yes' if data.get('graded', True) else 'No (human review)'}")
    print(f"Kit: {data.get('kit_file', 'N/A')}")
    print(f"Data: {data.get('data_file', 'N/A')}")
    if harness and model:
        print(f"Harness/Model: {harness} / {model}")
        print(f"Render dir: {render_dir}")
    else:
        print(f"Render dir (placeholder): {render_dir}  "
              "— pass --harness and --model for the real path")
    print(f"{'='*70}")
    print(f"\n{prompt_text}")
    if outputs:
        print("expected_outputs:")
        for out in outputs:
            print(f"  - {out}")
    print(f"{'='*70}\n")


def run_validator(prompts, eval_key, harness, model):
    eval_key = resolve_eval_key(prompts, eval_key)
    data = prompts[eval_key]

    if not data.get("graded", True):
        print(f"Eval '{eval_key}' is not graded (creative eval). No validator to run.")
        return None

    validator = data.get("validator")
    if not validator:
        print(f"Eval '{eval_key}' has no validator configured.")
        return None

    validator_path = os.path.join(SCRIPT_DIR, validator)
    if not os.path.isfile(validator_path):
        print(f"Validator not found: {validator_path}")
        sys.exit(1)

    kit_file = data.get("kit_file", "omni.cae.kit")
    render_dir = render_dir_for(harness, model)
    os.makedirs(render_dir, exist_ok=True)

    cmd = [
        os.path.join(REPO_ROOT, "repo.sh"), "launch",
        "-n", kit_file, "--",
        "--exec", validator_path,
        "--no-window",
        "--/app/asyncRendering=false",
        "--/rtx/materialDb/syncLoads=true",
        "--/omni.kit.plugin/syncUsdLoads=true",
        "--/rtx/hydra/materialSyncLoads=true",
        "--/rtx-transient/resourcemanager/texturestreaming/async=false",
        "--/rtx-transient/resourcemanager/enableTextureStreaming=false",
        "--/exts/omni.kit.window.viewport/blockingGetViewportDrawable=true",
        "--/rtx-transient/dlssg/enabled=false",
        "--/persistent/app/viewport/defaults/fillViewport=false",
    ]

    if eval_key in ("02_volume_render", "04_multi_viz", "06_time_varying_video"):
        cmd.extend([
            '--/renderer/enabled=rtx',
            '--/rtx/rendermode=RaytracedLighting',
            '--/rtx/directLighting/sampledLighting/enabled=true',
        ])

    env = {**os.environ, "KIT_CAE_EVAL_RENDER_DIR": render_dir}

    print(f"\nRunning validator for {eval_key} ({harness}/{model})...")
    print(f"Render dir: {render_dir}")
    print(f"Command: {' '.join(cmd[:6])} ... (+ {len(cmd)-6} args)\n")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                cwd=REPO_ROOT, env=env)
        duration = time.time() - start_time
        output = result.stdout + result.stderr

        eval_result = None
        if "EVAL_RESULT_BEGIN" in output and "EVAL_RESULT_END" in output:
            start = output.index("EVAL_RESULT_BEGIN") + len("EVAL_RESULT_BEGIN")
            end = output.index("EVAL_RESULT_END")
            try:
                eval_result = json.loads(output[start:end].strip())
            except json.JSONDecodeError:
                print("Warning: could not parse EVAL_RESULT JSON")

        if eval_result:
            eval_result["duration_seconds"] = round(duration, 1)
            eval_result["harness"] = harness
            eval_result["model"] = model
            eval_result["date"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

            print(f"{'='*50}")
            print(f"Result: {'PASS' if eval_result['pass'] else 'FAIL'} — "
                  f"Score: {eval_result['score']}/100")
            print(f"Duration: {duration:.1f}s")
            print(f"{'='*50}")
            for check in eval_result.get("checks", []):
                status = "✓" if check["pass"] else "✗"
                print(f"  {status} {check['name']}: {check.get('detail', '')}")
            print()

            save_result(eval_key, eval_result, harness, model)
            update_combo_summary(harness, model)
            update_top_summary()
            return eval_result

        print("No EVAL_RESULT found in output.")
        if result.returncode != 0:
            print(f"Exit code: {result.returncode}")
            for line in output.strip().split("\n")[-20:]:
                print(f"  {line}")
        return None

    except subprocess.TimeoutExpired:
        print("Validator timed out after 600s")
        return None
    except Exception as e:
        print(f"Error running validator: {e}")
        return None


def save_result(eval_key, result, harness, model):
    """Write per-eval JSON under results/<harness>/<model>/<eval_key>.json (latest-wins)."""
    target_dir = combo_dir(harness, model)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{eval_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved: {path}")


def _load_combo_results(harness, model):
    """Return a list of per-eval results for a combo, sorted by eval key."""
    target_dir = combo_dir(harness, model)
    if not os.path.isdir(target_dir):
        return []
    results = []
    for path in sorted(glob.glob(os.path.join(target_dir, "*.json"))):
        if os.path.basename(path) == "summary.json":
            continue
        try:
            with open(path) as f:
                results.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return results


def update_combo_summary(harness, model):
    """Regenerate results/<harness>/<model>/summary.json from per-eval JSONs."""
    results = _load_combo_results(harness, model)
    if not results:
        return
    passed = sum(1 for r in results if r.get("pass"))
    total = len(results)
    scores = [r.get("score", 0) for r in results]
    avg_score = round(sum(scores) / total) if total else 0
    last_updated = max((r.get("date", "") for r in results), default="")

    summary = {
        "harness": harness,
        "model": model,
        "last_updated": last_updated or datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "passed": passed,
        "total": total,
        "avg_score": avg_score,
        "evals": [
            {
                "eval": r.get("eval", "?"),
                "pass": bool(r.get("pass")),
                "score": r.get("score", 0),
                "duration_seconds": r.get("duration_seconds"),
                "date": r.get("date"),
            }
            for r in results
        ],
    }
    path = os.path.join(combo_dir(harness, model), "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def _all_combo_summaries():
    """Walk results/*/*/summary.json and yield parsed dicts."""
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "*", "summary.json"))):
        try:
            with open(path) as f:
                yield json.load(f)
        except (OSError, json.JSONDecodeError):
            continue


def update_top_summary():
    """Regenerate results/SUMMARY.md from all per-combo summary.json files."""
    summaries = list(_all_combo_summaries())
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Kit-CAE Eval Results",
        "",
        f"_Auto-generated by `run_eval.py`. Last updated: {now}._",
        "",
        "## Matrix",
        "",
    ]

    if not summaries:
        lines.append("_No eval runs recorded yet._")
        lines.append("")
    else:
        lines.append("| Harness | Model | Passed | Avg Score | Last Run |")
        lines.append("|---------|-------|--------|-----------|----------|")
        for s in sorted(summaries, key=lambda x: (x.get("harness", ""), x.get("model", ""))):
            lines.append(
                f"| {s.get('harness', '?')} | {s.get('model', '?')} | "
                f"{s.get('passed', 0)}/{s.get('total', 0)} | "
                f"{s.get('avg_score', 0)} | "
                f"{(s.get('last_updated') or '')[:16].replace('T', ' ')} |"
            )
        lines.append("")
        lines.append("## Per-combination detail")
        lines.append("")
        for s in sorted(summaries, key=lambda x: (x.get("harness", ""), x.get("model", ""))):
            harness = s.get("harness", "?")
            model = s.get("model", "?")
            lines.append(
                f"### {harness} / {model} — {s.get('passed', 0)}/{s.get('total', 0)} passed "
                f"(avg {s.get('avg_score', 0)})"
            )
            lines.append("")
            lines.append("| Eval | Pass | Score | Duration | Date |")
            lines.append("|------|------|-------|----------|------|")
            for e in s.get("evals", []):
                status = "✓" if e.get("pass") else "✗"
                duration = e.get("duration_seconds")
                dur_str = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "—"
                date_str = (e.get("date") or "")[:10]
                lines.append(
                    f"| {e.get('eval', '?')} | {status} | {e.get('score', 0)} | "
                    f"{dur_str} | {date_str} |"
                )
            lines.append("")

    path = os.path.join(RESULTS_DIR, "SUMMARY.md")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def show_results():
    """Print the matrix and per-combo detail from saved summaries."""
    summaries = list(_all_combo_summaries())
    if not summaries:
        print("No results yet.")
        return

    print(f"\nResults ({len(summaries)} harness/model combinations):\n")
    print(f"  {'Harness':<15} {'Model':<20} {'Passed':<10} {'Avg Score':<10} Last Run")
    print(f"  {'─'*15} {'─'*20} {'─'*10} {'─'*10} {'─'*20}")
    for s in sorted(summaries, key=lambda x: (x.get("harness", ""), x.get("model", ""))):
        passed_str = f"{s.get('passed', 0)}/{s.get('total', 0)}"
        print(f"  {s.get('harness', '?'):<15} {s.get('model', '?'):<20} "
              f"{passed_str:<10} {s.get('avg_score', 0):<10} "
              f"{(s.get('last_updated') or '')[:16]}")
    print()


def require_harness_model(args):
    if not args.harness or not args.model:
        print("Error: --harness and --model are required when running validators.")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Kit-CAE Agent Skill Eval Runner")
    parser.add_argument("--list", action="store_true", help="List available evals")
    parser.add_argument("--eval", type=str, help="Eval key (e.g., 01_import_inspect)")
    parser.add_argument("--prompt", action="store_true", help="Show eval prompt")
    parser.add_argument("--validate", action="store_true", help="Run validator")
    parser.add_argument("--all", action="store_true", help="Run all graded evals")
    parser.add_argument("--harness", type=str, default=None,
                        help="Harness name (required for --validate; e.g. claude-code, openclaw, codex)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (required for --validate; e.g. opus-4.7, codex-5.4)")
    parser.add_argument("--results", action="store_true", help="Show results summary")

    args = parser.parse_args()
    prompts = load_prompts()

    if args.list:
        list_evals(prompts)
    elif args.results:
        show_results()
    elif args.all and args.validate:
        require_harness_model(args)
        graded = {k: v for k, v in prompts.items() if v.get("graded", True)}
        print(f"\nRunning {len(graded)} graded evals for {args.harness}/{args.model}...\n")
        results = []
        for key in graded:
            result = run_validator(prompts, key, args.harness, args.model)
            if result:
                results.append(result)
        passed = sum(1 for r in results if r.get("pass"))
        print(f"\n{'='*50}")
        print(f"TOTAL: {passed}/{len(results)} passed")
        print(f"{'='*50}\n")
    elif args.eval:
        if args.prompt:
            show_prompt(prompts, args.eval, args.harness, args.model)
        elif args.validate:
            require_harness_model(args)
            run_validator(prompts, args.eval, args.harness, args.model)
        else:
            show_prompt(prompts, args.eval, args.harness, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
