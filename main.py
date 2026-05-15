
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

import yaml

from utils import ContinuousInstance, SingleTask, run_instance


DEFAULT_DATASET = "../../continuous-swe/data/final_filtered.jsonl"
_SEQ_META_KEYS = ("continuous_id", "repo", "base_commit", "docker_image", "source", "date_range")
_BUG_KEEP_KEYS = (
    "swebench_instance_id",
    "patch",
    "test_patch",
    "test_cmds",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def load_continuous_dataset(dataset_path: str | Path) -> list[ContinuousInstance]:
    with Path(dataset_path).open() as f:
        try:
            data = json.load(f)
            instances = data.get("instances", data) if isinstance(data, dict) else data
        except json.JSONDecodeError:
            f.seek(0)
            instances = [json.loads(line) for line in f if line.strip()]
    if not instances:
        raise ValueError(f"No instances found in {dataset_path}")
    return instances


def build_swebench_instance(sequence: ContinuousInstance, bug_fix: SingleTask) -> dict:
    return {
        "instance_id": bug_fix["swebench_instance_id"],
        "repo": sequence["repo"],
        "base_commit": sequence["base_commit"],
        "problem_statement": bug_fix["problem_statement"],
        "docker_image": sequence["docker_image"],
        "source": bug_fix.get("source", sequence.get("source", "")),
        "FAIL_TO_PASS": bug_fix.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": bug_fix.get("PASS_TO_PASS", []),
        "test_cmds": bug_fix.get("test_cmds", ""),
        "patch": bug_fix.get("patch", ""),
        "test_patch": bug_fix.get("test_patch", ""),
    }


def patch_path(output_dir: str | Path, instance_id: str) -> Path:
    return Path(output_dir) / instance_id / f"{instance_id}.patch"


def pred_path(output_dir: str | Path, instance_id: str) -> Path:
    return Path(output_dir) / instance_id / f"{instance_id}.pred"


def traj_path(output_dir: str | Path, instance_id: str) -> Path:
    return Path(output_dir) / instance_id / f"{instance_id}.traj"


def bug_is_complete(output_dir: str | Path, instance_id: str) -> bool:
    return (
        patch_path(output_dir, instance_id).exists()
        and pred_path(output_dir, instance_id).exists()
        and traj_path(output_dir, instance_id).exists()
    )


def read_patch_if_present(output_dir: str | Path, instance_id: str) -> str:
    path = patch_path(output_dir, instance_id)
    if not path.exists():
        return ""
    return path.read_text()


def collect_completed_prediction(
    sequence: ContinuousInstance,
    *,
    output_dir: str | Path,
    mode: str,
    model_name_or_path: str,
) -> dict | None:
    results = []
    for bug in sequence["bug_fixes"]:
        instance_id = bug["swebench_instance_id"]
        if not bug_is_complete(output_dir, instance_id):
            return None
        results.append(
            {
                "instance_id": instance_id,
                "model_patch": read_patch_if_present(output_dir, instance_id),
            }
        )
    return merge_agent_into_sequence(
        sequence,
        {"mode": mode, "results": results},
        model_name_or_path,
    )


def recollect_predictions(
    sequences: list[ContinuousInstance],
    *,
    output_dir: str | Path,
    output_file: str | Path,
    mode: str,
    model_name_or_path: str,
) -> None:
    completed = []
    for sequence in sequences:
        prediction = collect_completed_prediction(
            sequence,
            output_dir=output_dir,
            mode=mode,
            model_name_or_path=model_name_or_path,
        )
        if prediction is not None:
            completed.append(prediction)
    with Path(output_file).open("w") as f:
        for prediction in completed:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def prior_model_patches(output_dir: str | Path, bug_fixes: list[SingleTask], current_index: int) -> list[tuple[str, str]]:
    patches = []
    for prior_bug in bug_fixes[:current_index]:
        prior_id = prior_bug["swebench_instance_id"]
        patch_text = read_patch_if_present(output_dir, prior_id)
        if patch_text:
            patches.append((prior_id, patch_text))
    return patches


def prior_memory_trajectories(output_dir: str | Path, bug_fixes: list[SingleTask], current_index: int) -> list[Path]:
    if current_index == 0:
        return []
    path = traj_path(output_dir, bug_fixes[current_index - 1]["swebench_instance_id"])
    return [path] if path.exists() else []


def prior_ground_truth_patches(bug_fixes: list[SingleTask], current_index: int) -> list[tuple[str, str]]:
    patches = []
    for prior_bug in bug_fixes[:current_index]:
        prior_id = prior_bug["swebench_instance_id"]
        if test_patch := prior_bug.get("test_patch", ""):
            patches.append((f"test-{prior_id}", test_patch))
        if gold_patch := prior_bug.get("patch", ""):
            patches.append((f"gold-{prior_id}", gold_patch))
    return patches


def run_bug(
    sequence: ContinuousInstance,
    bug_fix: SingleTask,
    *,
    output_dir: str | Path,
    config_dir: str | Path,
    prior_patches: list[tuple[str, str]] | None = None,
    memory_trajectories: list[str | Path] | None = None,
) -> dict:
    instance = build_swebench_instance(sequence, bug_fix)
    instance_id = instance["instance_id"]
    if not bug_is_complete(output_dir, instance_id):
        run_instance(
            instance,
            config_dir=config_dir,
            output_dir=output_dir,
            prior_patches=prior_patches,
            memory_trajectories=memory_trajectories,
        )
    else:
        print(f"Skipping completed bug {instance_id}")

    return {
        "instance_id": instance_id,
        "model_patch": read_patch_if_present(output_dir, instance_id),
        "continuous_id": sequence["continuous_id"],
        "bug_order": bug_fix.get("order"),
    }


def run_sequential(
    continuous_instance: ContinuousInstance,
    *,
    output_dir: str | Path,
    config_dir: str | Path = "config/default.yaml",
    memory: bool = False,
) -> dict:
    bug_fixes = continuous_instance["bug_fixes"]
    results = []
    for i, bug_fix in enumerate(bug_fixes):
        result = run_bug(
            continuous_instance,
            bug_fix,
            output_dir=output_dir,
            config_dir=config_dir,
            prior_patches=prior_model_patches(output_dir, bug_fixes, i),
            memory_trajectories=prior_memory_trajectories(output_dir, bug_fixes, i) if memory else None,
        )
        result["mode"] = "sequential-memory" if memory else "sequential"
        results.append(result)
    return {
        "continuous_id": continuous_instance["continuous_id"],
        "mode": "sequential-memory" if memory else "sequential",
        "results": results,
    }


def run_sequential_memory(
    continuous_instance: ContinuousInstance,
    *,
    output_dir: str | Path,
    config_dir: str | Path = "config/default.yaml",
) -> dict:
    return run_sequential(continuous_instance, output_dir=output_dir, config_dir=config_dir, memory=True)


def run_interleaved(
    continuous_instance: ContinuousInstance,
    *,
    output_dir: str | Path,
    config_dir: str | Path = "config/default.yaml",
) -> dict:
    bug_fixes = continuous_instance["bug_fixes"]
    results = []
    for i, bug_fix in enumerate(bug_fixes):
        result = run_bug(
            continuous_instance,
            bug_fix,
            output_dir=output_dir,
            config_dir=config_dir,
            prior_patches=prior_ground_truth_patches(bug_fixes, i),
        )
        result["mode"] = "interleaved"
        results.append(result)
    return {"continuous_id": continuous_instance["continuous_id"], "mode": "interleaved", "results": results}


def merge_agent_into_sequence(sequence: ContinuousInstance, agent_result: dict, model_name_or_path: str) -> dict:
    patch_by_iid = {result["instance_id"]: result.get("model_patch", "") for result in agent_result.get("results", [])}
    out_bugs = []
    for bug in sequence.get("bug_fixes", []):
        instance_id = bug["swebench_instance_id"]
        merged = {key: bug[key] for key in _BUG_KEEP_KEYS if key in bug}
        merged["model_patch"] = patch_by_iid.get(instance_id, "")
        out_bugs.append(merged)

    out = {key: sequence[key] for key in _SEQ_META_KEYS if key in sequence}
    out["bug_fixes"] = out_bugs
    out["mode"] = agent_result.get("mode")
    out["model_name_or_path"] = model_name_or_path
    return out


def get_model_name(config_dir: str | Path) -> str:
    config = yaml.safe_load(Path(config_dir).read_text())
    return config["agent"]["model"]["name"]


def main(
    dataset_path: str | Path = DEFAULT_DATASET,
    mode: Literal["interleaved", "sequential", "sequential-memory"] = "sequential",
    *,
    config_dir: str | Path = "config/default.yaml",
    output_dir: str | Path | None = None,
    sequence_ids: list[str] | None = None,
    max_sequences: int | None = None,
) -> None:
    output_dir = Path(output_dir or f"logs/{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "predictions.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model_name_or_path = f"{get_model_name(config_dir)}_{mode}".replace("/", "__")

    dataset = load_continuous_dataset(dataset_path)
    if sequence_ids:
        allowed = set(sequence_ids)
        dataset = [sequence for sequence in dataset if sequence["continuous_id"] in allowed]
    if max_sequences is not None:
        dataset = dataset[:max_sequences]

    mode_fn = {
        "interleaved": run_interleaved,
        "sequential": run_sequential,
        "sequential-memory": run_sequential_memory,
    }[mode]

    for sequence in dataset:
        mode_fn(sequence, output_dir=output_dir, config_dir=config_dir)
        recollect_predictions(
            dataset,
            output_dir=output_dir,
            output_file=output_file,
            mode=mode,
            model_name_or_path=model_name_or_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SWE-agent on continuous SWE-bench sequences")
    parser.add_argument("--mode", choices=["interleaved", "sequential", "sequential-memory"], required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config_dir", default="config/default.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--sequence_ids", nargs="*", default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    args = parser.parse_args()

    main(
        dataset_path=os.path.abspath(args.dataset),
        mode=args.mode,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        sequence_ids=args.sequence_ids,
        max_sequences=args.max_sequences,
    )
