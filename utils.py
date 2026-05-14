from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, TypedDict

import yaml
from swerex.deployment.config import DockerDeploymentConfig

from sweagent.agent.agents import DefaultAgentConfig
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.environment.repo import PreExistingRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig
from sweagent.run.hooks.abstract import RunHook
from sweagent.run.run_single import RunSingle, RunSingleConfig
from sweagent.utils.log import get_logger


class SWEBenchInstance(TypedDict):
    instance_id: str
    source: str
    repo: str
    base_commit: str
    commit_hash: str
    problem_statement: str
    date: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    test_cmds: str
    patch: str
    test_patch: str
    docker_image: str


class SingleTask(TypedDict):
    order: int
    swebench_instance_id: str
    commit_hash: str
    problem_statement: str
    date: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    test_cmds: str
    patch: str
    test_patch: str


class ContinuousInstance(TypedDict):
    continuous_id: str
    repo: str
    base_commit: str
    source: str
    date_range: str
    docker_image: str
    swebench_instance_ids: list[str]
    bug_fixes: list[SingleTask]


class ApplyPriorPatchesHook(RunHook):
    def __init__(self, patches: list[tuple[str, str]]):
        self.patches = [(name, patch) for name, patch in patches if patch.strip()]
        self.logger = get_logger("swea-prior-patches", emoji="🧩")

    def on_instance_start(self, *, index: int, env, problem_statement):
        if not self.patches:
            return
        if env.repo is None:
            raise RuntimeError("Cannot apply prior patches without a repository")

        repo_dir = f"/{env.repo.repo_name}"
        env.communicate(
            " && ".join(
                [
                    f"cd {shlex.quote(repo_dir)}",
                    f"git config --global --add safe.directory {shlex.quote(repo_dir)}",
                    "git config user.email 'swe-agent@local'",
                    "git config user.name 'SWE-agent'",
                ]
            ),
            check="raise",
        )

        for patch_index, (patch_name, patch_text) in enumerate(self.patches, start=1):
            patch_path = f"/tmp/swe-agent-prior-{patch_index}.patch"
            env.write_file(patch_path, patch_text)
            apply_command = (
                f"cd {shlex.quote(repo_dir)} && "
                f"if git apply -v {patch_path}; then "
                "echo __SWE_PRIOR_PATCH_APPLIED__; "
                f"elif git apply --reverse --check {patch_path}; then "
                "echo __SWE_PRIOR_PATCH_ALREADY_APPLIED__; "
                f"elif git apply --3way -v {patch_path}; then "
                "echo __SWE_PRIOR_PATCH_APPLIED_3WAY__; "
                "else "
                "git reset --hard HEAD && git clean -fd && "
                "echo __SWE_PRIOR_PATCH_FAILED__; "
                "fi"
            )
            output = env.communicate(apply_command, timeout=120)
            if "__SWE_PRIOR_PATCH_FAILED__" in output:
                self.logger.warning("Prior patch failed for %s; continuing anyway", patch_name)
                continue

            commit_command = " && ".join(
                [
                    f"cd {shlex.quote(repo_dir)}",
                    "git add -A",
                    f"git commit -m {shlex.quote(f'prior-patch: {patch_name}')} --allow-empty",
                ]
            )
            env.communicate(commit_command, check="raise", timeout=120, error_msg=f"Failed to commit prior patch {patch_name}")


def _get_repo_name(instance: dict[str, Any]) -> str:
    image_name = instance["docker_image"].removeprefix("docker.io/")
    if image_name.startswith("jefzda/sweap-images"):
        return "app"
    if image_name.startswith(("swerebench/", "swerebenchv2/")):
        return instance["repo"].rstrip("/").split("/")[-1]
    return "testbed"


def run_instance(
    instance: dict[str, Any],
    config_dir: str | Path = "config/default.yaml",
    output_dir: str | Path = "./",
    prior_patches: list[tuple[str, str]] | None = None,
    memory_trajectories: list[str | Path] | None = None,
) -> None:
    instance_id: str = instance["instance_id"]
    base_commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    image_name: str = instance["docker_image"]

    config_path = Path(config_dir)
    config_dict = yaml.safe_load(config_path.read_text())
    agent_config_dict = config_dict["agent"]
    if agent_config_dict["model"].get("temperature") is None:
        agent_config_dict["model"].pop("temperature", None)
    agent_config = DefaultAgentConfig.model_validate(agent_config_dict)
    if memory_trajectories:
        agent_config.templates.demonstrations = [
            *agent_config.templates.demonstrations,
            *[Path(path).resolve() for path in memory_trajectories],
        ]
        agent_config.templates.put_demos_in_history = True
    env_config = EnvironmentConfig(
        deployment=DockerDeploymentConfig(image=image_name, python_standalone_dir="/root"),
        repo=PreExistingRepoConfig(repo_name=_get_repo_name(instance), base_commit=base_commit),
    )
    problem_statement_config = TextProblemStatement(
        text=problem_statement,
        id=instance_id,
    )

    run_config = RunSingleConfig(
        agent=agent_config,
        env=env_config,
        problem_statement=problem_statement_config,
        output_dir=Path(output_dir),
    )
    run_config._config_files = [config_path]  # type: ignore[attr-defined]
    run = RunSingle.from_config(run_config)
    if prior_patches:
        run.add_hook(ApplyPriorPatchesHook(prior_patches))
    run.run()

if __name__ == "__main__":
    import json
    with open("swe_debug.jsonl") as f:
        l=[json.loads(i) for i in f]
    instance = l[0]
    id_docker_compatible = instance["instance_id"].replace("__", "_1776_")
    instance["docker_image"] = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    run_instance(instance, "config/default.yaml", "./logs/debug/")
