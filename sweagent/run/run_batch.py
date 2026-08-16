"""
Run on a batch of instances/issues, e.g., SWE-bench.

[cyan][bold]=== BASIC OPTIONS ===[/bold][/cyan]

  -h --help           Show help text and exit
  --help_option      Print specific help text and exit

[cyan][bold]=== EXAMPLES ===[/bold][/cyan]

Basic usage: Run over a [bold][cyan]SWE-bench lite[/bold][/cyan][green]:

sweagent run-batch \\
    --instances.type swe_bench \\ # configure instances
    --instances.subset lite \\
    --instances.split dev  \\
    --instances.slice :50 \\     # first 50 instances
    --instances.shuffle=True \\  # shuffle instances (with fixed seed)
    --config config/default.yaml \\
    --agent.model.name gpt-4o  # configure model
[/green]

[cyan][bold]=== LOADING INSTANCES ===[/bold][/cyan]

[cyan][bold]From a file[/bold][/cyan] [green]--instances.type file --instances.path /path/to/file[/green].
[cyan][bold]From huggingface[/bold][/cyan] [green]--instances.type huggingface --instances.dataset_name=SWE_Bench_lite --instances.split=dev[/green].

All instance specifications support the [green]filter[/green], [green]slice[/green], and [green]shuffle[/green] options.
With [green]filter[/green], you can select specific instances, e.g., [green]--instances.filter='instance_id_1|instance_id_2'[/green].
"""

import json
import logging
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.live import Live

from sweagent.agent.agents import AbstractAgent, AgentConfig, get_agent_from_config
from sweagent.agent.hooks.status import SetStatusAgentHook
from sweagent.environment.hooks.status import SetStatusEnvironmentHook
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import ModelConfigurationError, TotalCostLimitExceededError
from sweagent.run._progress import RunBatchProgressManager
from sweagent.run.batch_instances import BatchInstance, BatchInstanceSourceConfig, SWEBenchInstances
from sweagent.run.common import BasicCLI, ConfigHelper
from sweagent.run.eval import evaluate
from sweagent.run.hooks.abstract import CombinedRunHooks, RunHook
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.run_single import RunSingleConfig
from sweagent.utils.config import load_environment_variables
from sweagent.utils.log import (
    add_file_handler,
    add_logger_names_to_stream_handlers,
    get_logger,
    register_thread_name,
    remove_file_handler,
    set_stream_handler_levels,
)
from swerex.deployment.hooks.status import SetStatusDeploymentHook

EVALUATION_TIMEOUT = 1800
EVALUATION_NAMESPACE = "swebench"


def _submission_from_info(info: object) -> str | None:
    if not isinstance(info, dict):
        return None
    submission = info.get("submission")
    return submission if isinstance(submission, str) else None


def _submission_from_trajectory(trajectory_path: Path) -> str | None:
    try:
        data = json.loads(trajectory_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    submission = _submission_from_info(data.get("info"))
    if submission is not None:
        return submission

    attempts = data.get("attempts")
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if not isinstance(attempt, dict):
                continue
            submission = _submission_from_info(attempt.get("info"))
            if submission is not None:
                return submission
    return None


def _submission_from_agent(agent: object) -> str | None:
    candidates = [agent, getattr(agent, "_agent", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        submission = _submission_from_info(getattr(candidate, "info", None))
        if submission is not None:
            return submission
    return None


def _write_trajectory_success(trajectory_path: Path, success: bool) -> None:
    try:
        data = json.loads(trajectory_path.read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    info = data.get("info")
    if not isinstance(info, dict):
        info = {}
        data["info"] = info
    info["success"] = success
    trajectory_path.write_text(json.dumps(data, indent=2))


def _get_primary_model_config(agent_config: AgentConfig):
    if hasattr(agent_config, "model"):
        return agent_config.model
    return agent_config.agent_configs[0].model


class RunBatchConfig(BaseSettings, cli_implicit_flags=False):
    instances: BatchInstanceSourceConfig = Field(description="Instances to run.")
    agent: AgentConfig = Field(description="Agent options.")
    output_dir: Path = Field(default=Path("DEFAULT"), description="Output directory.")
    epoch: int = Field(ge=0, description="Training epoch used in the output directory.")
    raise_exceptions: bool = False
    """Raise exceptions instead of skipping instances."""
    redo_existing: bool = False
    """Do not skip instances that already have a trajectory."""
    env_var_path: Path | None = None
    """Path to a .env file to load environment variables from."""
    num_workers: int = Field(default=1, ge=1)
    """Maximum number of sampled agents allowed to execute their rollout concurrently."""
    random_delay_multiplier: float = 0.3
    """We will wait for a random amount of time between 0 and `random_delay_multiplier`
    times the number of workers at the start of each instance. This is to avoid any
    potential race condition or issues with bottlenecks, e.g., when running on a platform
    with few CPUs that cannot handle the startup of all containers in time.
    """
    progress_bar: bool = True
    """Whether to show a progress bar. Progress bar is never shown for human models.
    Progress bar is always shown for multi-worker runs.
    """

    # pydantic config
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SWE_AGENT_")

    def get_run_type(self) -> Literal["train", "test"]:
        config_files = getattr(self, "_config_files", [])
        if not config_files:
            msg = "The default log layout requires --config config/train.yaml or --config config/test.yaml."
            raise ValueError(msg)
        run_type = Path(config_files[-1]).stem
        if run_type not in {"train", "test"}:
            msg = f"Expected the final config file to be train.yaml or test.yaml, got {config_files[-1]}."
            raise ValueError(msg)
        return run_type

    def set_default_output_dir(self) -> None:
        if self.output_dir != Path("DEFAULT"):
            return
        run_type = self.get_run_type()
        model_name = _get_primary_model_config(self.agent).name
        model_path = Path(model_name)
        if model_path.is_absolute() or ".." in model_path.parts:
            msg = f"Model name must be a safe relative path component, got {model_name!r}."
            raise ValueError(msg)
        self.output_dir = Path("logs") / model_path / str(self.epoch) / run_type

    @model_validator(mode="after")
    def disable_evaluation_artifacts(self) -> Self:
        if isinstance(self.instances, SWEBenchInstances) and self.instances.evaluate:
            msg = "instances.evaluate is not supported because per-sample evaluation is always enabled."
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class SampledBatchInstance:
    instance: BatchInstance
    sample_id: int

    @property
    def instance_id(self) -> str:
        return self.instance.problem_statement.id


class SampleAgent:
    """Own the agent and environment lifecycle for one sampled instance."""

    sample_instance: SampledBatchInstance
    agent: AbstractAgent
    env: SWEEnv

    def __init__(self, sample_instance: SampledBatchInstance, run_batch: "RunBatch"):
        self.sample_instance = sample_instance
        self.run_batch = run_batch
        self.run_id = run_batch.get_run_id(sample_instance)
        self.output_dir = run_batch.get_instance_output_dir(sample_instance.instance_id, sample_instance.sample_id)
        self.patch_path = self.output_dir / f"{sample_instance.instance_id}.patch"
        self.trajectory_path = self.output_dir / f"{sample_instance.instance_id}.traj"
        self._completed = False
        self._closed = False
        self._skipped = False
        self._startup_error: BaseException | None = None

        self._prepare_sample()
        if self._skipped:
            return

        try:
            self._start_agent_and_environment()
        except BaseException as e:
            # Constructor failures must pass through rollout's original exception
            # handling and finalization path (evaluation, status updates, and logs).
            self._startup_error = e

    def _prepare_sample(self) -> None:
        run_batch = self.run_batch
        sampled_instance = self.sample_instance
        run_batch.logger.info("Running sample %s", self.run_id)
        register_thread_name(self.run_id)
        run_batch._add_instance_log_file_handler(sampled_instance, multi_worker=run_batch._uses_parallel_threads)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.patch_path.exists():
            self.patch_path.write_text("")

        # Add randomness to avoid potential race conditions or a thundering herd.
        max_parallel_samples = run_batch._instance_workers * run_batch._samples
        if run_batch._progress_manager.n_completed < max_parallel_samples:
            time.sleep(random.random() * run_batch._random_delay_multiplier * (max_parallel_samples - 1))

        run_batch._progress_manager.on_instance_start(self.run_id)
        if previous_exit_status := run_batch.should_skip(sampled_instance):
            run_batch._progress_manager.on_instance_end(self.run_id, exit_status=f"skipped ({previous_exit_status})")
            run_batch._remove_instance_log_file_handler(self.run_id)
            self._skipped = True
            return

        self.trajectory_path.write_text("")

    def _start_agent_and_environment(self) -> None:
        """Initialize this sample's agent and start its dedicated SWEEnv."""
        instance = self.sample_instance.instance.model_copy(deep=True)
        self.instance = instance
        self.patch_path.write_text("")

        agent_config = self.run_batch.agent_config.model_copy(deep=True)
        agent_config.name = self.run_id.replace("/", "-")
        self.agent = get_agent_from_config(agent_config)
        single_run_replay_config = RunSingleConfig(
            agent=agent_config,
            problem_statement=instance.problem_statement,
            env=instance.env,
        )
        self.agent.replay_config = single_run_replay_config
        self.agent.add_hook(SetStatusAgentHook(self.run_id, self.run_batch._progress_manager.update_instance_status))

        self.run_batch._progress_manager.update_instance_status(self.run_id, "Starting environment")
        instance.env.name = self.run_id.replace("/", "-")
        self.env = SWEEnv.from_config(instance.env)
        self.env.add_hook(
            SetStatusEnvironmentHook(self.run_id, self.run_batch._progress_manager.update_instance_status)
        )
        self.env.deployment.add_hook(
            SetStatusDeploymentHook(self.run_id, self.run_batch._progress_manager.update_instance_status)
        )
        self.env.start()

    def _evaluate_sample(self) -> bool:
        sampled_instance = self.sample_instance
        eval_dir = self.output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        success = False

        swebench_instance = sampled_instance.instance.swebench_instance
        if swebench_instance is None:
            self.run_batch.logger.warning(
                "Cannot evaluate %s: original SWE-bench instance data is unavailable",
                self.run_id,
            )
        else:
            try:
                prediction = {
                    "model_name_or_path": _get_primary_model_config(self.run_batch.agent_config).name,
                    "instance_id": sampled_instance.instance_id,
                    "model_patch": (self.patch_path.read_text() if self.patch_path.exists() else ""),
                }
                eval_result = evaluate(
                    prediction=prediction,
                    instance=swebench_instance,
                    run_id=self.run_id,
                    log_dir=eval_dir,
                    timeout=EVALUATION_TIMEOUT,
                    namespace=EVALUATION_NAMESPACE,
                )
                if eval_result is None:
                    self.run_batch.logger.warning(
                        "Evaluation returned None for %s; recording success=False",
                        self.run_id,
                    )
                else:
                    success = eval_result
            except Exception:
                self.run_batch.logger.warning(
                    "Evaluation failed for %s; recording success=False",
                    self.run_id,
                    exc_info=True,
                )

        _write_trajectory_success(self.trajectory_path, success)
        return success

    def _recover_submission(self) -> None:
        if not hasattr(self, "agent"):
            return
        submission = _submission_from_agent(self.agent)
        if submission is None:
            submission = _submission_from_trajectory(self.trajectory_path)
        if submission is not None:
            try:
                self.patch_path.write_text(submission)
            except OSError:
                self.run_batch.logger.error(
                    "Failed to save recovered patch to %s",
                    self.patch_path,
                    exc_info=True,
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "env"):
            self.env.close()

    def rollout(self) -> SampledBatchInstance:
        if self._skipped:
            return self.sample_instance

        try:
            try:
                if self._startup_error is not None:
                    raise self._startup_error
                self.run_batch._chooks.on_instance_start(
                    index=self.sample_instance.sample_id,
                    env=self.env,
                    problem_statement=self.instance.problem_statement,
                )
                with self.run_batch._agent_run_semaphore:
                    result = self.agent.run(
                        problem_statement=self.instance.problem_statement,
                        env=self.env,
                        output_dir=self.output_dir,
                    )
                self.run_batch._chooks.on_instance_completed(result=result)
                self._completed = True
            except BaseException:
                # The outer handling below owns control flow, but the exception must
                # also be present in the per-agent log as in the original runner.
                if hasattr(self, "agent"):
                    self.agent.logger.error(traceback.format_exc())
                raise
            finally:
                if not self._completed:
                    self._recover_submission()
                self.close()
        except KeyboardInterrupt:
            raise _BreakLoop
        except (
            SystemExit,
            ModelConfigurationError,
            TotalCostLimitExceededError,
        ) as e:
            if self.run_batch._raise_exceptions:
                raise
            self.run_batch.logger.critical(f"❌ Exiting because {e.__class__.__name__} was called")
            raise _BreakLoop
        except Exception as e:
            self.run_batch.logger.error(traceback.format_exc())
            self.run_batch.logger.error(f"❌ Failed on {self.run_id}: {e}")
            self.run_batch._progress_manager.on_uncaught_exception(self.run_id, e)
            if self.run_batch._raise_exceptions:
                raise
        else:
            self.run_batch._progress_manager.on_instance_end(
                self.run_id,
                exit_status=result.info.get("exit_status", "unknown_exit"),
            )
        finally:
            self._evaluate_sample()
            self.run_batch._progress_manager.update_exit_status_table()
            self.run_batch._remove_instance_log_file_handler(self.run_id)
        return self.sample_instance

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors must not emit unraisable exceptions during interpreter
            # shutdown; rollout performs the normal deterministic cleanup.
            pass


class _BreakLoop(Exception):
    """Used for internal control flow"""


class RunBatch:
    def __init__(
        self,
        instances: list[BatchInstance],
        agent_config: AgentConfig,
        *,
        output_dir: Path = Path("."),
        hooks: list[RunHook] | None = None,
        raise_exceptions: bool = False,
        redo_existing: bool = False,
        num_workers: int = 1,
        progress_bar: bool = True,
        random_delay_multiplier: float = 0.3,
        epoch: int = 0,
        run_type: Literal["train", "test"] = "train",
    ):
        """Note: When initializing this class, make sure to add the hooks that are required by your actions.
        See `from_config` for an example.

        Args:
            hooks: If not specified, the default hooks will be used.
            num_workers: Maximum number of sampled agents allowed to execute their rollout
                concurrently. Default is 1.
            progress_bar: Whether to show a progress bar. Progress bar is never shown for human models.
                Progress bar is always shown for multi-worker runs.
            random_delay_multiplier: We will wait for a random amount of time between 0 and `random_delay_multiplier`
                times the number of workers at the start of each instance. This is to avoid any
                potential race conditions.
        """
        self.agent_config = agent_config
        self.logger = get_logger("swea-run", emoji="🏃")
        self.instances = instances
        self.output_dir = output_dir
        self.epoch = epoch
        self.run_type = run_type
        samples = _get_primary_model_config(agent_config).samples
        self.sampled_instances: list[list[SampledBatchInstance]] = [
            [SampledBatchInstance(instance=instance, sample_id=sample_id) for sample_id in range(samples)]
            for instance in instances
        ]
        if num_workers < 1:
            msg = "num_workers must be at least 1"
            raise ValueError(msg)
        if self._model_id in ["human", "human_thought"] and (num_workers > 1 or samples > 1):
            msg = "Cannot run human models with parallel workers or samples"
            raise ValueError(msg)
        self._raise_exceptions = raise_exceptions
        self._chooks = CombinedRunHooks()
        self._redo_existing = redo_existing
        self._num_workers = num_workers
        self._samples = samples
        self._instance_workers = num_workers // samples + 3
        self._agent_run_semaphore = threading.Semaphore(num_workers)
        self._uses_parallel_threads = samples > 1 or (
            self._model_id not in ["human", "human_thought"]
            and self._instance_workers > 1
            and len(self.sampled_instances) > 1
        )
        for hook in hooks or [SaveApplyPatchHook(show_success_message=False)]:
            self.add_hook(hook)
        self._progress_manager = RunBatchProgressManager(
            num_instances=sum(len(sample_group) for sample_group in self.sampled_instances)
        )
        self._show_progress_bar = progress_bar
        self._random_delay_multiplier = random_delay_multiplier

    @property
    def _model_id(self) -> str:
        try:
            return _get_primary_model_config(self.agent_config).id
        except AttributeError:
            return "unknown"

    def get_instance_output_dir(self, instance_id: str, sample_id: int) -> Path:
        return self.output_dir / instance_id / str(sample_id)

    def get_run_id(self, sampled_instance: SampledBatchInstance) -> str:
        return f"{self.epoch}_{self.run_type}_{sampled_instance.instance_id}_{sampled_instance.sample_id}"

    @classmethod
    def from_config(cls, config: RunBatchConfig) -> Self:
        load_environment_variables(config.env_var_path)
        config.set_default_output_dir()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        logger = get_logger("run", emoji="🏃")
        logger.debug("Loading instances from %s", f"{config.instances!r}")
        instances = config.instances.get_instance_configs()
        logger.info("Loaded %d instances", len(instances))
        if not instances:
            msg = (
                "No instances to run. Here are a few things to check:\n"
                "- With huggingface data: Check that you have the right split (test or dev)\n"
                "- Check your filter does not exclude all instances (check the info log messages)"
            )
            raise ValueError(msg)
        logger.debug("The first instance is %s", f"{instances[0]!r}")
        rb = cls(
            instances=instances,
            agent_config=config.agent,
            output_dir=config.output_dir,
            raise_exceptions=config.raise_exceptions,
            redo_existing=config.redo_existing,
            num_workers=config.num_workers,
            progress_bar=config.progress_bar,
            random_delay_multiplier=config.random_delay_multiplier,
            epoch=config.epoch,
            run_type=config.get_run_type(),
        )
        logger.info(
            "Expanded %d instances into %d runs using %d sample(s) per instance",
            len(instances),
            sum(len(sample_group) for sample_group in rb.sampled_instances),
            _get_primary_model_config(config.agent).samples,
        )
        return rb

    def add_hook(self, hook: RunHook) -> None:
        hook.on_init(run=self)
        self._chooks.add_hook(hook)

    def main(self) -> None:
        self.logger.info("Starting run. Find output files at %s", self.output_dir)
        self._chooks.on_start()

        if self._model_id in ["human", "human_thought"] or self._instance_workers <= 1:
            self.main_single_worker()
        else:
            self.main_multi_worker()

        self._chooks.on_end()

    def main_single_worker(self) -> None:
        with ExitStack() as stack:
            # Conditionally add progress bar
            if self._model_id not in ["human", "human_thought"] and self._show_progress_bar:
                stack.enter_context(Live(self._progress_manager.render_group))
            for sampled_instances in self.sampled_instances:
                try:
                    self.run_instance(sampled_instances)
                except _BreakLoop:
                    self.logger.info("Stopping loop over instances")
                    break

    def main_multi_worker(self) -> None:
        add_logger_names_to_stream_handlers()
        # Set all stream handlers to WARNING and set everything where we want to have
        # more verbosity explicitly
        set_stream_handler_levels(logging.WARNING)
        self.logger.setLevel(logging.TRACE)  # type: ignore

        with Live(self._progress_manager.render_group):
            with ThreadPoolExecutor(max_workers=self._instance_workers) as executor:
                futures = [
                    executor.submit(self.run_instance, sampled_instances)
                    for sampled_instances in self.sampled_instances
                ]
                try:
                    for future in as_completed(futures):
                        future.result()
                except (KeyboardInterrupt, _BreakLoop):
                    msg = (
                        "Received keyboard interrupt, waiting for running instances "
                        "to finish, but cancelled everything else"
                    )
                    self.logger.info(msg)
                    executor.shutdown(wait=False, cancel_futures=True)
                finally:
                    self._progress_manager.print_report()

    def run_instance(self, sampled_instances: list[SampledBatchInstance]) -> list[SampledBatchInstance]:
        """Run all samples for one task concurrently."""
        if not sampled_instances:
            return []

        def run_sample(sampled_instance: SampledBatchInstance) -> SampledBatchInstance:
            # Construct inside the worker so environment startup remains concurrent
            # across samples and thread-local logging is registered on the right thread.
            return SampleAgent(sampled_instance, self).rollout()

        completed_samples: list[SampledBatchInstance] = []
        with ThreadPoolExecutor(max_workers=len(sampled_instances)) as executor:
            futures = [executor.submit(run_sample, sampled_instance) for sampled_instance in sampled_instances]
            try:
                for future in as_completed(futures):
                    completed_samples.append(future.result())
            except (KeyboardInterrupt, _BreakLoop):
                executor.shutdown(wait=False, cancel_futures=True)
                raise _BreakLoop
        return completed_samples

    def should_skip(self, sampled_instance: SampledBatchInstance) -> bool | str:
        """Check if we should skip this instance.
        Returns previous exit status if the instance should be skipped.
        """
        if self._redo_existing:
            return False

        # Check if there's an existing trajectory for this instance
        log_path = self.get_instance_output_dir(sampled_instance.instance_id, sampled_instance.sample_id) / (
            sampled_instance.instance_id + ".traj"
        )
        if not log_path.exists():
            return False

        content = log_path.read_text()
        if not content.strip():
            self.logger.warning("Found empty trajectory: %s. Removing.", log_path)
            log_path.unlink()
            return False

        try:
            data = json.loads(content)
            # If the trajectory has no exit status, it's incomplete and we will redo it
            exit_status = data["info"].get("exit_status", None)
            if exit_status == "early_exit" or exit_status is None:
                self.logger.warning(f"Found existing trajectory with no exit status: {log_path}. Removing.")
                log_path.unlink()
                return False
        except Exception as e:
            self.logger.error(f"Failed to check existing trajectory: {log_path}: {e}. Removing.")
            # If we can't check the trajectory, we will redo it
            log_path.unlink()
            return False
        # otherwise, we will skip it
        self.logger.info(f"⏭️ Skipping existing trajectory: {log_path}")
        return exit_status

    def _add_instance_log_file_handler(
        self, sampled_instance: SampledBatchInstance, multi_worker: bool = False
    ) -> None:
        run_id = self.get_run_id(sampled_instance)
        log_filter = ""
        if multi_worker:

            def matches_run_logger(logger_name: str, suffix=f"-{run_id}") -> bool:
                return logger_name.endswith(suffix)

            log_filter = matches_run_logger
        add_file_handler(
            self.get_instance_output_dir(sampled_instance.instance_id, sampled_instance.sample_id) / "debug.log",
            filter=log_filter,
            level="debug",
            id_=f"{run_id}-debug",
        )

    def _remove_instance_log_file_handler(self, run_id: str) -> None:
        remove_file_handler(f"{run_id}-debug")


def run_from_config(config: RunBatchConfig):
    RunBatch.from_config(config).main()


def run_from_cli(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]
    assert __doc__ is not None
    help_text = (  # type: ignore
        __doc__ + "\n[cyan][bold]=== ALL THE OPTIONS ===[/bold][/cyan]\n\n" + ConfigHelper().get_help(RunBatchConfig)
    )
    run_from_config(BasicCLI(RunBatchConfig, help_text=help_text).get_config(args))  # type: ignore


if __name__ == "__main__":
    run_from_cli()
