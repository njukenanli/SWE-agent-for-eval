import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.agent.models import InstantEmptySubmitModelConfig
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.environment.swe_env import EnvironmentConfig
from sweagent.run._progress import RunBatchProgressManager
from sweagent.run.batch_instances import BatchInstance
from sweagent.run.run import main
from sweagent.run.run_batch import (
    RunBatch,
    RunBatchConfig,
    SampleAgent,
    SampledBatchInstance,
    _submission_from_agent,
    _submission_from_trajectory,
    _write_trajectory_success,
)
from sweagent.types import AgentRunResult
from sweagent.utils.log import get_logger, register_thread_name
from swerex.deployment.config import DummyDeploymentConfig


def test_default_output_dir_uses_model_epoch_and_config_name():
    config = SimpleNamespace(
        output_dir=Path("DEFAULT"),
        _config_files=[Path("config/train.yaml")],
        agent=SimpleNamespace(model=SimpleNamespace(name="my-qwen-model")),
        epoch=4,
        get_run_type=lambda: "train",
    )

    RunBatchConfig.set_default_output_dir(config)

    assert config.output_dir == Path("logs/my-qwen-model/4/train")


def test_best_effort_submission_from_trajectory(tmp_path):
    trajectory_path = tmp_path / "instance.traj"
    trajectory_path.write_text('{"attempts":[{"info":{"submission":"old"}},{"info":{"submission":"latest"}}]}')

    assert _submission_from_trajectory(trajectory_path) == "latest"


def test_best_effort_submission_from_agent():
    agent = SimpleNamespace(info={"submission": "in-memory patch"})

    assert _submission_from_agent(agent) == "in-memory patch"


def test_write_trajectory_success(tmp_path):
    trajectory_path = tmp_path / "instance.traj"
    trajectory_path.write_text('{"info":{"exit_status":"submitted"},"trajectory":[]}')

    _write_trajectory_success(trajectory_path, True)

    data = json.loads(trajectory_path.read_text())
    assert data["info"]["success"] is True
    assert data["info"]["exit_status"] == "submitted"


def test_evaluate_sample_writes_success_and_uses_expected_arguments(tmp_path, monkeypatch):
    instance_id = "owner__repo-1"
    swebench_instance = {"instance_id": instance_id}
    sampled_instance = SampledBatchInstance(
        instance=SimpleNamespace(
            problem_statement=SimpleNamespace(id=instance_id),
            swebench_instance=swebench_instance,
        ),
        sample_id=7,
    )
    sample_dir = tmp_path / instance_id / "7"
    sample_dir.mkdir(parents=True)
    (sample_dir / f"{instance_id}.patch").write_text("patch content")
    trajectory_path = sample_dir / f"{instance_id}.traj"
    trajectory_path.write_text('{"info":{"exit_status":"submitted"}}')
    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr("sweagent.run.run_batch.evaluate", fake_evaluate)
    runner = SimpleNamespace(
        output_dir=tmp_path,
        epoch=3,
        run_type="test",
        agent_config=SimpleNamespace(model=SimpleNamespace(name="my-model")),
        logger=MagicMock(),
        get_instance_output_dir=lambda current_instance_id, sample_id: tmp_path / current_instance_id / str(sample_id),
        get_run_id=lambda current: f"3_test_{current.instance_id}_{current.sample_id}",
    )

    sample_agent = SimpleNamespace(
        sample_instance=sampled_instance,
        run_batch=runner,
        run_id=runner.get_run_id(sampled_instance),
        output_dir=sample_dir,
        patch_path=sample_dir / f"{instance_id}.patch",
        trajectory_path=trajectory_path,
    )

    success = SampleAgent._evaluate_sample(sample_agent)

    assert success is True
    assert calls == [
        {
            "prediction": {
                "model_name_or_path": "my-model",
                "instance_id": instance_id,
                "model_patch": "patch content",
            },
            "instance": swebench_instance,
            "run_id": "3_test_owner__repo-1_7",
            "log_dir": sample_dir / "eval",
            "timeout": 1800,
            "namespace": "swebench",
        }
    ]
    assert json.loads(trajectory_path.read_text())["info"]["success"] is True


def test_evaluate_sample_none_warns_and_writes_false(tmp_path, monkeypatch):
    instance_id = "owner__repo-1"
    sampled_instance = SampledBatchInstance(
        instance=SimpleNamespace(
            problem_statement=SimpleNamespace(id=instance_id),
            swebench_instance={"instance_id": instance_id},
        ),
        sample_id=0,
    )
    sample_dir = tmp_path / instance_id / "0"
    sample_dir.mkdir(parents=True)
    (sample_dir / f"{instance_id}.patch").write_text("")
    trajectory_path = sample_dir / f"{instance_id}.traj"
    trajectory_path.write_text('{"info":{}}')
    monkeypatch.setattr("sweagent.run.run_batch.evaluate", lambda **kwargs: None)
    logger = MagicMock()
    runner = SimpleNamespace(
        output_dir=tmp_path,
        epoch=1,
        run_type="train",
        agent_config=SimpleNamespace(model=SimpleNamespace(name="my-model")),
        logger=logger,
        get_instance_output_dir=lambda current_instance_id, sample_id: tmp_path / current_instance_id / str(sample_id),
        get_run_id=lambda current: f"1_train_{current.instance_id}_{current.sample_id}",
    )

    sample_agent = SimpleNamespace(
        sample_instance=sampled_instance,
        run_batch=runner,
        run_id=runner.get_run_id(sampled_instance),
        output_dir=sample_dir,
        patch_path=sample_dir / f"{instance_id}.patch",
        trajectory_path=trajectory_path,
    )

    success = SampleAgent._evaluate_sample(sample_agent)

    assert success is False
    logger.warning.assert_called_once()
    assert json.loads(trajectory_path.read_text())["info"]["success"] is False


def test_run_instance_runs_all_samples_in_parallel(monkeypatch):
    sampled_instances = [SimpleNamespace(sample_id=sample_id) for sample_id in range(3)]
    barrier = threading.Barrier(len(sampled_instances))
    thread_ids = set()
    lock = threading.Lock()

    def run_sample(sampled_instance):
        with lock:
            thread_ids.add(threading.get_ident())
        barrier.wait(timeout=2)
        return sampled_instance

    class FakeSampleAgent:
        def __init__(self, sampled_instance, _run_batch):
            self.sampled_instance = sampled_instance

        def rollout(self):
            return run_sample(self.sampled_instance)

    monkeypatch.setattr("sweagent.run.run_batch.SampleAgent", FakeSampleAgent)
    runner = SimpleNamespace()

    completed = RunBatch.run_instance(runner, sampled_instances)

    assert {sample.sample_id for sample in completed} == {0, 1, 2}
    assert len(thread_ids) == len(sampled_instances)


def test_sampled_instances_are_grouped_by_task():
    instances = [
        SimpleNamespace(problem_statement=SimpleNamespace(id=instance_id)) for instance_id in ("task-a", "task-b")
    ]
    runner = RunBatch(
        instances=instances,
        agent_config=SimpleNamespace(model=SimpleNamespace(id="model", name="model", samples=3)),
        hooks=[MagicMock()],
        parallel_instances=5,
        progress_bar=False,
    )

    assert [[sample.instance_id for sample in sample_group] for sample_group in runner.sampled_instances] == [
        ["task-a", "task-a", "task-a"],
        ["task-b", "task-b", "task-b"],
    ]
    assert [[sample.sample_id for sample in sample_group] for sample_group in runner.sampled_instances] == [
        [0, 1, 2],
        [0, 1, 2],
    ]
    assert runner._num_workers == 2


def test_main_multi_worker_runs_task_groups_in_parallel(monkeypatch):
    task_groups = [["task-a"], ["task-b"]]
    barrier = threading.Barrier(len(task_groups))
    thread_ids = set()
    lock = threading.Lock()

    def run_instance(sampled_instances):
        with lock:
            thread_ids.add(threading.get_ident())
        barrier.wait(timeout=2)
        return sampled_instances

    class DummyLive:
        def __init__(self, _render_group):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("sweagent.run.run_batch.Live", DummyLive)
    monkeypatch.setattr("sweagent.run.run_batch.add_logger_names_to_stream_handlers", lambda: None)
    monkeypatch.setattr("sweagent.run.run_batch.set_stream_handler_levels", lambda _level: None)
    runner = SimpleNamespace(
        _num_workers=2,
        sampled_instances=task_groups,
        run_instance=run_instance,
        logger=MagicMock(),
        _progress_manager=SimpleNamespace(render_group=object(), print_report=lambda: None),
    )

    RunBatch.main_multi_worker(runner)

    assert len(thread_ids) == len(task_groups)


def test_parallel_samples_write_to_distinct_debug_logs(tmp_path):
    instance_id = "owner__repo-logging"
    sampled_instances = [
        SampledBatchInstance(
            instance=SimpleNamespace(problem_statement=SimpleNamespace(id=instance_id)),
            sample_id=sample_id,
        )
        for sample_id in (1, 10)
    ]
    barrier = threading.Barrier(len(sampled_instances))
    runner = SimpleNamespace(
        output_dir=tmp_path,
        epoch=2,
        run_type="train",
        get_run_id=lambda sample: f"2_train_{sample.instance_id}_{sample.sample_id}",
        get_instance_output_dir=lambda current_instance_id, sample_id: tmp_path / current_instance_id / str(sample_id),
    )

    def write_sample_log(sampled_instance):
        run_id = runner.get_run_id(sampled_instance)
        register_thread_name(run_id)
        RunBatch._add_instance_log_file_handler(runner, sampled_instance, multi_worker=True)
        try:
            barrier.wait(timeout=2)
            get_logger("sample-log-test").debug("marker[%s]", sampled_instance.sample_id)
        finally:
            RunBatch._remove_instance_log_file_handler(runner, run_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(write_sample_log, sampled_instances))

    sample_1_log = (tmp_path / instance_id / "1" / "debug.log").read_text()
    sample_10_log = (tmp_path / instance_id / "10" / "debug.log").read_text()
    assert "marker[1]" in sample_1_log
    assert "marker[10]" not in sample_1_log
    assert "marker[10]" in sample_10_log
    assert "marker[1]" not in sample_10_log


def test_each_sample_owns_artifacts_under_full_log_hierarchy(tmp_path, monkeypatch):
    model_name = "my-qwen-model"
    epoch = 4
    run_type = "train"
    instance_id = "owner__repo-artifacts"
    output_dir = tmp_path / "logs" / model_name / str(epoch) / run_type
    instance = SimpleNamespace(problem_statement=SimpleNamespace(id=instance_id))
    runner = RunBatch(
        instances=[instance],
        agent_config=SimpleNamespace(model=SimpleNamespace(id=model_name, name=model_name, samples=3)),
        output_dir=output_dir,
        hooks=[MagicMock()],
        epoch=epoch,
        run_type=run_type,
        progress_bar=False,
        random_delay_multiplier=0,
    )

    def start_fake_agent_and_environment(sample_agent):
        sampled_instance = sample_agent.sample_instance
        sample_agent.instance = sampled_instance.instance

        def run_agent(**_kwargs):
            sample_dir = sample_agent.output_dir
            marker = f"sample[{sampled_instance.sample_id}]"
            get_logger("artifact-test").debug(marker)
            (sample_dir / f"{instance_id}.patch").write_text(marker)
            (sample_dir / f"{instance_id}.traj").write_text(
                json.dumps(
                    {
                        "info": {"exit_status": "submitted"},
                        "trajectory": [],
                    }
                )
            )
            return AgentRunResult(
                info={"exit_status": "submitted", "submission": marker},
                trajectory=[],
            )

        sample_agent.agent = SimpleNamespace(
            run=run_agent,
            logger=MagicMock(),
            info={},
        )
        sample_agent.env = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        SampleAgent,
        "_start_agent_and_environment",
        start_fake_agent_and_environment,
    )
    monkeypatch.setattr(SampleAgent, "_evaluate_sample", lambda _self: True)

    runner.run_instance(runner.sampled_instances[0])

    instance_dir = output_dir / instance_id
    assert {path.name for path in instance_dir.iterdir()} == {"0", "1", "2"}
    for sample_id in range(3):
        marker = f"sample[{sample_id}]"
        sample_dir = instance_dir / str(sample_id)
        debug_path = sample_dir / "debug.log"
        patch_path = sample_dir / f"{instance_id}.patch"
        trajectory_path = sample_dir / f"{instance_id}.traj"

        assert debug_path.is_file()
        assert patch_path.is_file()
        assert trajectory_path.is_file()
        assert marker in debug_path.read_text()
        assert patch_path.read_text() == marker
        assert json.loads(trajectory_path.read_text())["info"]["exit_status"] == "submitted"


def test_sample_agent_initializes_default_agent_and_starts_own_env(tmp_path, monkeypatch):
    instance_id = "owner__repo-default-agent"
    started_envs = []

    class FakeEnv:
        def __init__(self):
            self.deployment = SimpleNamespace(add_hook=MagicMock())
            self.started = False
            self.closed = False

        def add_hook(self, _hook):
            pass

        def start(self):
            self.started = True
            started_envs.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr("sweagent.run.run_batch.SWEEnv.from_config", lambda _config: FakeEnv())
    instance = BatchInstance(
        env=EnvironmentConfig(deployment=DummyDeploymentConfig()),
        problem_statement=TextProblemStatement(id=instance_id, text="test problem"),
        swebench_instance=None,
    )
    runner = RunBatch(
        instances=[instance],
        agent_config=DefaultAgentConfig(model=InstantEmptySubmitModelConfig()),
        output_dir=tmp_path,
        hooks=[MagicMock()],
        progress_bar=False,
        random_delay_multiplier=0,
    )

    sample_agent = SampleAgent(runner.sampled_instances[0][0], runner)

    assert isinstance(sample_agent.agent, DefaultAgent)
    assert sample_agent.env is started_envs[0]
    assert sample_agent.env.started is True
    sample_agent.agent.run = MagicMock(
        return_value=AgentRunResult(
            info={"exit_status": "submitted", "submission": ""},
            trajectory=[],
        )
    )

    assert sample_agent.rollout() == runner.sampled_instances[0][0]
    assert sample_agent.env.closed is True


def test_sample_agent_startup_error_recovers_patch_and_finalizes(tmp_path, monkeypatch):
    instance_id = "owner__repo-startup-error"
    instance = SimpleNamespace(
        problem_statement=SimpleNamespace(id=instance_id),
        swebench_instance=None,
    )
    runner = RunBatch(
        instances=[instance],
        agent_config=SimpleNamespace(model=SimpleNamespace(id="model", name="model", samples=1)),
        output_dir=tmp_path,
        hooks=[MagicMock()],
        progress_bar=False,
        random_delay_multiplier=0,
    )
    env = SimpleNamespace(close=MagicMock())

    def fail_startup(sample_agent):
        sample_agent.instance = instance
        sample_agent.agent = SimpleNamespace(
            info={"submission": "recovered patch"},
            logger=MagicMock(),
        )
        sample_agent.env = env
        msg = "environment startup failed"
        raise RuntimeError(msg)

    evaluate_sample = MagicMock(return_value=False)
    monkeypatch.setattr(SampleAgent, "_start_agent_and_environment", fail_startup)
    monkeypatch.setattr(SampleAgent, "_evaluate_sample", evaluate_sample)
    sample_agent = SampleAgent(runner.sampled_instances[0][0], runner)

    assert sample_agent.rollout() == runner.sampled_instances[0][0]
    assert (tmp_path / instance_id / "0" / f"{instance_id}.patch").read_text() == "recovered patch"
    env.close.assert_called_once_with()
    evaluate_sample.assert_called_once_with()


def test_sample_agent_preserves_skip_without_starting_or_evaluating(tmp_path, monkeypatch):
    instance_id = "owner__repo-skipped"
    instance = SimpleNamespace(problem_statement=SimpleNamespace(id=instance_id))
    runner = RunBatch(
        instances=[instance],
        agent_config=SimpleNamespace(model=SimpleNamespace(id="model", name="model", samples=1)),
        output_dir=tmp_path,
        hooks=[MagicMock()],
        progress_bar=False,
        random_delay_multiplier=0,
    )
    sample_dir = tmp_path / instance_id / "0"
    sample_dir.mkdir(parents=True)
    (sample_dir / f"{instance_id}.traj").write_text(json.dumps({"info": {"exit_status": "submitted"}}))
    start_sample = MagicMock()
    evaluate_sample = MagicMock()
    monkeypatch.setattr(SampleAgent, "_start_agent_and_environment", start_sample)
    monkeypatch.setattr(SampleAgent, "_evaluate_sample", evaluate_sample)

    sample_agent = SampleAgent(runner.sampled_instances[0][0], runner)

    assert sample_agent.rollout() == runner.sampled_instances[0][0]
    start_sample.assert_not_called()
    evaluate_sample.assert_not_called()
    assert (sample_dir / f"{instance_id}.patch").is_file()


def test_progress_status_reads_and_writes_hold_lock():
    progress = RunBatchProgressManager(num_instances=1)

    class LockCheckingStatuses(dict):
        def values(self):
            assert progress._lock.locked()
            return super().values()

        def __getitem__(self, key):
            assert progress._lock.locked()
            return super().__getitem__(key)

    progress._instances_by_exit_status = LockCheckingStatuses({"submitted": []})
    assert progress.n_completed == 0

    progress.on_instance_start("task-0")
    progress.on_instance_end("task-0", "submitted")

    assert progress.n_completed == 1


@pytest.mark.slow
def test_expert_instances(test_data_sources_path: Path, tmp_path: Path):
    ds_path = test_data_sources_path / "expert_instances.yaml"
    assert ds_path.exists()
    cmd = [
        "run-batch",
        "--config",
        "config/test.yaml",
        "--agent.model.name",
        "instant_empty_submit",
        "--instances.type",
        "expert_file",
        "--instances.path",
        str(ds_path),
        "--output_dir",
        str(tmp_path),
        "--epoch",
        "0",
        "--raise_exceptions",
        "True",
    ]
    main(cmd)
    for _id in ["simple_test_problem", "simple_test_problem_2"]:
        assert (tmp_path / _id / "0" / f"{_id}.traj").exists(), list(tmp_path.iterdir())


@pytest.mark.slow
def test_simple_instances(test_data_sources_path: Path, tmp_path: Path):
    ds_path = test_data_sources_path / "simple_instances.yaml"
    assert ds_path.exists()
    cmd = [
        "run-batch",
        "--config",
        "config/test.yaml",
        "--agent.model.name",
        "instant_empty_submit",
        "--instances.path",
        str(ds_path),
        "--output_dir",
        str(tmp_path),
        "--epoch",
        "0",
        "--agent.model.samples",
        "2",
        "--raise_exceptions",
        "True",
    ]
    main(cmd)
    for sample_id in range(2):
        sample_dir = tmp_path / "simple_test_problem" / str(sample_id)
        assert (sample_dir / "simple_test_problem.traj").exists(), list(tmp_path.iterdir())
        assert {path.name for path in sample_dir.iterdir()} == {
            "debug.log",
            "simple_test_problem.traj",
            "simple_test_problem.patch",
            "eval",
        }


def test_empty_instances_simple(test_data_sources_path: Path, tmp_path: Path):
    ds_path = test_data_sources_path / "simple_instances.yaml"
    assert ds_path.exists()
    cmd = [
        "run-batch",
        "--config",
        "config/test.yaml",
        "--agent.model.name",
        "instant_empty_submit",
        "--instances.path",
        str(ds_path),
        "--output_dir",
        str(tmp_path),
        "--epoch",
        "0",
        "--raise_exceptions",
        "True",
        "--instances.filter",
        "doesnotmatch",
    ]
    with pytest.raises(ValueError, match="No instances to run"):
        main(cmd)


def test_empty_instances_expert(test_data_sources_path: Path, tmp_path: Path):
    ds_path = test_data_sources_path / "expert_instances.yaml"
    assert ds_path.exists()
    cmd = [
        "run-batch",
        "--config",
        "config/test.yaml",
        "--agent.model.name",
        "instant_empty_submit",
        "--instances.path",
        str(ds_path),
        "--instances.type",
        "expert_file",
        "--output_dir",
        str(tmp_path),
        "--epoch",
        "0",
        "--raise_exceptions",
        "True",
        "--instances.filter",
        "doesnotmatch",
    ]
    with pytest.raises(ValueError, match="No instances to run"):
        main(cmd)


# This doesn't work because we need to retrieve environment variables from the environment
# in order to format our templates.
# def test_run_batch_swe_bench_instances(tmp_path: Path):
#     cmd = [
#         "run-batch",
#         "--agent.model.name",
#         "instant_empty_submit",
#         "--instances.subset",
#         "lite",
#         "--instances.split",
#         "test",
#         "--instances.slice",
#         "0:1",
#         "--output_dir",
#         str(tmp_path),
#         "--raise_exceptions",
#         "--instances.deployment.type",
#         "dummy",
#     ]
#     main(cmd)
