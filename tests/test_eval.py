from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from swebench.harness.constants import LOG_INSTANCE

from sweagent.run import eval as eval_module


class FakeContainer:
    def __init__(self, name: str):
        self.id = name
        self.name = name

    def start(self) -> None:
        pass

    def exec_run(self, *_args, **_kwargs):
        return SimpleNamespace(exit_code=0, output=b"")


class FakeTestSpec:
    instance_id = "owner__repo-1"
    is_remote_image = True
    instance_image_key = "swebench/sweb.eval.owner__repo-1:latest"
    eval_script = "true"

    def get_instance_container_name(self, run_id: str) -> str:
        return f"sweb.eval.{self.instance_id.lower()}.{run_id}"


def test_evaluate_closes_docker_client(tmp_path, monkeypatch):
    client = MagicMock()
    test_spec = FakeTestSpec()
    monkeypatch.setattr(eval_module.docker, "from_env", lambda: client)
    monkeypatch.setattr(eval_module, "make_test_spec", lambda *_args, **_kwargs: test_spec)
    monkeypatch.setattr(eval_module, "run_instance", lambda *_args, **_kwargs: True)

    result = eval_module.evaluate(
        prediction={"model_patch": "patch"},
        instance={"instance_id": test_spec.instance_id},
        run_id="run-0",
        log_dir=tmp_path,
        timeout=1800,
        namespace="swebench",
    )

    assert result is True
    client.close.assert_called_once_with()


def test_evaluate_closes_docker_client_when_setup_fails(tmp_path, monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(eval_module.docker, "from_env", lambda: client)
    monkeypatch.setattr(
        eval_module,
        "make_test_spec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("setup failed")),
    )

    try:
        eval_module.evaluate(
            prediction={"model_patch": "patch"},
            instance={"instance_id": "owner__repo-1"},
            run_id="run-0",
            log_dir=tmp_path,
            timeout=1800,
            namespace="swebench",
        )
    except RuntimeError as error:
        assert str(error) == "setup failed"
    else:
        raise AssertionError("Expected test-spec creation to fail")

    client.close.assert_called_once_with()


def test_run_instance_creates_local_image_build_link(tmp_path, monkeypatch):
    test_spec = FakeTestSpec()
    test_spec.is_remote_image = False
    monkeypatch.setattr(eval_module, "INSTANCE_IMAGE_BUILD_DIR", tmp_path / "builds")
    expected_build_dir = (
        tmp_path
        / "builds"
        / test_spec.instance_image_key.replace(":", "__")
    )
    monkeypatch.setattr(
        eval_module,
        "build_container",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after setup")),
    )
    monkeypatch.setattr(eval_module, "cleanup_container", lambda *_args: None)

    log_dir = tmp_path / "sample" / "eval"
    result = eval_module.run_instance(
        test_spec=test_spec,
        pred={"model_patch": "patch"},
        rm_image=False,
        client=SimpleNamespace(),
        run_id="run-0",
        timeout=1800,
        log_dir=log_dir,
    )

    assert result is None
    image_build_link = log_dir / "image_build_dir"
    assert image_build_link.is_symlink()
    assert image_build_link.readlink() == expected_build_dir.absolute()


def test_parallel_samples_use_distinct_eval_logs_and_container_names(tmp_path, monkeypatch):
    test_spec = FakeTestSpec()
    container_names: list[str] = []

    def fake_build_container(current_test_spec, _client, run_id, logger, *_args):
        container_name = current_test_spec.get_instance_container_name(run_id)
        container_names.append(container_name)
        logger.info("evaluation marker for %s", run_id)
        return FakeContainer(container_name)

    monkeypatch.setattr(eval_module, "build_container", fake_build_container)
    monkeypatch.setattr(eval_module, "cleanup_container", lambda *_args: None)
    monkeypatch.setattr(eval_module, "copy_to_container", lambda *_args: None)
    monkeypatch.setattr(
        eval_module,
        "exec_run_with_timeout",
        lambda *_args: ("tests passed", False, 0.1),
    )
    monkeypatch.setattr(
        eval_module,
        "get_eval_report",
        lambda **_kwargs: {test_spec.instance_id: {"resolved": True}},
    )

    run_ids = [
        "3_train_owner__repo-1_0",
        "3_train_owner__repo-1_1",
    ]

    def evaluate_sample(run_id: str) -> bool | None:
        sample_id = run_id.rsplit("_", 1)[-1]
        return eval_module.run_instance(
            test_spec=test_spec,
            pred={"model_patch": "diff --git a/a b/a"},
            rm_image=False,
            client=SimpleNamespace(),
            run_id=run_id,
            timeout=1800,
            log_dir=tmp_path / sample_id / "eval",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(evaluate_sample, run_ids))

    assert results == [True, True]
    assert set(container_names) == {
        test_spec.get_instance_container_name(run_id) for run_id in run_ids
    }
    assert len(set(container_names)) == len(run_ids)

    for run_id in run_ids:
        sample_id = run_id.rsplit("_", 1)[-1]
        log_path = Path(tmp_path) / sample_id / "eval" / LOG_INSTANCE
        log_content = log_path.read_text()
        assert f"evaluation marker for {run_id}" in log_content
        assert all(
            f"evaluation marker for {other_run_id}" not in log_content
            for other_run_id in run_ids
            if other_run_id != run_id
        )
