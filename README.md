# Benchmarking SWE-agent with any LLM on SWE benchmarks

## Prepare venv

```bash
python3.12 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip && pip install --editable .

cp -r swerex venv/lib/python3.12/site-packages/swerex
```

## To use Azure_OpenAI API

set agent.model.name = azure/gpt-... in `./config/default.yaml`

```bash
pip install openai azure-identity-broker --upgrade
```

modify sweagent/agent/models.py::LiteLLMModel::_single_query to accept azure_ad_token_provider

## Rollout
Use `config/train.yaml` for training rollouts and `config/test.yaml` for test
rollouts. `agent.model.samples` controls how many independent times each
instance runs.

```bash
nohup sweagent run-batch \
  --config config/train.yaml \
  --num_workers 2 \
  --instances.type swe_bench \
  --instances.subset dataset/path.jsonl \
  --epoch 0 \
  > /dev/null 2>&1 &
```

Training uses `samples: 8`, so every instance starts from the beginning eight
independent times and writes sample IDs `0` through `7`. Test uses `samples: 1`
and writes only sample ID `0`.

`--num_workers` globally limits how many sampled agents may execute a rollout
at once. Each task still starts all of its configured samples in a separate
inner thread pool; a shared semaphore limits concurrent agent rollouts across
all task and sample pools. The outer task pool uses
`num_workers // samples + 3` threads so environment startup and evaluation can
overlap with active rollouts.

Each run writes only these artifacts:

```text
logs/{model.name}/{epoch}/{train|test}/{instance_id}/{sample_id}/
  debug.log
  {instance_id}.traj
  {instance_id}.patch
  eval/
```

The patch file is created before the run starts. If a run exits abnormally,
SWE-agent recovers the latest submission from agent state or the trajectory
when possible; otherwise the patch remains empty.

After each sample, SWE-bench evaluates the patch with a 1,800-second timeout.
Evaluation logs are written to `eval/`, and the trajectory records the boolean
result at `info.success`. Evaluation run IDs use
`{epoch}_{train|test}_{instance_id}_{sample_id}`.
