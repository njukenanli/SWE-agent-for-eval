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
Prepare config file. The default config file is config/default.yaml

```bash
nohup sweagent run-batch --config config/default.yaml --num_workers 1 --instances.type swe_bench --instances.subset dataset/path.jsonl > log.out 2>&1 &
```
