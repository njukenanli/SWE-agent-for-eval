## Prepare venv


```bash
python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip && pip install --editable .

cp -r swerex venv/lib/python3.12/site-packages/swerex
```

## prepare config
The default config is `./config/default.yaml`

## To use Azure_OpenAI API

set agent.model.name = azure/gpt-... in `./config/default.yaml`

```bash
pip install openai azure-identity-broker --upgrade
```

## Rollout
```bash
python main.py --mode {interleaved/sequential/sequential-memory} --config_dir config/default.yaml
```

Runtime log is printed to `logs/{mode}/{instance_id}/{instance_id}.trace.log` for debug