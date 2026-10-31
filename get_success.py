import os, json

with open("/home/v-kenanli/workspace/comparison/SWE-agent/dataset.jsonl") as f:
    ds = [json.loads(i) for i in f]

python = set(i["instance_id"] for i in ds if i["language"]=="Python")

base = "/home/v-kenanli/workspace/comparison/SWE-agent2/res"
build = set()
organize = set()
for file in os.listdir(base):
    with open(f"{base}/{file}") as f:
        text = f.read()
    if "build success" in text:
        build.add(file.strip(".txt"))
    if "release success" in text:
        organize.add(file.strip(".txt"))

print(len(build), len(build&python))
print(len(organize), len(organize&python))