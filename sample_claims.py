#!/usr/bin/env python3
"""Sample 200 balanced claims from FEVER paper_dev.jsonl."""
import json
import random

random.seed(42)  # reproducible

INPUT = "/home/user/workspace/benchmark/paper_dev.jsonl"
OUTPUT = "/home/user/workspace/benchmark/benchmark_claims.jsonl"

# Load all claims
by_label = {"SUPPORTS": [], "REFUTES": [], "NOT ENOUGH INFO": []}
with open(INPUT) as f:
    for line in f:
        c = json.loads(line)
        label = c.get("label", "")
        claim = c.get("claim", "")
        # Filter: 10+ words
        if len(claim.split()) < 10:
            continue
        if label in by_label:
            by_label[label].append({"claim_id": c["id"], "claim_text": claim, "ground_truth": label})

print(f"After 10-word filter:")
for k, v in by_label.items():
    print(f"  {k}: {len(v)}")

# Sample: 67 / 67 / 66
sample = []
sample += random.sample(by_label["SUPPORTS"], 67)
sample += random.sample(by_label["REFUTES"], 67)
sample += random.sample(by_label["NOT ENOUGH INFO"], 66)
random.shuffle(sample)

with open(OUTPUT, "w") as f:
    for c in sample:
        f.write(json.dumps(c) + "\n")

# Verify
counts = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
with open(OUTPUT) as f:
    for line in f:
        counts[json.loads(line)["ground_truth"]] += 1

print(f"\nSampled {sum(counts.values())} claims:")
for k, v in counts.items():
    print(f"  {k}: {v}")

# Show a few examples
print("\nSample claims:")
with open(OUTPUT) as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        c = json.loads(line)
        print(f"  [{c['ground_truth']}] {c['claim_text'][:100]}")
