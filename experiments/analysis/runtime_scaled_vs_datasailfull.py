import os
import pandas as pd

BASE_DIR = "results/experiments"
OUTPUT_FILE = "results/analysis/runtime_comparison.csv"

DATASETS = ["bace", "bbbp", "tox21", "hiv"]

results = []

for d in DATASETS:
    full_path = os.path.join(BASE_DIR, d, "runtime_full.txt")
    scaled_path = os.path.join(BASE_DIR, d, "runtime_scaled_stratified.txt")

    # Check existence
    has_full = os.path.exists(full_path)
    has_scaled = os.path.exists(scaled_path)

    if not has_scaled:
        print(f"Skipping {d} (missing scaled runtime)")
        continue

    # Read scaled runtime
    with open(scaled_path) as f:
        scaled_time = float(f.readline().split(":")[1])

    # Read full runtime if available
    if has_full:
        with open(full_path) as f:
            full_time = float(f.readline().split(":")[1])
        speedup = full_time / scaled_time
    else:
        full_time = None
        speedup = None
        print(f"{d}: full DataSAIL missing → keeping scaled only")

    results.append({
        "dataset": d,
        "full_runtime_seconds": full_time,
        "scaled_runtime_seconds": scaled_time,
        "speedup": speedup
    })

# Convert to DataFrame
df = pd.DataFrame(results)

# Save CSV
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df = df.round(2)
df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved runtime comparison to:", OUTPUT_FILE)
print(df)