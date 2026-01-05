"""
<DATE> - <TITLE>
Standardized Experiment Report Template
"""

from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 70)
print("<TITLE>")
print("=" * 70)

# ------------------------
# Dataset / file checks
# ------------------------
datasets = {
    "raw": Path("data/raw/moleculenet/bace.csv"),
    "sampled_25pct": Path("data/processed/bace_sampled_25pct.csv"),
    "remaining_75pct": Path("data/processed/bace_remaining_75pct.csv")
}

for name, path in datasets.items():
    if path.exists():
        df = pd.read_csv(path)
        print(f"\n{name} dataset found: {path}")
        print(f"  Samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print(f"\n{name} dataset not found: {path}")

# ------------------------
# Class distribution checks
# ------------------------
if "Class" in df.columns:
    print("\nClass distribution:")
    dist = df["Class"].value_counts(normalize=True)
    for k, v in dist.items():
        print(f"  Class {k}: {v:.3f}")

# ------------------------
# Summary
# ------------------------
print("\n" + "=" * 70)
print("WHAT WE DID")
print("=" * 70)
print("""
- Describe key milestones and steps here
""")

print("\n" + "=" * 70)
print("WHY THIS MATTERS")
print("=" * 70)
print("""
- Explain why the milestone is important for thesis
""")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
- Describe next experiments or pipeline steps
""")
