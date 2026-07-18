import pandas as pd

train = pd.read_csv("results/experiments/splits/bace/datasail/train.csv")
val = pd.read_csv("results/experiments/splits/bace/datasail/val.csv")
test = pd.read_csv("results/experiments/splits/bace/datasail/test.csv")

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))
print("Total:", len(train) + len(val) + len(test))
