import json
import datasets
import pandas as pd
from dcm import dcm

dataset_file = 'examples/datasets/creditcard.csv'
datasets.download_creditcard(dataset_file)

print("Loading dataset ... ", end="", flush=True)
dataset = pd.read_csv(dataset_file)
X = dataset.drop(columns=['Class']).values
y = dataset['Class'].values
print("DONE")

print("Calculating F1 ... ", end="", flush=True)
ratios, F1 = dcm.F1(X, y)
print("DONE")
print("F1 = {}".format(F1))
print("Discriminant Ratios = {}".format(json.dumps(ratios, indent=2)))

print("Calculating C1 and C2 ... ", end="", flush=True)
C1, C2 = dcm.C12(X, y)
print("DONE")
print("C1 = {}".format(C1))
print("C2 = {}".format(C2))

print("Calculating N1 ... ", end="", flush=True)
N1 = dcm.N1(X, y)
print("DONE")
print("N1 = {}".format(N1))
