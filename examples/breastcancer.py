import json
from dcm import dcm
from sklearn.datasets import load_breast_cancer

print("Loading dataset ... ", end="", flush=True)
X_df, y_df = load_breast_cancer(return_X_y=True, as_frame=True)
print("DONE")

X = X_df.to_numpy()
y = y_df.to_numpy()

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