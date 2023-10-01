import json
import dcm
from sklearn import datasets

n_samples = 300

for noise in [0.01, 0.2, 0.7]:
    print(f"\n \nEvaluating with noise level {noise}")
    X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=noise)

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