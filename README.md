# Data Complexity

The Data Complexity Measures in pure Python.

## Install
```bash
$ pip install data-complexity
```

## How it works

One can import the model and use the common _.fit()_ and _.transform()_ functions (sklearn-like interface)

```python
import dcm
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

model = dcm.ComplexityProfile()
model.fit(X, y)
model.transform()
```

## References
[1] How Complex is your classification problem? A survey on measuring classification complexity, https://arxiv.org/abs/1808.03591

[2] The Extended Complexity Library (ECoL), https://github.com/lpfgarcia/ECoL
