# data-complexity
The Data Complexity Measures in Python


## Install
```bash
$ pip install data-complexity
```


## How it works
### Maximum Fisher's Discriminant Ratio (F1)
```python
import dcm
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

feat = dcm.FeatureBasedMeasures()
feat.fit(X, y)
feat.transform()
```

### Fraction of Borderline Points (N1)
```python
import dcm
from sklearn import datasets

bc = datasets.load_breast_cancer(as_frame=True)
X = bc.data.values
y = bc.target.values

nmeans = dcm.NeighborhoodMeasures()
nmeans.fit(X,y)
nmeans.transform()
```

### Entropy of Class Proportions (C1) and Imbalance Ratio (C2)
```python
import dcm
from sklearn import datasets

bc = datasets.load_breast_cancer(as_frame=True)
X = bc.data.values
y = bc.target.values

imb = dcm.ImbalanceMeasures()
imb.fit(X,y)
imb.transform()
```

### Other Measures
Coming soon...


## References
[1] How Complex is your classification problem? A survey on measuring classification complexity, https://arxiv.org/abs/1808.03591

[2] The Extended Complexity Library (ECoL), https://github.com/lpfgarcia/ECoL
