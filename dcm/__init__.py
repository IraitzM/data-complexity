__version__ = "0.1.3"

import os

__DEBUG__ = os.environ.get("DCM_DEBUG", 0)

from .feature_based import FeatureBasedMeasures
from .neighborhood import NeighborhoodMeasures
from .balance import BalanceMeasures
from .imbalance import ImbalanceMeasures
from .linearity import LinearityMeasures
