import os

__DEBUG__ = os.environ.get("DCM_DEBUG", 0)

from .main import ComplexityProfile
