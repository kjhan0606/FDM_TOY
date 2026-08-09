"""FDM SMBH delay toy model."""

from .config import CaseConfig, load_config
from .orbit import IntegrationResult, integrate_case

__all__ = ["CaseConfig", "IntegrationResult", "integrate_case", "load_config"]
__version__ = "0.1.0"
