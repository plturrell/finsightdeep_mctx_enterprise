"""
Legacy Core Module

This module is maintained for backward compatibility.
For new code, use the src.aiq.owl package instead.
"""

# Import all components from the new package
from src.aiq.owl.auth import *
from src.aiq.owl.config import *
from src.aiq.owl.exceptions import *
from src.aiq.owl.logging import *
from src.aiq.owl.rate_limit import *
from src.aiq.owl.security import *

# For backward compatibility, also import the secrets module
# which has been renamed to security in the new structure
from src.aiq.owl.security import (
    SecretManager, HanaSecrets, DataSphereSecrets, initialize_secrets
)

# Warn about deprecation
import warnings
warnings.warn(
    "The api.app.core module is deprecated. "
    "Please use src.aiq.owl instead.",
    DeprecationWarning,
    stacklevel=2
)