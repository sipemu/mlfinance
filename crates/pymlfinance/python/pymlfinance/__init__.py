import sys as _sys

from pymlfinance._native import *  # noqa: F401,F403
from pymlfinance import _native

# Register submodules at expected paths for backward compat
# e.g. `from pymlfinance.core import ewma` or `from pymlfinance import core`
for _name in ("core", "data", "labeling", "sampling", "features", "modeling", "backtesting"):
    _submod = getattr(_native, _name)
    _sys.modules[f"{__name__}.{_name}"] = _submod
