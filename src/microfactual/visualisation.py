"""[DEPRECATED] Visualisation utilities (British spelling).

Deprecated since v0.2.0; scheduled for removal in v1.0. Use
``microfactual.visualization`` (with a 'z'). The file-saving ROC helpers now
live in ``microfactual.visualization.roc_io``.

Importing this module emits a ``DeprecationWarning``. It re-exports the functions
so existing code keeps working (``plot_roc`` maps to the file-saving helper here
for backward compatibility; the modern figure-returning ``plot_roc`` is in
``microfactual.visualization``).
"""

import warnings

from microfactual.visualization.roc_io import plot_roc_to_path as plot_roc
from microfactual.visualization.roc_io import save_roc_curve

warnings.warn(
    "'microfactual.visualisation' is deprecated and will be removed in v1.0. "
    "Use 'microfactual.visualization' (with a 'z').",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["plot_roc", "save_roc_curve"]
