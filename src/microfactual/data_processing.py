"""[DEPRECATED] Data-processing utilities.

Deprecated since v0.2.0; scheduled for removal in v1.0. The implementations now
live in ``microfactual.core.processing``; prefer the sklearn-style transforms in
``microfactual.preprocessing`` (``AbundanceFilter``, ``PrevalenceFilter``,
``CLRTransform``) or ``microfactual.core.dataset``.

Importing this module emits a ``DeprecationWarning``. It re-exports the functions
so existing code keeps working.
"""

import warnings

from microfactual.core.processing import clr_transform, filter_data, load_data

warnings.warn(
    "'microfactual.data_processing' is deprecated and will be removed in v1.0. "
    "Use 'microfactual.core.dataset' / 'microfactual.preprocessing', or import "
    "these functions from the top-level 'microfactual' package.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["load_data", "filter_data", "clr_transform"]
