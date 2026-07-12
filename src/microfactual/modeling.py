"""[DEPRECATED] Model-training utilities.

Deprecated since v0.2.0; scheduled for removal in v1.0. The implementation now
lives in ``microfactual.models.training``; prefer
``microfactual.models.classifiers.MicrobiomeClassifier``.

Importing this module emits a ``DeprecationWarning``. It re-exports ``train_model``
so existing code keeps working.
"""

import warnings

from microfactual.models.training import train_model

warnings.warn(
    "'microfactual.modeling' is deprecated and will be removed in v1.0. "
    "Use 'microfactual.models.classifiers.MicrobiomeClassifier'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["train_model"]
