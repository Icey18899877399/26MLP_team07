"""Public exception hierarchy for ml_core callers."""


class MLCoreError(Exception):
    """Base class for every documented ml_core failure."""


class UnknownModelError(MLCoreError):
    """Raised when a model identifier is not registered."""


class UnknownDatasetError(MLCoreError):
    """Raised when a dataset identifier is not registered."""


class IncompatibleDatasetError(MLCoreError):
    """Raised when a model cannot run on the selected dataset."""


class InvalidConfigError(MLCoreError):
    """Raised when common experiment configuration is malformed."""


class InvalidParameterError(MLCoreError):
    """Raised when model-specific parameters are malformed."""


class ExperimentExecutionError(MLCoreError):
    """Raised when a valid experiment fails unexpectedly."""
