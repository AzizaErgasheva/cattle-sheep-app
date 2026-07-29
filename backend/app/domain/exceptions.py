"""Domain-level errors. The API layer translates these into HTTP responses."""


class DomainError(Exception):
    """Base class for all domain-level errors."""


class InvalidImageError(DomainError):
    """Raised when the uploaded bytes can't be decoded as an image."""


class ModelNotLoadedError(DomainError):
    """Raised when a classifier/explainer is used but its model file is missing."""


class ModelNotFoundError(DomainError):
    """Raised when a requested model_name isn't in the registry (bad selector input)."""
