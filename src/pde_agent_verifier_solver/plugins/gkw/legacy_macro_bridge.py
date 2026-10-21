"""Compatibility import for audit-bound stable GKW macro evidence."""

from .bridge_evidence import (
    GKW_LEGACY_MACRO_BRIDGE_SCHEMA,
    evaluate_gkw_legacy_macro_match,
    validate_gkw_legacy_macro_evidence,
)

__all__ = [
    "GKW_LEGACY_MACRO_BRIDGE_SCHEMA",
    "evaluate_gkw_legacy_macro_match",
    "validate_gkw_legacy_macro_evidence",
]
