"""Compatibility exports for the shared COMSOL runtime."""

from external_tools.comsol_runtime import (
    APPLICATION_MODEL,
    ComsolRuntime,
    provenance_path,
    select_comsol,
)

__all__ = ["APPLICATION_MODEL", "ComsolRuntime", "provenance_path", "select_comsol"]
