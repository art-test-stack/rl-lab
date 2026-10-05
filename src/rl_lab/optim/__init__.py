"""Named parameter groups for framework-managed optimization."""

from .groups import build_parameter_groups, load_optimizer_config

__all__ = ["build_parameter_groups", "load_optimizer_config"]
