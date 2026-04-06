"""Plugin registry exports."""

from fluxforge.plugins.registry import (
    REGISTRIES,
    PluginMetadata,
    PluginRegistries,
    PluginRegistry,
    RegisteredPlugin,
    bootstrap_builtin_registries,
)

__all__ = [
    "REGISTRIES",
    "PluginMetadata",
    "PluginRegistries",
    "PluginRegistry",
    "RegisteredPlugin",
    "bootstrap_builtin_registries",
]
