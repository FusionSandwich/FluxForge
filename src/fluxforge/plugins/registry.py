"""Shared plugin registry infrastructure for next-generation FluxForge features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Iterator, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class PluginMetadata:
    """Describes a registered capability."""

    key: str
    description: str
    recommended: bool = False
    standards_locked: bool = False
    version: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegisteredPlugin(Generic[T]):
    """Single registry entry."""

    key: str
    implementation: T
    metadata: PluginMetadata


class PluginRegistry(Generic[T]):
    """Small typed registry used to keep analytical choices additive."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._plugins: dict[str, RegisteredPlugin[T]] = {}
        self._default_key: str | None = None

    def register(
        self,
        key: str,
        implementation: T,
        *,
        description: str,
        recommended: bool = False,
        standards_locked: bool = False,
        version: str | None = None,
        tags: Iterable[str] = (),
        set_default: bool = False,
    ) -> RegisteredPlugin[T]:
        """Register a new plugin implementation."""

        if key in self._plugins:
            raise ValueError(f"{self.name} plugin {key!r} is already registered")
        metadata = PluginMetadata(
            key=key,
            description=description,
            recommended=recommended,
            standards_locked=standards_locked,
            version=version,
            tags=tuple(tags),
        )
        entry = RegisteredPlugin(key=key, implementation=implementation, metadata=metadata)
        self._plugins[key] = entry
        if self._default_key is None or recommended or set_default:
            self._default_key = key
        return entry

    def clear(self) -> None:
        """Remove all registered plugins."""

        self._plugins.clear()
        self._default_key = None

    def get(self, key: str) -> T:
        """Return the plugin implementation for *key*."""

        return self.get_entry(key).implementation

    def get_entry(self, key: str) -> RegisteredPlugin[T]:
        """Return the full registry entry for *key*."""

        try:
            return self._plugins[key]
        except KeyError as exc:
            raise KeyError(f"{self.name} plugin {key!r} is not registered") from exc

    def set_default(self, key: str) -> None:
        """Set the default plugin key."""

        if key not in self._plugins:
            raise KeyError(f"{self.name} plugin {key!r} is not registered")
        self._default_key = key

    @property
    def default_key(self) -> str | None:
        """Return the current default plugin key, if any."""

        return self._default_key

    def default(self) -> T:
        """Return the default plugin implementation."""

        if self._default_key is None:
            raise LookupError(f"{self.name} registry has no default plugin")
        return self._plugins[self._default_key].implementation

    def entries(self) -> tuple[RegisteredPlugin[T], ...]:
        """Return all registered plugins in insertion order."""

        return tuple(self._plugins.values())

    def keys(self) -> tuple[str, ...]:
        """Return registered plugin keys."""

        return tuple(self._plugins.keys())

    def __contains__(self, key: object) -> bool:
        return key in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self) -> Iterator[RegisteredPlugin[T]]:
        return iter(self._plugins.values())


@dataclass
class PluginRegistries:
    """Named registries for the analytical surfaces defined in the roadmap."""

    peak_search_methods: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry("peak_search_methods")
    )
    roi_background_models: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry("roi_background_models")
    )
    peak_fitters: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry("peak_fitters")
    )
    unfolders: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry("unfolders")
    )
    calibration_models: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry("calibration_models")
    )
    nuclide_id_engines: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry("nuclide_id_engines")
    )
    standards_modules: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry("standards_modules")
    )
    render_backends: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry("render_backends")
    )

    def clear_all(self) -> None:
        """Clear every registry."""

        for registry in (
            self.peak_search_methods,
            self.roi_background_models,
            self.peak_fitters,
            self.unfolders,
            self.calibration_models,
            self.nuclide_id_engines,
            self.standards_modules,
            self.render_backends,
        ):
            registry.clear()


def bootstrap_builtin_registries(
    registries: PluginRegistries | None = None,
) -> PluginRegistries:
    """Return the shared registry container used by built-in code paths."""

    return registries if registries is not None else REGISTRIES


REGISTRIES = PluginRegistries()
