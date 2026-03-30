from fluxforge.plugins import PluginRegistries, PluginRegistry, bootstrap_builtin_registries


def test_plugin_registry_registers_and_tracks_default():
    registry = PluginRegistry("render_backends")

    registry.register(
        "pyqtgraph",
        object(),
        description="Primary renderer",
        recommended=True,
    )
    registry.register("vispy", object(), description="Optional renderer stub")

    assert len(registry) == 2
    assert registry.default_key == "pyqtgraph"
    assert registry.get("pyqtgraph") is registry.default()


def test_plugin_registry_rejects_duplicate_keys():
    registry = PluginRegistry("peak_fitters")
    registry.register("gaussian", object(), description="Gaussian fitter")

    try:
        registry.register("gaussian", object(), description="Duplicate")
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:  # pragma: no cover - defensive fallback
        raise AssertionError("duplicate registration should have failed")


def test_plugin_registries_clear_all():
    registries = PluginRegistries()
    registries.peak_fitters.register(
        "gaussian",
        object(),
        description="Gaussian fitter",
        recommended=True,
    )
    registries.render_backends.register(
        "pyqtgraph",
        object(),
        description="Renderer",
        recommended=True,
    )

    registries.clear_all()

    assert len(registries.peak_fitters) == 0
    assert len(registries.render_backends) == 0


def test_bootstrap_builtin_registries_returns_shared_container():
    assert bootstrap_builtin_registries() is bootstrap_builtin_registries()
