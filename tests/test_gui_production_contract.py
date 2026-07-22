import pytest

from fluxforge.gui.production_contract import (
    ACTION_CATALOG_SCHEMA_VERSION,
    ActionCatalog,
    ActionCatalogEntry,
    ActionKind,
    CopyScope,
    ProductionContractError,
    ProductionCopy,
    assert_production_copy,
    assert_valid_action_catalog,
    lint_production_copy,
    validate_action_catalog,
)


def _entry(**updates):
    values = {
        "action_id": "spectrum.roi.create",
        "kind": ActionKind.PLOT_TOOL,
        "label": "Create ROI",
        "surface": "spectrum.canvas",
        "test_ids": ("test_shift_drag_creates_roi",),
        "description": "Create a signal and background region on the spectrum.",
    }
    values.update(updates)
    return ActionCatalogEntry(**values)


def test_schema_version_is_explicit_and_stable():
    assert ACTION_CATALOG_SCHEMA_VERSION == 1


@pytest.mark.parametrize(
    "copy, expected_token",
    [
        ("Phase 5 parity runner", "numbered phase"),
        ("Review the optimization roadmap", "roadmap"),
        ("This planned screen uses the Qt stack", "planned"),
        ("Reserved for a future capability", "future capability"),
        ("Open the legacy GUI", "legacy GUI"),
        ("bGamma-style residual panel", "bGamma"),
    ],
)
def test_copy_lint_rejects_implementation_narration(copy, expected_token):
    violations = lint_production_copy(
        [ProductionCopy(copy, source="optimization.header")]
    )

    assert expected_token in {violation.token for violation in violations}
    assert all(violation.source == "optimization.header" for violation in violations)


def test_copy_lint_does_not_reject_scientific_phase_language():
    assert not lint_production_copy(
        [
            "Select the irradiation phase and count interval.",
            "The detector is ready for acquisition.",
            "Compare measured and fitted efficiency.",
        ]
    )


@pytest.mark.parametrize("scope", [CopyScope.DEVELOPER, CopyScope.ABOUT_DIAGNOSTICS])
def test_copy_lint_allows_explicit_nonproduction_scopes(scope):
    item = ProductionCopy(
        "Phase 5 roadmap and legacy GUI diagnostics",
        source="diagnostics.build",
        scope=scope,
    )

    assert not lint_production_copy([item])


def test_copy_lint_requires_explicit_scope_not_a_filename_convention():
    item = ProductionCopy(
        "Phase 5 roadmap",
        source="developer_tools.py",
        scope=CopyScope.PRODUCTION,
    )

    with pytest.raises(ProductionContractError, match="developer_tools.py"):
        assert_production_copy([item])


def test_valid_action_catalog_round_trips_through_mapping():
    entry = _entry()
    catalog = ActionCatalog(entries=(entry,))

    restored = ActionCatalog.from_mapping(catalog.as_dict())

    assert restored == catalog
    assert not validate_action_catalog(restored)
    assert_valid_action_catalog(restored)


def test_action_catalog_rejects_unknown_schema_version():
    catalog = ActionCatalog(entries=(_entry(),), schema_version=99)

    violations = validate_action_catalog(catalog)

    assert len(violations) == 1
    assert violations[0].field == "schema_version"


def test_action_catalog_rejects_unstable_duplicate_and_uncovered_actions():
    entries = [
        _entry(action_id="Create ROI", test_ids=()),
        _entry(action_id="Create ROI", test_ids=("",)),
    ]

    violations = validate_action_catalog(entries)
    fields = [violation.field for violation in violations]

    assert fields.count("action_id") == 4
    assert fields.count("test_ids") == 3


def test_action_catalog_rejects_blank_surface_and_forbidden_visible_copy():
    entry = _entry(
        label="Phase 5 ROI tool",
        surface=" ",
        description="Planned parity target",
    )

    violations = validate_action_catalog([entry])

    assert "surface" in {violation.field for violation in violations}
    copy_messages = [
        violation.message for violation in violations if violation.field == "copy"
    ]
    assert any("numbered phase" in message for message in copy_messages)
    assert any("planned" in message for message in copy_messages)
    assert any("parity target" in message for message in copy_messages)


def test_developer_action_is_cataloged_and_covered_but_copy_exempt():
    entry = _entry(
        action_id="developer.parity.open",
        kind=ActionKind.ACTION,
        label="Open Phase 5 parity evidence",
        surface="developer.tools",
        test_ids=("test_developer_tools_expose_parity_evidence",),
        scope=CopyScope.DEVELOPER,
    )

    assert not validate_action_catalog([entry])


def test_catalog_coverage_requirement_can_be_disabled_during_inventory():
    entry = _entry(test_ids=())

    assert not validate_action_catalog([entry], require_test_coverage=False)
    with pytest.raises(ProductionContractError, match="behavioral test"):
        assert_valid_action_catalog([entry])
