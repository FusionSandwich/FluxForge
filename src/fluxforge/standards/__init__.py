"""Standards registry exports."""

from __future__ import annotations

from fluxforge.plugins import PluginRegistries, bootstrap_builtin_registries
from fluxforge.standards.base import (
    LockedSetting,
    StandardsCheck,
    StandardsEvaluation,
    StandardsEvaluationContext,
    StandardsModule,
)
from fluxforge.standards.c1030 import (
    C1030Module,
    PuIsotopicsResult,
    PuLineObservation,
    compute_pu_isotopics,
)
from fluxforge.standards.c1232 import C1232Module
from fluxforge.standards.e1218 import E1218Module
from fluxforge.standards.e1297 import E1297Module, currie_mda
from fluxforge.standards.e181 import E181Module
from fluxforge.standards.e261 import E261Module
from fluxforge.standards.qa_monitor import QAMonitor, QARecord, QAStatus


def register_builtin_standards_modules(
    registries: PluginRegistries,
) -> PluginRegistries:
    """Register built-in ASTM and QA standards modules."""

    registries.standards_modules.clear()
    for module_cls, recommended in (
        (E181Module, True),
        (E1297Module, False),
        (E1218Module, False),
        (C1232Module, False),
        (C1030Module, False),
        (E261Module, False),
    ):
        module = module_cls()
        registries.standards_modules.register(
            module.standard_id,
            module,
            description=module.description(),
            recommended=recommended,
            standards_locked=True,
            tags=("standards", module.standard_id.lower().replace(" ", "_")),
            set_default=recommended,
        )
    return registries


_shared_registries = bootstrap_builtin_registries()
if len(_shared_registries.standards_modules) == 0:
    register_builtin_standards_modules(_shared_registries)


__all__ = [
    "C1030Module",
    "C1232Module",
    "E1218Module",
    "E1297Module",
    "E181Module",
    "E261Module",
    "LockedSetting",
    "PuIsotopicsResult",
    "PuLineObservation",
    "QAMonitor",
    "QARecord",
    "QAStatus",
    "StandardsCheck",
    "StandardsEvaluation",
    "StandardsEvaluationContext",
    "StandardsModule",
    "compute_pu_isotopics",
    "currie_mda",
    "register_builtin_standards_modules",
]
