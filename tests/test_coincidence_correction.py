import pytest

from fluxforge.corrections.coincidence import CoincidenceCorrector


class _Eff:
    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def evaluate(self, energy_keV: float) -> float:
        # Simple lookup; default small efficiency
        return float(self._mapping.get(round(energy_keV, 1), 0.0))


def test_co60_coincidence_correction_uses_neighbor_total_efficiency():
    eff = _Eff({1332.5: 0.10, 1173.2: 0.20})
    corrector = CoincidenceCorrector(efficiency_curve=eff)

    # Co-60 correction accounts for branch intensity (~0.9998)
    # Factor = 1/(1 - ε×I) ≈ 1/(1 - 0.10×0.9998) for 1173 keV
    c1173 = corrector.calculate_correction("Co-60", 1173.2)
    assert c1173.factor == pytest.approx(1.0 / (1.0 - 0.10), rel=1e-3)

    c1332 = corrector.calculate_correction("Co60", 1332.5)
    assert c1332.factor == pytest.approx(1.0 / (1.0 - 0.20), rel=1e-3)


def test_unknown_isotope_defaults_to_no_correction():
    corrector = CoincidenceCorrector(efficiency_curve=None)
    corr = corrector.calculate_correction("Na-24", 1368.6)
    assert corr.factor == 1.0
    assert corr.uncertainty == 0.0
