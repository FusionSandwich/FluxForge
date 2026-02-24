from fluxforge.analysis.line_search import search_decay_lines, list_nuclide_lines


def test_search_decay_lines_cs137():
    matches = search_decay_lines(661.66, tolerance_keV=1.0, line_type="gamma")
    assert any(abs(m.energy_keV - 661.66) < 1.0 for m in matches)


def test_list_nuclide_lines_co60():
    lines = list_nuclide_lines("Co-60", line_type="gamma", min_intensity=0.5)
    energies = [round(line.energy_keV, 1) for line in lines]
    assert 1173.2 in energies
    assert 1332.5 in energies
