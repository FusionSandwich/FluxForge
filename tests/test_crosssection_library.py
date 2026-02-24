import numpy as np

from fluxforge.data.crosssections import create_irdff_placeholder_library


def test_search_by_target_and_outgoing():
    library = create_irdff_placeholder_library()
    matches = library.search(target="Au-197", outgoing="g")

    assert matches
    assert any("Au-197" in xs.reaction for xs in matches)


def test_search_by_product():
    library = create_irdff_placeholder_library()
    matches = library.search(product="Au-198")

    assert matches
    assert any("Au-198" in xs.reaction for xs in matches)


def test_cross_section_evaluation():
    library = create_irdff_placeholder_library()
    xs = library.search(target="Au-197", outgoing="g")[0]
    value = xs.evaluate(1e-6)

    assert np.isfinite(value)
    assert value >= 0
