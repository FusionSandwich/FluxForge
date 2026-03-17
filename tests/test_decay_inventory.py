import math
from pathlib import Path

from fluxforge.physics.decay_library import DecayDataset
from fluxforge.physics.decay_inventory import DecayInventory

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_decay_inventory_matches_reference():
    path = (
        TEST_DATA_DIR
        / "radioactivedecay"
        / "icrp107_ame2020_nubase2020"
        / "decay_data.npz"
    )
    dataset = DecayDataset.from_radioactivedecay_npz(path)

    inv = DecayInventory.from_quantities({"Mo-99": 2.0}, unit="bq", dataset=dataset)
    decayed = inv.decay(20.0, units="h")
    activities = decayed.activities("bq")

    assert math.isclose(activities["Mo-99"], 1.620786, rel_tol=0.02)
    assert math.isclose(activities["Tc-99m"], 1.371983, rel_tol=0.02)
    assert math.isclose(activities["Tc-99"], 9.053e-09, rel_tol=0.02)


def test_cumulative_decays_reference():
    path = (
        TEST_DATA_DIR
        / "radioactivedecay"
        / "icrp107_ame2020_nubase2020"
        / "decay_data.npz"
    )
    dataset = DecayDataset.from_radioactivedecay_npz(path)

    inv = DecayInventory.from_quantities({"Mo-99": 2.0}, unit="bq", dataset=dataset)
    cumulative = inv.cumulative_decays(20.0, units="h")

    assert math.isclose(cumulative["Mo-99"], 129870.3165, rel_tol=0.02)
    assert math.isclose(cumulative["Tc-99m"], 71074.3192, rel_tol=0.02)
