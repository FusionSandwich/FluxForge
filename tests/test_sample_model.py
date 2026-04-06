from fluxforge.core.sample import Container, Cover, MaterialComponent, Sample


def test_sample_to_dict_roundtrip_smoke():
    sample = Sample(
        sample_id="wire1",
        geometry="wire",
        mass_g=0.12,
        density_g_cm3=8.9,
        dimensions_cm={"radius": 0.01, "length": 1.0},
        composition=[MaterialComponent("Co-59", 1.0)],
        covers=[Cover(material="Cd", thickness_cm=0.05, density_g_cm3=8.65)],
        container=Container(material="Al", thickness_cm=0.02, density_g_cm3=2.7),
        metadata={"note": "test"},
    )

    payload = sample.to_dict()
    assert payload["sample_id"] == "wire1"
    assert payload["geometry"] == "wire"
    assert payload["composition"][0]["nuclide"] == "Co-59"
    assert payload["covers"][0]["material"] == "Cd"
    assert payload["container"]["material"] == "Al"
