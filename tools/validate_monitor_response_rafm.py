"""Offline, additive #208/#209 compatibility replay using source-joined RAFM data.

This is a software receipt, not scientific admission. Requires existing evaluated
archives and the external audit CSV; does not download or edit source inputs.
Run under PYTHONPATH=<checkout>/src, with --checkout identifying that revision.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--join", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-repaired", action="store_true")
    args = parser.parse_args()
    import fluxforge.physics.monitor_response as physical
    from fluxforge.data.irdff import (
        DEFAULT_CACHE_DIR,
        IRDFFDatabase,
        IRDFF_TAB_ARCHIVE_NAME,
        IRDFF_ABS_ARCHIVE_NAME,
    )
    from fluxforge.examples.rafm_workflow import (
        load_rafm_example_metadata,
        build_flux_wire_reactions,
        TimingInfo,
        cd_cover_from_config,
        normalize_pairing_key,
    )
    from fluxforge.io.flux_wire import read_processed_txt
    from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

    assert Path(physical.__file__).resolve().is_relative_to(args.checkout.resolve())
    example = args.root / "examples/RAFM_irradiation"
    # Use the reviewed metadata of the code checkout, and actual original source bytes.
    metadata = load_rafm_example_metadata(args.checkout / "examples/RAFM_irradiation")
    archive = DEFAULT_CACHE_DIR / "tab" / IRDFF_TAB_ARCHIVE_NAME
    absorption = archive.with_name(IRDFF_ABS_ARCHIVE_NAME)
    if not archive.is_file() or not absorption.is_file():
        raise FileNotFoundError(
            "Existing evaluated reaction AND absorption archives required; no downloads"
        )
    db = IRDFFDatabase(
        auto_download=False, archive_path=archive, expected_archive_sha256=sha(archive)
    )
    inputs = {
        str(p.resolve()): sha(p)
        for p in [
            archive,
            absorption,
            args.join,
            args.root
            / "artifacts/manual_review/rafm_validation_full/unfolding/gls.json",
            args.checkout / "docs/reviews/INL_MONITOR_MASS_REVIEW_2026-09-29.md",
            *sorted(
                (args.checkout / "examples/RAFM_irradiation/metadata").glob("*.json")
            ),
        ]
    }
    records = list(csv.DictReader(args.join.open(encoding="utf-8", newline="")))
    # Co/Sc bare+Cd pairs and three actual Ti specimens (three reactions each).
    selected = [
        r
        for r in records
        if r["gls_row"] != "excluded"
        and (
            r["reaction_id"] in {"Co-59(n,g)Co-60", "Sc-45(n,g)Sc-46"}
            or r["reaction_id"].startswith("Ti-")
        )
    ]
    assert len(selected) == 13
    historical = json.loads(
        (
            args.root
            / "artifacts/manual_review/rafm_validation_full/unfolding/gls.json"
        ).read_text()
    )
    edges = np.array(historical["energy_edges_eV"])
    specs, reactions, sources = [], [], []
    for index, record in enumerate(selected):
        for stem in ["raw_asc", "qg_report", "saved_analysis_json"]:
            path = Path(record[f"{stem}_path"])
            digest = sha(path)
            assert digest == record[f"{stem}_sha256"], f"Source join mismatch: {path}"
            inputs[str(path.resolve())] = digest
        saved = json.loads(Path(record["saved_analysis_json_path"]).read_text())
        isotope = record["isotope"]
        payload = saved["isotopes"][isotope]
        qg = read_processed_txt(
            Path(record["qg_report_path"]), profile_name="rafm_25cm"
        )
        nuclide = next(n for n in qg.nuclides if n.isotope == isotope)
        assert np.isclose(
            nuclide.activity_bq, float(record["qg_report_activity_Bq"]), rtol=1e-12
        )
        assert np.isclose(nuclide.activity_bq, payload["activity_bq"], rtol=1e-12)
        assert np.isclose(
            payload["activity_eoi_bq"], float(record["eoi_activity_Bq"]), rtol=1e-12
        )
        timing = TimingInfo(
            "flux_wires",
            False,
            "source_join",
            None,
            float(record["irradiation_time_s"]),
            float(record["decay_time_s"]),
            None,
            None,
            str(args.join),
        )
        sample = record["sample_id"]
        key = normalize_pairing_key(sample, metadata.pairing_aliases)
        reaction = build_flux_wire_reactions(
            sample, key, {isotope: payload}, timing, metadata
        )[0]
        assert reaction.reaction_id == record["reaction_id"]
        assert reaction.reaction_rate > 0 and reaction.reaction_rate_unc > 0
        covered = "-cd-" in sample.lower()
        spec = physical.MonitorResponseSpec(
            f"{sample}|{reaction.reaction_id}",
            sample,
            reaction.reaction_id,
            cover=cd_cover_from_config(metadata.config) if covered else None,
        )
        specs.append(spec)
        reactions.append(reaction)
        sources.append(
            dict(
                observation_id=spec.observation_id,
                sample_id=sample,
                isotope=isotope,
                reaction=spec.reaction,
                covered=covered,
                target_atoms=reaction.n_atoms,
                saved_rate=float(record["saved_rate_per_atom_s"]),
                replayed_rate=reaction.reaction_rate,
                replayed_rate_unc=reaction.reaction_rate_unc,
                rate_uncertainty_budget=asdict(reaction.uncertainty_budget),
            )
        )

    rows, unc, response_records = physical.build_monitor_response_matrix(
        specs, edges, db
    )
    rates = np.array([r.reaction_rate for r in reactions])
    rate_unc = np.array([r.reaction_rate_unc for r in reactions])
    differential_response = rows * np.diff(edges) * 1e-24
    # Explicit diagnostic prior, equal integral flux per group; no reactor prior admission.
    group_integral_prior = np.full(len(edges) - 1, 1e11 / (len(edges) - 1))
    differential_prior = group_integral_prior / np.diff(edges)
    forward = rows @ group_integral_prior * 1e-24
    unfolder = SpectrumUnfolder(custom_energy_edges=edges, verbose=False)
    unfolder.irdff_db = db
    for spec, r in zip(specs, reactions):
        unfolder.add_reaction(
            spec.reaction,
            r.reaction_rate,
            r.reaction_rate_unc,
            rate_per_atom=r.reaction_rate,
            sample_id=r.sample_id,
            cover="Cd" if spec.cover else None,
            response_spec=spec,
        )
    unfolder.set_initial_guess(
        differential_prior, source="diagnostic equal integral group prior"
    )
    aggregate_kwargs = dict(row_keys=[s.physics_key for s in specs])
    signature = inspect.signature(unfolder._aggregate_duplicate_reaction_rows)
    if "observation_ids" in signature.parameters:
        aggregate_kwargs["observation_ids"] = [s.observation_id for s in specs]
    agg = unfolder._aggregate_duplicate_reaction_rows(
        differential_response,
        [s.reaction for s in specs],
        rates,
        rate_unc,
        unc * np.diff(edges) * 1e-24,
        **aggregate_kwargs,
    )
    assert len(agg["response"]) == 7  # 4 distinct Co/Sc + 3 compatible Ti operators
    workflows = {}
    for method in ["GRAVEL", "MLEM"]:
        for aggregate in [False, True]:
            result = unfolder.unfold(
                method=method,
                max_iterations=30,
                aggregate_duplicate_reactions=aggregate,
            )
            workflows[f"{method}_aggregate_{aggregate}"] = dict(
                response=result.response_matrix.tolist(),
                rates=result.measured_rates.tolist(),
                flux=result.flux.tolist(),
                predictions=(result.response_matrix @ result.flux).tolist(),
                memberships=result.metadata.get("duplicate_reaction_memberships"),
                original_rows=result.metadata["duplicate_reaction_original_rows"],
                output_rows=result.metadata["duplicate_reaction_aggregated_rows"],
            )
    # Change only copies of an actual covered Co response; never edit source data.
    co_cd = next(s for s in specs if s.reaction.startswith("Co-") and s.cover)
    from dataclasses import replace

    variants = []
    for change in [
        dict(density_g_cm3=4.325),
        dict(atomic_mass=120.0),
        dict(thickness_cm=np.nextafter(co_cd.cover.thickness_cm, 1.0)),
        dict(thickness_unc_cm=0.001),
    ]:
        spec = replace(
            co_cd, observation_id="variant", cover=replace(co_cd.cover, **change)
        )
        pair = [physical.build_monitor_response(s, edges, db) for s in [co_cd, spec]]
        group = unfolder._aggregate_duplicate_reaction_rows(
            np.array([p.group_cross_section_barn for p in pair]),
            [co_cd.reaction] * 2,
            rates[:2],
            rate_unc[:2],
            np.array([p.group_uncertainty_barn for p in pair]),
            row_keys=[co_cd.physics_key, spec.physics_key],
        )
        variants.append(
            dict(
                change=change,
                equal_keys=co_cd.physics_key == spec.physics_key,
                output_rows=len(group["response"]),
                response=[p.group_cross_section_barn.tolist() for p in pair],
                uncertainty=[p.group_uncertainty_barn.tolist() for p in pair],
            )
        )
    invalid = []
    for field in ["thickness_cm", "density_g_cm3", "atomic_mass", "thickness_unc_cm"]:
        for label, value in [
            ("nan", float("nan")),
            ("inf", float("inf")),
            ("negative", -1.0),
        ]:
            # Deliberately bypass the frozen constructor to challenge the build boundary.
            bad_cover = replace(co_cd.cover)
            object.__setattr__(bad_cover, field, value)
            bad = replace(co_cd, cover=bad_cover)
            probe = SpectrumUnfolder(custom_energy_edges=edges, verbose=False)
            probe.irdff_db = db
            probe.add_reaction(
                bad.reaction,
                rates[0],
                rate_unc[0],
                rate_per_atom=rates[0],
                response_spec=bad,
            )
            try:
                probe.unfold(method="MLEM", max_iterations=2)
            except ValueError as exc:
                invalid.append(
                    dict(field=field, value=label, rejected=True, error=str(exc))
                )
            else:
                invalid.append(dict(field=field, value=label, rejected=False))
    if args.require_repaired:
        assert all(v["output_rows"] == 2 and not v["equal_keys"] for v in variants)
        assert all(v["rejected"] for v in invalid)
        assert agg["metadata"]["memberships"]
    receipt = dict(
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True
        ).strip(),
        code_files={
            str(p.relative_to(args.checkout)): sha(p)
            for p in [
                args.checkout / "src/fluxforge/physics/monitor_response.py",
                args.checkout / "src/fluxforge/workflows/spectrum_unfolding.py",
            ]
        },
        validation_script_sha256=sha(__file__),
        offline=True,
        downloads=False,
        input_sha256=inputs,
        sources=sources,
        edges_eV=edges.tolist(),
        row_cross_sections_barn=rows.tolist(),
        row_uncertainties_barn=unc.tolist(),
        physical_row_metadata=[r.metadata for r in response_records],
        group_integral_prior=group_integral_prior.tolist(),
        forward_predictions=forward.tolist(),
        aggregation=agg["metadata"],
        workflows=workflows,
        variants=variants,
        invalid_copies=invalid,
        units=dict(
            energy="eV",
            cross_section="barn",
            group_integral_flux="n/cm2/s",
            differential_flux="n/cm2/s/eV",
            rates="reactions/target_atom/s",
        ),
        assumptions=[
            "1/E within-group collapse",
            "Nominal 0.0508 cm isotropic Cd slab",
            "No body self-shielding: measured body geometry and total data unavailable",
            "Ti rows are admitted only as same-operator software replicates",
            "Activity/timing from existing QG-source joined records; current reviewed mass metadata",
            "Rate budgets retain configured model terms; full covariance is unqualified",
            "Diagnostic fixed prior and bounded 30 iterations; no scientific convergence claim",
        ],
        limitations=[
            "No measured spectrum, confidence interval or publication validation admitted",
            "As-built Cd closure, gaps, body geometry, certificates and covariance remain unqualified",
            "Historical diagnostic GLS flux is not reused; only its energy boundaries",
            "Ni-57 excluded; Cu-Cd absent raw spectrum/timing not repaired by this check",
            "Physical GLS draft PR205 is separate and not integrated here",
        ],
    )
    # Verify source bytes remained unchanged, then create a fresh additive result directory.
    assert all(sha(p) == digest for p, digest in inputs.items())
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out / "receipt.json"
    output.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "output_sha256.json").write_text(
        json.dumps({output.name: sha(output)}, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                sha256=sha(output),
                rows=len(rows),
                aggregate_rows=len(agg["response"]),
                rejected=sum(v["rejected"] for v in invalid),
                variants=[v["output_rows"] for v in variants],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
