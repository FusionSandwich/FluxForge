import os
import glob

issues_dir = "/filespace/s/smandych/CAE/projects/ALARA/FluxForge/examples/RAFM_irradiation/issues"

# Find max ID
existing_files = glob.glob(os.path.join(issues_dir, "*.md"))
max_num = 0
for f in existing_files:
    basename = os.path.basename(f)
    if basename[0].isdigit():
        try:
            num = int(basename.split('_')[0])
            max_num = max(max_num, num)
        except ValueError:
            pass

start_num = max_num + 1

standards = [
    {
        "name": "ASTM E261 and E844 (Activation-Foil Dosimetry)",
        "tasks": [
            ("Implementation", "ASTM_E261_E844_Implementation.md", "Implement ASTM E261/E844 standards for activation-foil dosimetry workflows.", "Target location: `src/fluxforge/physics/activation.py` and `src/fluxforge/dosimetry_metadata.py`.\n- Design rigorous detector specimen representations (foil/wire composition, geometry, mass, isotopic composition assumptions).\n- Build tracking logic for irradiation history and post-irradiation handling metadata (linking E844 surveillance rules).\n- Implement precise reaction rate calculations mapped with strict uncertainty formulations as mandated by E261 for spectral/fluence generation upstream of unfolding."),
            ("Testing", "ASTM_E261_E844_Testing.md", "Test suites for strict E261 reaction rates and E844 metadata handling.", "Target location: `tests/test_astm_dosimetry.py`.\n- Add mock scenarios for boundary condition testing of foil reaction rates encompassing uncertainty propagation paths.\n- Implement schema validation that ensures an instantiated E844 surveillance dosimeter set cannot be parsed without the correct metadata standards acting as QC traces."),
            ("Example", "ASTM_E261_E844_Example.md", "Example script for a reactor surveillance dosimeter set.", "Target location: `examples/ASTM_E844_reactor_surveillance.py`.\n- Demonstrate creating a multi-foil/wire set array.\n- Emulate injecting detailed E844 QC tagging properties to each wire.\n- Generate E261 precise reaction rates and output data mimicking a completed report matrix intended for downstream solvers.")
        ]
    },
    {
        "name": "IEC 61452, ASTM E3376, ASTM E181 (HPGe Calibration_Spectrometry)",
        "tasks": [
            ("Implementation", "IEC61452_ASTM_E3376_Implementation.md", "Create HPGe calibration & correction architectures conforming to IEC 61452 and ASTM E3376.", "Target location: `src/fluxforge/analysis/detector_calibration.py` and `src/fluxforge/corrections/`.\n- Design a primary calibration subsystem incorporating IEC guidelines for accurate energy drift models, full-energy peak efficiency fitting, and FWHM resolution definitions.\n- Stand up rigorous physics-informed modules handling: dead time loss, pile-up/random-summing behavior, true coincidence summing (TCS), decay/cooling timing math, and geometry/efficiency transfer modifications."),
            ("Testing", "IEC61452_ASTM_E3376_Testing.md", "Testing frameworks validating HPGe models against analytical tolerances.", "Target location: `tests/test_detector_calibration.py` and `tests/test_iec_corrections.py`.\n- Construct matrix scenarios that assert full-energy peak efficiencies exactly match baseline IEC reference values.\n- Verify dead-time correction limits and coincidence matrix values do not violate physical bounds during mock peak evaluation events."),
            ("Example", "IEC61452_ASTM_E3376_Example.md", "Script executing an IEC 61452 compliant benchmark calibration.", "Target location: `examples/IEC61452_hpge_calibration.py`.\n- Instantiate an un-calibrated mock spectrum.\n- Apply FWHM mapping and Efficiency standard tables according to ASTM E181/E3376 requirements.\n- Apply standard corrections representing pile-up limits and output reproducible validation graphs mapping the calibration functions over energy.")
        ]
    },
    {
        "name": "IAEA TECDOC-2026 (k0-NAA Method Expectations)",
        "tasks": [
            ("Implementation", "IAEA_TECDOC2026_Implementation.md", "Construct k0-NAA methodologies covering standard TECDOC-2026 feature lists.", "Target location: `src/fluxforge/physics/k0_naa.py` and `src/fluxforge/data/`.\n- Build the irradiation facility analysis logic for extracting ratio components f and alpha via multi-monitor procedures.\n- Add logic addressing intermittent irradiation complexities, self-shielding factors, epithermal modifiers, non-1/v target corrections, and isotope burn-up formulas.\n- Hard-code explicit tracking of deployed k0 nuclear databases ensuring data/version serialization tied strictly to TECDOC QA expectations."),
            ("Testing", "IAEA_TECDOC2026_Testing.md", "Verification of the k0-NAA computational spine against references.", "Target location: `tests/test_k0_naa_methods.py`.\n- Setup mock experiments applying standard intercomparison testing constraints directly mirrored from TECDOC-2026 baseline documents for f, alpha, and analytical limits.\n- Assess blank correction models handling non-uniform backgrounds accurately representing statistical limits."),
            ("Example", "IAEA_TECDOC2026_Example.md", "k0-NAA end-to-end routine workflow.", "Target location: `examples/TECDOC2026_k0_naa_analysis.py`.\n- Draft an example referencing classic Zr/Au foil activation.\n- Extract computational bare vs. Cd-covered calculations establishing intermediate f/alpha vectors.\n- Process the final sample mass fraction derivations using strictly standard k0 constants from the registered dataset.")
        ]
    },
    {
        "name": "IAEA TRS 487 (QA_QC in NAA)",
        "tasks": [
            ("Implementation", "IAEA_TRS487_Implementation.md", "Embed Quality Assurance metrics mimicking TRS-487 institutional protocols.", "Target location: `src/fluxforge/qaqc/` and core execution logs.\n- Build classes managing traceability of spectrum files referencing equipment calibration dates and parameters.\n- Insert explicit controls over software library versioning stamps in derived outputs, logging blank data interactions, and interference alerts (primary/fission/threshold).\n- Standardize QA/QC error trapping events preventing un-verified user states."),
            ("Testing", "IAEA_TRS487_Testing.md", "QA/QC state validation coverage.", "Target location: `tests/test_qaqc_compliance.py`.\n- Assert that executing calculations injects mandatory provenance trace IDs seamlessly into the outputs.\n- Force negative testing via deprecated data streams ensuring error prevention layers log correct TRS-487 styled compliance violations."),
            ("Example", "IAEA_TRS487_Example.md", "IAEA quality assurance report generation.", "Target location: `examples/IAEA_TRS487_QA_Report.py`.\n- Build an orchestration script executing a generalized comparator INAA pipeline while recording standard QA/QC milestones natively.\n- Dump an institutional mock JSON/PDF log certifying the data sets conform to expectations set by IAEA TRS 487 ISO/IEC-17025 style rules.")
        ]
    }
]

for std in standards:
    for task_type, filename, desc, implementation in std["tasks"]:
        num = start_num
        start_num += 1
        full_name = f"{num:02d}_{filename}"
        path = os.path.join(issues_dir, full_name)
        
        with open(path, "w") as f:
            f.write(f"# Issue: {std['name']} - {task_type}\n\n")
            f.write("**Status:** Planned\n")
            f.write(f"**Context:** Deep Research Report Standardization (`{std['name']}`)\n\n")
            f.write("## Description\n")
            f.write(f"{desc}\n\n")
            f.write("## Implementation / Details\n")
            f.write(f"{implementation}\n")
        print(f"Created {full_name}")

