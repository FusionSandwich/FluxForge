import os

issues_dir = "/filespace/s/smandych/CAE/projects/ALARA/FluxForge/examples/RAFM_irradiation/issues"

gui_issues = [
    (
        "GUI_UI_skeleton_and_spectrum_viewer.md",
        "GUI foundation: UI skeleton + spectrum viewer (read-only) + artifact export.",
        "Setup the Dear PyGui application skeleton, project browser, and a read-only interactive spectrum viewer with pan/zoom. Add export functionality for run bundles and annotated figures.",
    ),
    (
        "GUI_Annotations_ROI_and_peak_tools.md",
        "GUI annotations + ROI + peak tools",
        "Implement interactive tools for the spectrum viewer: point selection, manual ROI drawing (drag, resize, overlap), peak finding panel with live re-fit and background subtraction (QuantumGold parity).",
    ),
    (
        "GUI_Calibration_UI.md",
        "GUI calibration UI",
        "Create interactive calibration editing components: energy calibration slider/regression, efficiency calibration curve generation from selected reference lines, and resolution curve fitting.",
    ),
    (
        "GUI_Pipeline_tabs.md",
        "GUI pipeline tabs",
        "Build the end-to-end neutron unfolding/dosimetry views: data preparation, activity/reaction rates, k0-NAA inputs, RMLE/STAYSL solver selection, and batch execution UI.",
    ),
    (
        "GUI_Validation_dashboards.md",
        "GUI validation dashboards",
        "Add comparison and diagnostic views: C/E tables, parity plots, residual analysis, and uncertainty/covariance heatmaps.",
    ),
    (
        "GUI_Packaging_automation.md",
        "GUI packaging automation",
        "Setup cross-platform frozen binary generation (e.g. PyInstaller/cx_Freeze) for Windows and Linux, and integrate with CI for automated release builds.",
    ),
]


def get_next_num():
    files = os.listdir(issues_dir)
    max_num = 0
    for f in files:
        if f[0].isdigit():
            num = int(f.split("_")[0])
            if num > max_num:
                max_num = num
    return max_num + 1


start_num = get_next_num()

for i, (filename, title, desc) in enumerate(gui_issues):
    num = start_num + i
    full_name = f"{num:02d}_{filename}"
    path = os.path.join(issues_dir, full_name)
    with open(path, "w") as f:
        f.write(f"# Issue: {title}\n\n")
        f.write("**Status:** Planned\n")
        f.write("**Context:** GUI Implementation Plan\n\n")
        f.write("## Description\n")
        f.write(f"{desc}\n")
    print(f"Created {full_name}")
