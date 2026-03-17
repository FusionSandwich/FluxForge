import os

issues_dir = "/filespace/s/smandych/CAE/projects/ALARA/FluxForge/examples/RAFM_irradiation/issues"

gui_issues = [
    # Core Framework & State
    ("GUI_Core_DearPyGui_App_Loop.md", "GUI Core: DearPyGui App Loop and Main Setup", "Establish the Dear PyGui application skeleton, viewport creation, and primary grid layout/tab navigation structure."),
    ("GUI_Core_Thread_Worker_Non_Blocking.md", "GUI Core: Asynchronous Thread Worker", "Implement a background thread/process executor to run FluxForge core functions without freezing the GUI."),
    ("GUI_Core_CLI_Macro_Transcript_Panel.md", "GUI Core: CLI Macro Transcript Logger", "Create a panel that logs every GUI action as an equivalent `fluxforge ...` CLI command and provides a 'Copy to Clipboard' function."),
    
    # Ingest & Setup
    ("GUI_Ingest_Project_Workspace_Wizard.md", "GUI Ingest: Project Workspace Wizard", "Create UI for initializing a FluxForge project directory and managing the `run_summary.json` configurations."),
    ("GUI_Ingest_Spectrum_File_Picker.md", "GUI Ingest: Spectrum File Picker & Metadata Viewer", "Implement a file picker that loads N42/SPE/IEC formats and displays parsed MCA header info (livetime, realtime, geometry)."),
    
    # Spectrum Viewer
    ("GUI_Viewer_Base_Canvas_Pan_Zoom.md", "GUI Viewer: Base Canvas and Pan/Zoom Controls", "Build a high-performance plotting canvas in DPG. Add smooth mouse pan, scroll-wheel zoom, and y-axis log/linear toggles for 16k-channel spectra."),
    ("GUI_Viewer_Multi_Spectrum_Overlay.md", "GUI Viewer: Multi-Spectrum Overlays", "Implement a 'buffer' system dropping multiple spectrum objects onto the same canvas to allow visual comparison."),
    ("GUI_Viewer_Interactive_Cursor_Readout.md", "GUI Viewer: Interactive Cursor Readout", "Render a vertical cursor line tied to mouse X position that displays interpolated Energy (keV), Channel, and Counts."),
    ("GUI_Viewer_Selectable_Point_Annotations.md", "GUI Viewer: Clickable Point Annotations", "Add ability to shift-click peaks to spawn a persistent, draggable text annotation marker."),
    ("GUI_Viewer_Manual_ROI_Click_Drag.md", "GUI Viewer: Click-and-Drag ROI Bounding Boxes", "Create interactive shade regions (ROIs) that users can draw via mouse-drag and manually resize via edge handles."),
    
    # Peak Workflows
    ("GUI_Peaks_Search_Configuration_Panel.md", "GUI Peaks: Peak Search Configuration Panel", "Implement control panel inputs for SNR threshold, derivative/top-hat toggles, and 'Find Peaks' execution button."),
    ("GUI_Peaks_Linear_Continuum_UI.md", "GUI Peaks: Linear Continuum Visualization", "Ensure that background continuum subtraction bounds (the ROI edges) are visibly drawn as a baseline under the raw counts."),
    ("GUI_Peaks_Multiplet_Deconvolution_Plotting.md", "GUI Peaks: Multiplet Deconvolution Display", "When multiple gaussian fits run in a single ROI, plot the total fit line and overlapping individual gaussian sub-peaks."),
    ("GUI_Peaks_Results_Data_Grid.md", "GUI Peaks: Peak Results Data Grid Table", "Build a sortable DPG table presenting peak centroid (keV), FWHM, gross/net counts, uncertainty, and significance mapping."),
    
    # Calibration Workflows
    ("GUI_Calib_Energy_Sliders_Fit_View.md", "GUI Calibration: Interactive Energy Calibration Sliders", "UI component allowing rapid shifting of intercept/slope calibration sliders alongside a linear regression scatter plot view."),
    ("GUI_Calib_Efficiency_Standard_Lines.md", "GUI Calibration: Efficiency Library Picker", "Display UI for loading reference sources (e.g. Eu152/Ba133), matching them to found peaks, and plotting the log-log derived efficiency curve."),
    ("GUI_Calib_Resolution_FWHM_Curve.md", "GUI Calibration: Resolution FWHM Curve Plotter", "Provide a dedicated plotting panel to review FWHM vs. Energy behavior and approve the resolution polynomial."),
    
    # Validation & Uncertainty
    ("GUI_Validation_Calculated_Expected_Parity.md", "GUI Validation: Calculated vs. Expected Parity Plot", "Implement x=y parity scatter plots grouping isotopes alongside their C/E ratio deviation bands."),
    ("GUI_Validation_Covariance_Heatmap.md", "GUI Validation: Covariance Correlation Heatmap", "Render 2D color heatmaps for visualizing nuclear data and activity covariance matrices."),
    
    # Pipelines
    ("GUI_Pipeline_Irradiation_History_Forms.md", "GUI Pipeline: Irradiation History Inputs", "Form setup mapping multi-segment timing (Start, Stop, Power) arrays into `IrradiationSegment` objects."),
    ("GUI_Pipeline_Reaction_Rate_Table.md", "GUI Pipeline: Activity to Reaction Rate Translator UI", "Add tabulated conversion fields linking found isotope mass constraints, half-lives, and calculated saturated SigPhi values."),
    ("GUI_Pipeline_Unfold_Solver_Config.md", "GUI Pipeline: Unfolding Solver Launch Config", "Build RMLE/MLEM/STAYSL radio buttons, iteration inputs, and real-time chi-squared convergence log streams."),
    
    # Batch & Export
    ("GUI_Batch_Conversion_Summing_UI.md", "GUI Batch: Spectrum Summing and Conversion", "UI queue letting users select folders of multiple raw files to append/sum into combined single outputs automatically."),
    ("GUI_Export_HTML_PDF_Bundle.md", "GUI Export: Run Bundle PDF/HTML Report Generator", "Integrate a 'Generate Report' button that compiles the GUI's tables, current canvas view states, and input params into an exported artifact."),
    
    # Deployment
    ("GUI_Packaging_PyInstaller_Release.md", "GUI Packaging: Spec and Build Scripts", "Finalize `build.py` specs utilizing PyInstaller/cx_Freeze to output click-to-run Linux AppImages and Windows .exe blobs.")
]

start_num = 82

for i, (filename, title, desc) in enumerate(gui_issues):
    num = start_num + i
    full_name = f"{num:02d}_{filename}"
    path = os.path.join(issues_dir, full_name)
    with open(path, "w") as f:
        f.write(f"# Issue: {title}\n\n")
        f.write("**Status:** Planned\n")
        f.write("**Context:** GUI Implementation Plan - Granular Task\n\n")
        f.write("## Description\n")
        f.write(f"{desc}\n")
    print(f"Created {full_name}")

