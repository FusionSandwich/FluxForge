"""Read source-bound saved Quantum settings; diagnostic, not a universal ANS parser.

Use: python inspect_saved_qg_settings.py --originals PATH_TO_originals --output NEW_JSON
Python standard library only. Originals are never modified.
The study layout is checked against rev4, channel bounds, file length/ROI records,
and report anchors. Appendix C's printed offsets are inconsistent with this layout.
Saved state does not establish the processing state used for a separately exported report.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

LAYOUT = {
    "revision": (0, "<h"),
    "live_time_s": (104, "<d"),
    "real_time_s": (96, "<d"),
    "analysis_ctrl": (860, "<H"),
    "use_library_efficiencies": (1292, "<h"),
    "roi_width_fwhm": (852, "<f"),
    "background_width_channels": (846, "<h"),
    "background_gap_fwhm": (848, "<f"),
    "first_channel": (1020, "<h"),
    "last_channel": (1022, "<h"),
    "roi_count": (1026, "<h"),
}
MANUAL_URL = "https://www-nh.scphys.kyoto-u.ac.jp/~adachi/public/zeze_high_school/QuantumMCA.pdf"

def inspect_file(path, report_dir):
    data = path.read_bytes()
    if len(data) < 1548:
        raise ValueError(f"{path.name}: truncated study header")
    values = {key: struct.unpack_from(fmt, data, offset)[0]
              for key, (offset, fmt) in LAYOUT.items()}
    if values["revision"] != 4:
        raise ValueError(f"{path.name}: unsupported revision")
    if (values["first_channel"], values["last_channel"]) != (0, 8191):
        raise ValueError(f"{path.name}: unsupported study channel layout")
    if values["roi_count"] < 0 or len(data) != 1548 + 8192 * 4 + values["roi_count"] * 50:
        raise ValueError(f"{path.name}: study header/count/ROI length does not close")
    if not 0 < values["live_time_s"] <= values["real_time_s"]:
        raise ValueError(f"{path.name}: invalid timing anchor")
    if values["analysis_ctrl"] & ~3:
        raise ValueError(f"{path.name}: unexpected analysis-control bits")
    if values["use_library_efficiencies"] not in (0, -1, 1):
        raise ValueError(f"{path.name}: invalid library-efficiency boolean")
    library = data[1294:1306].decode("ascii").strip(" \0")
    if not library.lower().endswith(".mdb"):
        raise ValueError(f"{path.name}: invalid study library anchor")
    report_path = report_dir / (path.stem + ".txt")
    report = report_path.read_text(encoding="utf-8", errors="replace") if report_path.exists() else None
    anchors = {}
    if report is not None:
        anchors["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        anchors["library_name_matches"] = bool(re.search(
            r"Library:\s*" + re.escape(library), report, flags=re.IGNORECASE))
        anchors["report_says_library_efficiencies_ignored"] = "Library efficiencies were ignored" in report
        anchors["library_efficiency_flag_matches_report"] = (
            values["use_library_efficiencies"] == 0
            if anchors["report_says_library_efficiencies_ignored"] else None)
        anchors["report_says_measurement_date"] = "Activities reported as of Measurement Date." in report
        times = re.search(r"LT:\s*([\d,]+\.\d+)\s+RT:\s*([\d,]+\.\d+)", report)
        if times:
            anchors["report_live_time_matches"] = abs(float(times[1].replace(",", "")) - values["live_time_s"]) <= 0.005
            anchors["report_real_time_matches"] = abs(float(times[2].replace(",", "")) - values["real_time_s"]) <= 0.005
        if any(v is False for k,v in anchors.items() if k.endswith("matches") or k.endswith("matches_report")):
            raise ValueError(f"{path.name}: report anchor contradiction: {anchors}")
    return {
        "source": "ANS/" + path.name, "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data), "values": values, "library_name": library,
        "saved_setting_inference": {
            "ambient_background_correction_enabled": not bool(values["analysis_ctrl"] & 1),
            "continuum_correction_enabled": not bool(values["analysis_ctrl"] & 2),
            "meaning_source": "Manufacturer Appendix C AnalysisCtrl bit meanings",
            "qualification": "SAVED_HEADER_ONLY; final report processing state not established",
        },
        "report_anchors": anchors,
        "wrong_printed_offset_counterexample": {
            "analysis_ctrl_at_printed_852": struct.unpack_from("<H", data, 852)[0],
            "use_lib_eff_at_printed_1280": struct.unpack_from("<h", data, 1280)[0],
        },
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--originals", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="New JSON file (no overwrite)")
    args = ap.parse_args()
    files = sorted((args.originals / "ANS").glob("*.ANS"))
    if not files:
        ap.error("No original ANS files found")
    rows = [inspect_file(path, args.originals / "QG_report") for path in files]
    result = {
        "status": "BOUNDED_SAVED_SETTINGS_DIAGNOSTIC",
        "manual_url": MANUAL_URL,
        "manual_version": "4.04.00; installed 2025 version not established",
        "layout_offsets": {k: {"offset":o, "format":f} for k,(o,f) in LAYOUT.items()},
        "layout_scope": "These rev4 study files; 1548-byte observed header. Not a general parser.",
        "layout_warning": "Printed Appendix C offsets fail after detector materials; observed tool fields shifted +8 and trailing descriptors/library +12. Structural and report anchors are mandatory.",
        "qualification": "Saved settings can precede report analysis. Some files have zero saved ROIs despite report peaks. Ambient-off bit does not prove the final report state or identify a background.",
        "files_checked": len(rows), "rows": rows,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "files_checked":len(rows),
        "analysis_ctrl_values":sorted({r["values"]["analysis_ctrl"] for r in rows}),
        "use_library_efficiencies":sorted({r["values"]["use_library_efficiencies"] for r in rows}),
        "report_anchors_present":sum(bool(r["report_anchors"]) for r in rows),
        "output":str(args.output)}))

if __name__ == "__main__":
    main()
