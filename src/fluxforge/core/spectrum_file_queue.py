"""Folder discovery and lossless append/sum conversion to one .ffs output."""

from copy import deepcopy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from fluxforge.io.reader_factory import create_reader_factory, read_spectrum_any
from fluxforge.io.session import session_from_spectra, write_ffs_session
from fluxforge.io.spe import GammaSpectrum


def discover_spectrum_files(folder, *, recursive=False):
    directory = Path(folder).resolve()
    if not directory.is_dir():
        raise ValueError("Select an existing input folder.")
    suffixes = set(create_reader_factory().supported_extensions())
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return tuple(
        sorted(
            (
                path.resolve()
                for path in candidates
                if path.is_file() and path.suffix.lower() in suffixes
            ),
            key=lambda path: str(path).casefold(),
        )
    )


def unique_spectrum_paths(paths):
    result, seen = [], set()
    for source in paths:
        path = Path(source).resolve()
        identity = os.path.normcase(str(path))
        if identity not in seen:
            seen.add(identity)
            result.append(path)
    return tuple(result)


def sum_queued_spectra(spectra, *, independent_acquisitions=False):
    """Sum matching bins and complete channel covariance under explicit independence.

    No spectrum is dropped or rebinned. Unknown timing remains unknown rather
    than borrowing a partial live/real-time total from other inputs.
    """
    if not spectra:
        raise ValueError("Queue at least one spectrum.")
    if independent_acquisitions is not True:
        raise ValueError(
            "Summing requires explicitly declared independent acquisitions."
        )
    base = spectra[0]
    covariance = base.count_covariance_matrix().copy()
    counts = np.array(base.counts, dtype=float, copy=True)
    semantics = (
        "counts_unit",
        "count_decay_corrected",
        "counts_reference",
        "processing",
    )
    for spectrum in spectra[1:]:
        if (
            spectrum.counts.shape != base.counts.shape
            or not np.array_equal(spectrum.channels, base.channels)
            or not np.array_equal(spectrum.energies, base.energies)
        ):
            raise ValueError(
                "Sum inputs must have identical channel and energy bins; "
                "use Append for different grids."
            )
        if spectrum.detector_id != base.detector_id or json.dumps(
            spectrum.calibration, sort_keys=True
        ) != json.dumps(base.calibration, sort_keys=True):
            raise ValueError(
                "Sum inputs must have matching detector and calibration metadata."
            )
        if any(
            spectrum.metadata.get(key) != base.metadata.get(key) for key in semantics
        ):
            raise ValueError(
                "Sum inputs have different count-reference or processing metadata."
            )
        counts += spectrum.counts
        covariance += spectrum.count_covariance_matrix()
    if counts.size == 0 or not np.all(np.isfinite(counts)):
        raise ValueError("Summed counts must be finite and nonempty.")
    timing = {}
    for name in ("live_time", "real_time"):
        values = [float(getattr(spectrum, name)) for spectrum in spectra]
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("Input acquisition times must be finite and nonnegative.")
        timing[name] = sum(values) if all(value > 0 for value in values) else 0.0
        if not np.isfinite(timing[name]):
            raise ValueError("Summed acquisition time overflowed.")
    return GammaSpectrum(
        counts=counts,
        counts_covariance=covariance,
        channels=base.channels.copy(),
        energies=None if base.energies is None else base.energies.copy(),
        live_time=timing["live_time"],
        real_time=timing["real_time"],
        start_time=None,
        spectrum_id="summed-spectrum",
        detector_id=base.detector_id,
        calibration=deepcopy(base.calibration),
        metadata={
            "operation": "sum",
            "independent_acquisitions": True,
            "input_count": len(spectra),
            "timing_complete": {name: value > 0 for name, value in timing.items()},
            "input_metadata": [deepcopy(spectrum.metadata) for spectrum in spectra],
            **{
                key: deepcopy(base.metadata[key])
                for key in semantics
                if key in base.metadata
            },
        },
    )


def convert_spectrum_file_queue(
    paths,
    output,
    *,
    mode="append",
    independent_acquisitions=False,
    progress_callback=None,
):
    """Read the whole queue before atomically writing one new native session.

    Append preserves distinct spectra in one session. Sum preserves complete
    within-spectrum covariance and assumes no cross-acquisition covariance only
    when explicitly requested. Existing files and source files are not replaced.
    """
    if mode not in {"append", "sum"}:
        raise ValueError("Choose Append or Sum.")
    sources = unique_spectrum_paths(paths)
    if not sources:
        raise ValueError("Queue at least one input file.")
    target = Path(output).resolve()
    if target.suffix.lower() != ".ffs":
        raise ValueError("Combined outputs must use the .ffs extension.")
    if target.exists() or target in sources:
        raise ValueError("Choose a new output file; existing files are preserved.")
    spectra, bindings = [], []
    for index, path in enumerate(sources, start=1):
        if not path.is_file():
            raise ValueError(f"Input is no longer a file: {path}")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        spectrum = read_spectrum_any(path)
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != after:
            raise ValueError(f"Input changed while being read: {path}")
        spectra.append(spectrum)
        bindings.append(
            {
                "path": str(path),
                "sha256": before,
                "live_time_s": spectrum.live_time,
                "real_time_s": spectrum.real_time,
            }
        )
        if progress_callback:
            progress_callback(index, len(sources))
    if mode == "sum":
        spectra = [
            sum_queued_spectra(
                spectra, independent_acquisitions=independent_acquisitions
            )
        ]
    session = session_from_spectra(
        spectra,
        source_files=sources if mode == "append" else (),
        metadata={
            "operation": mode,
            "sources": bindings,
            "independent_acquisitions": (
                independent_acquisitions if mode == "sum" else None
            ),
            "scientific_admission": False,
        },
    )
    # Recheck after processing, before the existing atomic session writer.
    if target.exists():
        raise ValueError("Output appeared during processing; choose another file.")
    # The writer validates/serializes to a private file. Link publication is
    # atomic and refuses an existing target, including a concurrent creation.
    descriptor, temporary = tempfile.mkstemp(
        suffix=".ffs", prefix=".fluxforge-queue-", dir=target.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        write_ffs_session(temporary_path, session)
        os.link(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "output": str(target),
        "mode": mode,
        "input_count": len(sources),
        "output_spectra": len(spectra),
    }
