"""Spectrum reader factory and file-type dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

from fluxforge.io.cnf import read_cnf_file
from fluxforge.io.genie import read_genie_spectrum
from fluxforge.io.hpge import read_chn_file
from fluxforge.io.n42 import read_n42_spectrum
from fluxforge.io.spc import read_spc_file
from fluxforge.io.spe import GammaSpectrum, read_spe_file
from fluxforge.io.spectrum_csv import read_spectrum_csv


SpectrumReader = Callable[[Union[str, Path]], GammaSpectrum]


def _chn_to_gamma(path: str | Path) -> GammaSpectrum:
    spectrum = read_chn_file(path)
    return GammaSpectrum(
        counts=spectrum.counts,
        channels=spectrum.channels,
        live_time=spectrum.live_time_s,
        real_time=spectrum.real_time_s,
        start_time=spectrum.start_time,
        spectrum_id=Path(path).stem,
        detector_id=spectrum.detector_id,
        calibration={
            "energy": [
                spectrum.calibration.get("offset", 0.0),
                spectrum.calibration.get("gain", 1.0),
                spectrum.calibration.get("quadratic", 0.0),
            ]
        },
        metadata=dict(spectrum.metadata),
    )


def _n42_to_gamma(path: str | Path) -> GammaSpectrum:
    return read_n42_spectrum(path).to_gamma_spectrum()


@dataclass
class SpectrumReaderFactory:
    """Extension-based spectrum reader dispatch."""

    readers: dict[str, SpectrumReader] = field(default_factory=dict)

    def register(self, suffix: str, reader: SpectrumReader) -> None:
        self.readers[suffix.lower()] = reader

    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self.readers))

    def read(self, path: str | Path) -> GammaSpectrum:
        source = Path(path)
        reader = self.readers.get(source.suffix.lower())
        if reader is None:
            raise ValueError(
                f"Unsupported spectrum file extension: {source.suffix or '<none>'}"
            )
        return reader(source)


def create_reader_factory() -> SpectrumReaderFactory:
    """Create the default reader factory."""

    factory = SpectrumReaderFactory()
    factory.register(".n42", _n42_to_gamma)
    factory.register(".xml", _n42_to_gamma)
    factory.register(".chn", _chn_to_gamma)
    factory.register(".spc", read_spc_file)
    factory.register(".cnf", read_cnf_file)
    factory.register(".asc", read_genie_spectrum)
    factory.register(".spe", read_spe_file)
    factory.register(".csv", read_spectrum_csv)
    return factory


def read_spectrum_any(path: str | Path) -> GammaSpectrum:
    """Read a spectrum using the default extension dispatch."""

    return create_reader_factory().read(path)


__all__ = [
    "SpectrumReaderFactory",
    "create_reader_factory",
    "read_spectrum_any",
]
