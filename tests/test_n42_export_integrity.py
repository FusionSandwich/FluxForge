"""Unsupported processed counts must never be silently truncated on export."""

import numpy as np
import pytest

from fluxforge.io.n42 import N42Measurement, write_n42_file


@pytest.mark.parametrize("counts", [[-1.0, 2.0], [1.5, 2.0], [np.nan], [np.inf], []])
def test_n42_rejects_unsupported_counts_before_replacing_output(tmp_path, counts):
    destination = tmp_path / "measurement.n42"
    original = b"preserved existing export"
    destination.write_bytes(original)

    with pytest.raises(ValueError, match="finite nonnegative integer counts"):
        write_n42_file(destination, N42Measurement(counts=np.asarray(counts)))

    assert destination.read_bytes() == original
