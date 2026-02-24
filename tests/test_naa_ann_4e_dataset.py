from pathlib import Path

import numpy as np
import pytest

from fluxforge.analysis.naa_ann import load_naa_ann4e_dataset, prepare_naa_ann4e_dataset


DATA_ZIP = Path(
    "testing/NAA-ANN-1/data augmentation code/versions/2022-04-27/NAA2 data augmentation output 4e.zip"
)


@pytest.mark.skipif(not DATA_ZIP.exists(), reason="NAA-ANN-1 4e dataset zip not available.")
def test_load_naa_ann4e_dataset_shapes():
    dataset = load_naa_ann4e_dataset(DATA_ZIP, max_files=5)
    assert dataset.features.shape[0] == 5
    assert dataset.labels.shape == (5, 3)
    assert dataset.label_min.shape == (3,)
    assert dataset.label_max.shape == (3,)
    assert len(dataset.sample_ids) == 5


@pytest.mark.skipif(not DATA_ZIP.exists(), reason="NAA-ANN-1 4e dataset zip not available.")
def test_prepare_naa_ann4e_dataset_normalization():
    dataset = load_naa_ann4e_dataset(DATA_ZIP, max_files=5)
    features, labels = prepare_naa_ann4e_dataset(dataset)
    assert np.all(features >= 0.0)
    assert np.nanmax(features) <= 1.0 + 1e-6
    assert np.all(labels >= 0.0)
    assert np.nanmax(labels) <= 1.0 + 1e-6
