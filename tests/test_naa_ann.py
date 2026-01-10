"""
Tests for NAA-ANN (Artificial Neural Network for NAA) module.

These tests verify the optional TensorFlow-based neural network
functionality for element quantification from gamma spectra.
"""

import pytest
import numpy as np

# Import common utilities first
from fluxforge.analysis.naa_ann import (
    HAS_TENSORFLOW,
    get_tensorflow_info,
    NAAANNConfig,
    AugmentationConfig,
    SpectralAugmentor,
    NAAANNResult,
)


class TestTensorFlowAvailability:
    """Test TensorFlow availability and configuration."""
    
    def test_tensorflow_info(self):
        """Test that TensorFlow info returns valid structure."""
        info = get_tensorflow_info()
        
        assert isinstance(info, dict)
        assert 'installed' in info
        assert 'version' in info
        assert 'gpu_available' in info
        assert 'gpu_devices' in info
        
        if HAS_TENSORFLOW:
            assert info['installed'] is True
            assert info['version'] is not None
        else:
            assert info['installed'] is False
    
    def test_has_tensorflow_flag(self):
        """Test that HAS_TENSORFLOW flag is set correctly."""
        assert isinstance(HAS_TENSORFLOW, bool)


class TestDataClasses:
    """Test data classes (work without TensorFlow)."""
    
    def test_naa_ann_config_defaults(self):
        """Test NAAANNConfig has sensible defaults."""
        config = NAAANNConfig()
        
        assert config.n_channels == 8192
        assert config.n_outputs == 3
        assert config.n_embeddings == 10
        assert config.n_patches == 3
        assert config.activation == 'gelu'
        assert config.dropout_rate == 0.01
        assert config.epochs == 2000
        assert config.patience == 120
    
    def test_naa_ann_config_custom(self):
        """Test NAAANNConfig with custom values."""
        config = NAAANNConfig(
            n_channels=4096,
            n_outputs=2,
            epochs=100,
        )
        
        assert config.n_channels == 4096
        assert config.n_outputs == 2
        assert config.epochs == 100
    
    def test_augmentation_config_defaults(self):
        """Test AugmentationConfig defaults."""
        config = AugmentationConfig()
        
        assert config.n_synthetic == 50
        assert config.noise_scale == 1.0
        assert config.energy_shift_max == 2.0
    
    def test_naa_ann_result(self):
        """Test NAAANNResult dataclass."""
        result = NAAANNResult(
            element='Se',
            concentration=0.5,
            uncertainty=0.05,
            detection_limit=0.01,
            confidence=0.95,
            spectrum_id='test_001',
        )
        
        assert result.element == 'Se'
        assert result.concentration == 0.5
        assert result.uncertainty == 0.05
        assert result.detection_limit == 0.01
        assert result.confidence == 0.95


class TestSpectralAugmentor:
    """Test spectral augmentation (works without TensorFlow)."""
    
    def test_augmentor_creation(self):
        """Test SpectralAugmentor creation."""
        augmentor = SpectralAugmentor()
        assert augmentor.config is not None
    
    def test_augmentor_with_config(self):
        """Test SpectralAugmentor with custom config."""
        config = AugmentationConfig(n_synthetic=10)
        augmentor = SpectralAugmentor(config)
        assert augmentor.config.n_synthetic == 10
    
    def test_augment_spectrum(self):
        """Test generating synthetic spectra."""
        config = AugmentationConfig(n_synthetic=5)
        augmentor = SpectralAugmentor(config)
        augmentor.set_seed(42)
        
        # Create a test spectrum with some peaks
        spectrum = np.zeros(1024)
        spectrum[100:110] = 500  # Peak 1
        spectrum[300:310] = 1000  # Peak 2
        spectrum += np.random.poisson(50, 1024)  # Background
        
        synthetic = augmentor.augment_spectrum(spectrum)
        
        assert len(synthetic) == 5
        for synth in synthetic:
            assert len(synth) == 1024
            assert synth.min() >= 0  # Non-negative counts
    
    def test_augmentation_variability(self):
        """Test that augmentation produces different spectra."""
        config = AugmentationConfig(n_synthetic=3, noise_scale=1.0)
        augmentor = SpectralAugmentor(config)
        augmentor.set_seed(42)  # Set seed for reproducibility
        
        spectrum = np.random.poisson(100, 512).astype(float)
        synthetic = augmentor.augment_spectrum(spectrum)
        
        # Check that synthetics are different from original
        for synth in synthetic:
            # Should not be exactly the same as original
            assert not np.allclose(synth, spectrum, rtol=0.01)
    
    def test_augment_dataset(self):
        """Test dataset augmentation."""
        config = AugmentationConfig(n_synthetic=3)
        augmentor = SpectralAugmentor(config)
        augmentor.set_seed(42)
        
        # Original dataset: 2 spectra, 3 labels each
        spectra = [
            np.random.poisson(100, 256),
            np.random.poisson(150, 256),
        ]
        labels = np.array([
            [1.0, 0.1, 0.01],
            [2.0, 0.2, 0.02],
        ])
        
        X, y, _ = augmentor.augment_dataset(spectra, labels)
        
        # 2 original + 2*3 synthetic = 8 total
        assert X.shape[0] == 8
        assert y.shape[0] == 8
        assert X.shape[1] == 256
        assert y.shape[1] == 3


# Tests requiring TensorFlow
@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestNAAANNModel:
    """Test NAA-ANN neural network model (requires TensorFlow)."""
    
    def test_import_tensorflow_components(self):
        """Test that TensorFlow components can be imported."""
        from fluxforge.analysis.naa_ann import NAAANNModel, configure_gpu
        assert NAAANNModel is not None
        assert configure_gpu is not None
    
    def test_configure_gpu(self):
        """Test GPU configuration function."""
        from fluxforge.analysis.naa_ann import configure_gpu
        
        config = configure_gpu(memory_growth=True)
        
        assert isinstance(config, dict)
        assert 'available' in config
        assert 'devices' in config
        assert 'configured' in config
    
    def test_model_creation(self):
        """Test NAAANNModel creation."""
        from fluxforge.analysis.naa_ann import NAAANNModel
        
        config = NAAANNConfig(n_channels=512, n_outputs=3)
        model = NAAANNModel(config, element='Se')
        
        assert model.element == 'Se'
        assert model.config.n_channels == 512
        assert model.model is None  # Not built yet
    
    def test_model_build(self):
        """Test NAAANNModel building."""
        from fluxforge.analysis.naa_ann import NAAANNModel
        
        config = NAAANNConfig(n_channels=256, n_outputs=3, n_patches=2)
        model = NAAANNModel(config, element='Test')
        model.build()
        
        assert model.model is not None
        assert model._is_built is True
        # Check model has parameters
        assert model.model.count_params() > 0
    
    def test_model_predict_shape(self):
        """Test model prediction shape."""
        from fluxforge.analysis.naa_ann import NAAANNModel
        
        config = NAAANNConfig(n_channels=256, n_outputs=3, n_patches=2)
        model = NAAANNModel(config, element='Test')
        model.build()
        
        # Set normalization params for postprocessing
        model.normalization_params = {'y_min': np.array([0, 0, 0]), 'y_max': np.array([1, 1, 1])}
        
        # Test prediction
        test_spectra = np.random.poisson(100, (5, 256)).astype(float)
        predictions = model.predict(test_spectra)
        
        assert predictions.shape == (5, 3)
    
    def test_preprocess_spectra(self):
        """Test spectrum preprocessing."""
        from fluxforge.analysis.naa_ann import NAAANNModel
        
        config = NAAANNConfig(n_channels=256, peak_cap=1000)
        model = NAAANNModel(config)
        
        # Create spectrum with very high peak and some baseline
        spectra = np.ones((2, 256)) * 10  # Small baseline
        spectra[0, 100] = 5000  # Above cap
        spectra[0, 50] = 500    # Another peak
        spectra[1, 100] = 500   # Below cap
        
        processed = model.preprocess_spectra(spectra)
        
        # Check normalization (max should be ~1)
        assert np.allclose(processed.max(axis=1), 1.0, atol=0.01)
        # Both spectra should have positive values
        assert processed.min() >= 0
    
    def test_analyze_spectrum(self):
        """Test single spectrum analysis."""
        from fluxforge.analysis.naa_ann import NAAANNModel
        
        config = NAAANNConfig(n_channels=256, n_outputs=3, n_patches=2)
        model = NAAANNModel(config, element='Selenium')
        model.build()
        model.normalization_params = {'y_min': np.array([0, 0, 0]), 'y_max': np.array([1, 0.5, 0.1])}
        
        spectrum = np.random.poisson(100, 256).astype(float)
        result = model.analyze_spectrum(spectrum, spectrum_id='test_001')
        
        assert isinstance(result, NAAANNResult)
        assert result.element == 'Selenium'
        assert result.spectrum_id == 'test_001'
        assert isinstance(result.concentration, float)
        assert isinstance(result.uncertainty, float)
        assert isinstance(result.detection_limit, float)


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestNAAANNTraining:
    """Test NAA-ANN training pipeline (requires TensorFlow)."""
    
    def test_short_training(self):
        """Test a very short training run."""
        from fluxforge.analysis.naa_ann import NAAANNModel
        
        # Very small model for fast testing
        config = NAAANNConfig(
            n_channels=128,
            n_outputs=3,  # Match default loss_weights which has 3 elements
            n_patches=1,
            n_embeddings=4,
            n_hidden=8,
            epochs=3,
            patience=5,
            batch_size=4,
        )
        
        model = NAAANNModel(config, element='TestElement')
        model.build()
        
        # Create tiny dataset with clear pattern
        n_train = 20
        X_train = np.random.poisson(100, (n_train, 128)).astype(float)
        y_train = np.random.rand(n_train, 3) * 0.5 + 0.25  # 3 outputs
        
        # Train (very short) - no validation for super short runs
        history = model.train(X_train, y_train)
        
        assert 'loss' in history
        assert len(history['loss']) > 0


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestNAAANNAnalyzer:
    """Test high-level NAA-ANN analyzer (requires TensorFlow)."""
    
    def test_analyzer_creation(self):
        """Test NAAANNAnalyzer creation."""
        from fluxforge.analysis.naa_ann import NAAANNAnalyzer
        
        analyzer = NAAANNAnalyzer()
        assert analyzer.models == {}
    
    def test_analyzer_batch_analyze_empty(self):
        """Test batch analyze with no models."""
        from fluxforge.analysis.naa_ann import NAAANNAnalyzer
        
        analyzer = NAAANNAnalyzer()
        
        spectra = [np.random.poisson(100, 256) for _ in range(3)]
        results = analyzer.batch_analyze(spectra)
        
        assert len(results) == 3
        assert all(r == {} for r in results)  # No models loaded


class TestNAAANNIntegration:
    """Integration tests for NAA-ANN module."""
    
    def test_augmentation_to_training_pipeline(self):
        """Test that augmentation output is compatible with model input."""
        if not HAS_TENSORFLOW:
            pytest.skip("TensorFlow not installed")
        
        from fluxforge.analysis.naa_ann import (
            NAAANNModel, create_training_dataset
        )
        
        # Create small dataset
        spectra = [np.random.poisson(100, 256) for _ in range(10)]
        labels = np.random.rand(10, 3)
        
        X_train, y_train, X_test, y_test = create_training_dataset(
            spectra, labels, 'Test',
            augmentation_config=AugmentationConfig(n_synthetic=2),
            test_split=0.2,
            seed=42,
        )
        
        # Check shapes
        assert X_train.shape[1] == 256
        assert y_train.shape[1] == 3
        
        # Create model with matching channels
        config = NAAANNConfig(n_channels=256, n_outputs=3, n_patches=2)
        model = NAAANNModel(config, element='Test')
        model.build()
        
        # Should be able to preprocess
        processed = model.preprocess_spectra(X_train)
        assert processed.shape == X_train.shape
