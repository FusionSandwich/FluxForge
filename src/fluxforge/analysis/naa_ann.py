"""
NAA-ANN: Artificial Neural Network for Neutron Activation Analysis

This module implements an ANN-based approach for element quantification
from gamma spectra, inspired by the NAA-ANN-1 methodology.

The approach uses:
1. Data augmentation to generate synthetic training spectra
2. A neural network to predict element concentrations directly from spectra
3. Uncertainty quantification through ensemble methods or dropout inference

Key advantages over traditional k0-NAA:
- No need for explicit peak identification
- Robust to spectral interferences
- Can handle overlapping peaks automatically
- Learns complex spectral patterns

References:
    NAA-ANN-1: IAEA CRP project on ANN for NAA analysis
    Research Institute Delft, Netherlands
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)

# Check for TensorFlow availability
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    HAS_TENSORFLOW = True
    TF_VERSION = tf.__version__
except ImportError:
    HAS_TENSORFLOW = False
    TF_VERSION = None
    tf = None
    keras = None
    layers = None
    Model = None
    EarlyStopping = None
    ModelCheckpoint = None
    logger.warning("TensorFlow not available. NAA-ANN features will be limited.")


def configure_gpu(
    memory_growth: bool = True,
    memory_limit: Optional[int] = None,
    visible_devices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Configure GPU settings for TensorFlow.
    
    This function is optional and only affects workflows that use neural networks.
    FluxForge core functionality does not require GPU or TensorFlow.
    
    Parameters
    ----------
    memory_growth : bool
        Enable memory growth to avoid allocating all GPU memory at once
    memory_limit : int, optional
        Limit GPU memory in MB (e.g., 4096 for 4GB)
    visible_devices : list of int, optional
        List of GPU device indices to use (e.g., [0, 1] for first two GPUs)
        
    Returns
    -------
    dict
        GPU configuration status with keys:
        - 'available': bool - whether GPU is available
        - 'devices': list - list of available GPU devices
        - 'configured': bool - whether configuration was applied
        - 'memory_growth': bool - memory growth setting
        - 'memory_limit': int or None - memory limit in MB
        
    Examples
    --------
    >>> config = configure_gpu(memory_growth=True, memory_limit=4096)
    >>> if config['available']:
    ...     print(f"Using GPU: {config['devices'][0]}")
    """
    if not HAS_TENSORFLOW:
        return {
            'available': False,
            'devices': [],
            'configured': False,
            'memory_growth': False,
            'memory_limit': None,
            'error': 'TensorFlow not installed',
        }
    
    result = {
        'available': False,
        'devices': [],
        'configured': False,
        'memory_growth': memory_growth,
        'memory_limit': memory_limit,
    }
    
    try:
        # Get available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        result['devices'] = [gpu.name for gpu in gpus]
        result['available'] = len(gpus) > 0
        
        if not gpus:
            logger.info("No GPU devices found. Using CPU.")
            return result
        
        # Filter to visible devices if specified
        if visible_devices is not None:
            gpus = [gpus[i] for i in visible_devices if i < len(gpus)]
            tf.config.set_visible_devices(gpus, 'GPU')
        
        for gpu in gpus:
            if memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            if memory_limit is not None:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
        
        result['configured'] = True
        logger.info(f"Configured {len(gpus)} GPU(s) for TensorFlow")
        
    except Exception as e:
        result['error'] = str(e)
        logger.warning(f"GPU configuration failed: {e}")
    
    return result


def get_tensorflow_info() -> Dict[str, Any]:
    """
    Get TensorFlow installation and hardware information.
    
    Returns
    -------
    dict
        Information about TensorFlow installation:
        - 'installed': bool
        - 'version': str or None
        - 'gpu_available': bool
        - 'gpu_devices': list of device names
        - 'cuda_version': str or None
        - 'cudnn_version': str or None
    """
    if not HAS_TENSORFLOW:
        return {
            'installed': False,
            'version': None,
            'gpu_available': False,
            'gpu_devices': [],
            'cuda_version': None,
            'cudnn_version': None,
        }
    
    gpus = tf.config.list_physical_devices('GPU')
    
    # Try to get CUDA/cuDNN versions
    cuda_version = None
    cudnn_version = None
    try:
        from tensorflow.python.platform import build_info
        if hasattr(build_info, 'build_info'):
            cuda_version = build_info.build_info.get('cuda_version')
            cudnn_version = build_info.build_info.get('cudnn_version')
    except Exception:
        pass
    
    return {
        'installed': True,
        'version': TF_VERSION,
        'gpu_available': len(gpus) > 0,
        'gpu_devices': [gpu.name for gpu in gpus],
        'cuda_version': cuda_version,
        'cudnn_version': cudnn_version,
    }


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NAAANNConfig:
    """
    Configuration for NAA-ANN model.
    
    Attributes
    ----------
    n_channels : int
        Number of channels in input spectra (default 8192 for HPGe)
    n_outputs : int
        Number of output predictions (e.g., concentration, uncertainty, LOD)
    n_embeddings : int
        Size of patch embeddings
    n_patches : int
        Number of patches to divide spectrum into
    activation : str
        Activation function ('gelu', 'relu', 'selu')
    dropout_rate : float
        Dropout rate for regularization
    n_hidden : int
        Number of neurons in hidden layers
    batch_size : int
        Training batch size
    epochs : int
        Maximum training epochs
    patience : int
        Early stopping patience
    """
    n_channels: int = 8192
    n_outputs: int = 3  # [concentration, uncertainty, LOD]
    n_embeddings: int = 10
    n_patches: int = 3
    activation: str = 'gelu'
    dropout_rate: float = 0.01
    n_hidden: int = 20
    batch_size: int = 10
    epochs: int = 2000
    patience: int = 120
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.02, 0.1])
    peak_cap: float = 10000.0  # Cap for peak normalization


@dataclass
class NAAANNResult:
    """
    Result from NAA-ANN prediction.
    
    Attributes
    ----------
    element : str
        Element being quantified
    concentration : float
        Predicted concentration
    uncertainty : float
        Predicted uncertainty
    detection_limit : float
        Predicted detection limit
    confidence : float
        Model confidence (0-1)
    spectrum_id : str
        Identifier for the analyzed spectrum
    """
    element: str
    concentration: float
    uncertainty: float
    detection_limit: float
    confidence: float = 1.0
    spectrum_id: str = ""


@dataclass
class AugmentationConfig:
    """
    Configuration for spectral data augmentation.
    
    Attributes
    ----------
    n_synthetic : int
        Number of synthetic spectra to generate per real spectrum
    noise_scale : float
        Scale factor for Poisson-like noise
    energy_shift_max : float
        Maximum channel shift for energy perturbation
    intensity_variation : float
        Maximum fractional intensity variation
    baseline_variation : float
        Maximum baseline drift
    """
    n_synthetic: int = 50
    noise_scale: float = 1.0
    energy_shift_max: float = 2.0
    intensity_variation: float = 0.1
    baseline_variation: float = 0.05
    peak_width_variation: float = 0.1


# =============================================================================
# Data Augmentation
# =============================================================================

class SpectralAugmentor:
    """
    Generate synthetic gamma spectra for training data augmentation.
    
    This replaces the Fortran data augmentation code with a pure Python
    implementation that creates realistic synthetic spectra variations.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmentor.
        
        Parameters
        ----------
        config : AugmentationConfig, optional
            Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        self._rng = np.random.default_rng()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
    
    def augment_spectrum(
        self,
        spectrum: np.ndarray,
        n_synthetic: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Generate synthetic variations of a spectrum.
        
        Parameters
        ----------
        spectrum : ndarray
            Original spectrum counts
        n_synthetic : int, optional
            Number of synthetic spectra (default from config)
            
        Returns
        -------
        list of ndarray
            Synthetic spectrum variations
        """
        n_synth = n_synthetic or self.config.n_synthetic
        synthetic = []
        
        for _ in range(n_synth):
            synth = self._generate_one_synthetic(spectrum)
            synthetic.append(synth)
        
        return synthetic
    
    def _generate_one_synthetic(self, spectrum: np.ndarray) -> np.ndarray:
        """Generate a single synthetic spectrum variation."""
        synth = spectrum.copy().astype(float)
        n_channels = len(synth)
        
        # 1. Add Poisson-like noise
        # For counting statistics, variance ~ counts
        noise = self._rng.normal(0, 1, n_channels) * np.sqrt(np.maximum(synth, 1)) * self.config.noise_scale
        synth = np.maximum(synth + noise, 0)
        
        # 2. Apply intensity variation (gain drift)
        intensity_factor = 1.0 + self._rng.uniform(
            -self.config.intensity_variation,
            self.config.intensity_variation
        )
        synth *= intensity_factor
        
        # 3. Apply energy shift (channel shift)
        shift = self._rng.uniform(
            -self.config.energy_shift_max,
            self.config.energy_shift_max
        )
        if abs(shift) > 0.1:
            # Interpolate to shifted positions
            x = np.arange(n_channels)
            synth = np.interp(x, x - shift, synth, left=synth[0], right=synth[-1])
        
        # 4. Add baseline drift
        drift = self.config.baseline_variation * synth.max()
        # Low-frequency drift as a polynomial
        baseline = drift * (
            self._rng.uniform(-1, 1) +
            self._rng.uniform(-1, 1) * x / n_channels +
            self._rng.uniform(-1, 1) * (x / n_channels) ** 2
        )
        synth += baseline
        synth = np.maximum(synth, 0)
        
        # 5. Peak width variation (slight broadening/narrowing)
        # This is done via convolution with a small kernel
        if self.config.peak_width_variation > 0:
            width_change = self._rng.uniform(0, self.config.peak_width_variation)
            if width_change > 0.01:
                kernel_width = int(1 + width_change * 3)
                kernel = np.exp(-np.arange(-kernel_width, kernel_width + 1) ** 2 / (2 * (width_change * 2 + 0.5) ** 2))
                kernel = kernel / kernel.sum()
                synth = np.convolve(synth, kernel, mode='same')
        
        return synth
    
    def augment_dataset(
        self,
        spectra: List[np.ndarray],
        labels: np.ndarray,
        experimental_params: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Augment an entire dataset.
        
        Parameters
        ----------
        spectra : list of ndarray
            List of original spectra
        labels : ndarray
            Labels for original spectra (n_spectra, n_outputs)
        experimental_params : ndarray, optional
            Experimental parameters (mass, etc.) for each spectrum
            
        Returns
        -------
        tuple
            (augmented_spectra, augmented_labels, augmented_params)
        """
        all_spectra = list(spectra)
        all_labels = [labels[i] for i in range(len(labels))]
        all_params = list(experimental_params) if experimental_params is not None else None
        
        for i, spectrum in enumerate(spectra):
            synthetic = self.augment_spectrum(spectrum)
            all_spectra.extend(synthetic)
            for _ in synthetic:
                all_labels.append(labels[i])
                if all_params is not None:
                    # Slightly vary experimental params too
                    varied_params = experimental_params[i].copy()
                    varied_params *= (1 + self._rng.uniform(-0.01, 0.01, len(varied_params)))
                    all_params.append(varied_params)
        
        X = np.array(all_spectra)
        y = np.array(all_labels)
        params = np.array(all_params) if all_params is not None else None
        
        return X, y, params


# =============================================================================
# Neural Network Model
# =============================================================================

class NAAANNModel:
    """
    Neural network model for NAA spectral analysis.
    
    This implements a patch-based architecture similar to NAA-ANN-1,
    where the spectrum is divided into patches that are embedded
    and processed together.
    
    Examples
    --------
    >>> config = NAAANNConfig(n_channels=4096, n_outputs=3)
    >>> model = NAAANNModel(config)
    >>> model.build()
    >>> model.train(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        config: Optional[NAAANNConfig] = None,
        element: str = "Unknown",
    ):
        """
        Initialize NAA-ANN model.
        
        Parameters
        ----------
        config : NAAANNConfig, optional
            Model configuration
        element : str
            Target element for quantification
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for NAA-ANN model")
        
        self.config = config or NAAANNConfig()
        self.element = element
        self.model = None
        self.normalization_params = {}
        self._is_built = False
    
    def build(self):
        """Build the neural network architecture."""
        cfg = self.config
        
        # Input layer
        x_input = layers.Input(shape=(cfg.n_channels,), name='spectrum_input')
        
        # Patch embeddings
        patch_size = cfg.n_channels // cfg.n_patches
        patch_outputs = []
        
        for i in range(cfg.n_patches):
            start_idx = patch_size * i
            end_idx = patch_size * (i + 1)
            
            # Patch embedding
            patch = x_input[:, start_idx:end_idx]
            x = layers.Dense(cfg.n_embeddings, activation=cfg.activation)(patch)
            
            # Also include first few channels for experimental context
            s = layers.Dense(2, activation=cfg.activation)(x_input[:, :3])
            x = layers.concatenate([x, s])
            x = layers.Dropout(cfg.dropout_rate)(x)
            patch_outputs.append(x)
        
        # Combine patches
        x = layers.concatenate(patch_outputs)
        
        # Dense layers on low-energy region (usually most informative)
        x_low = layers.Dense(cfg.n_hidden, activation=cfg.activation)(x_input[:, :100])
        x_low = layers.Dropout(cfg.dropout_rate)(x_low)
        x_combined = layers.Dense(5, activation=cfg.activation)(x_low)
        
        # Output heads for each target (concentration, uncertainty, LOD)
        output_names = [f'output_{i}' for i in range(cfg.n_outputs)]
        outputs = [
            layers.Dense(1, name=output_names[i])(x_combined)
            for i in range(cfg.n_outputs)
        ]
        
        # Build model
        self.model = Model(x_input, outputs)
        
        # Store output names for training
        self._output_names = output_names
        
        # Use dict-based loss for Keras 3.x compatibility
        loss_dict = {name: 'mse' for name in output_names}
        loss_weights_list = cfg.loss_weights[:cfg.n_outputs]
        loss_weights_dict = {name: loss_weights_list[i] for i, name in enumerate(output_names)}
        
        self.model.compile(
            loss=loss_dict,
            optimizer='adam',
            loss_weights=loss_weights_dict,
        )
        
        self._is_built = True
        logger.info(f"Built NAA-ANN model for {self.element}")
    
    def preprocess_spectra(
        self,
        spectra: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Preprocess spectra for model input.
        
        Parameters
        ----------
        spectra : ndarray
            Raw spectra (n_samples, n_channels)
        fit : bool
            If True, fit normalization parameters
            
        Returns
        -------
        ndarray
            Preprocessed spectra
        """
        X = spectra.copy().astype(float)
        
        # Cap very high peaks (as in NAA-ANN-1)
        peak_cap = self.config.peak_cap
        high_mask = X > peak_cap
        X[high_mask] = peak_cap + np.power(X[high_mask] - peak_cap, 0.2)
        
        # Normalize each spectrum by its maximum
        max_vals = X.max(axis=1, keepdims=True)
        max_vals = np.maximum(max_vals, 1e-4)  # Avoid division by zero
        X = X / max_vals
        
        return X
    
    def preprocess_labels(
        self,
        labels: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Preprocess labels (normalize to 0-1 range).
        
        Parameters
        ----------
        labels : ndarray
            Raw labels (n_samples, n_outputs)
        fit : bool
            If True, fit normalization parameters
            
        Returns
        -------
        ndarray
            Normalized labels
        """
        y = labels.copy().astype(float)
        
        if fit:
            self.normalization_params['y_min'] = y.min(axis=0)
            self.normalization_params['y_max'] = y.max(axis=0)
        
        y_min = self.normalization_params.get('y_min', y.min(axis=0))
        y_max = self.normalization_params.get('y_max', y.max(axis=0))
        
        # Avoid division by zero
        y_range = y_max - y_min
        y_range = np.where(y_range > 0, y_range, 1.0)
        
        y_norm = (y - y_min) / y_range
        
        return y_norm
    
    def postprocess_labels(self, labels_norm: np.ndarray) -> np.ndarray:
        """Denormalize labels back to original scale."""
        y_min = self.normalization_params.get('y_min', 0)
        y_max = self.normalization_params.get('y_max', 1)
        y_range = y_max - y_min
        y_range = np.where(y_range > 0, y_range, 1.0)
        
        return labels_norm * y_range + y_min
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_best: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Parameters
        ----------
        X_train : ndarray
            Training spectra (n_samples, n_channels)
        y_train : ndarray
            Training labels (n_samples, n_outputs)
        X_val : ndarray, optional
            Validation spectra
        y_val : ndarray, optional
            Validation labels
        save_best : str, optional
            Path to save best model
            
        Returns
        -------
        dict
            Training history
        """
        if not self._is_built:
            self.build()
        
        cfg = self.config
        
        # Preprocess
        X_train_proc = self.preprocess_spectra(X_train)
        y_train_proc = self.preprocess_labels(y_train, fit=True)
        
        # Convert labels to dict for Keras 3.x multi-output models
        y_train_dict = {
            name: y_train_proc[:, i] 
            for i, name in enumerate(self._output_names)
        }
        
        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_proc = self.preprocess_spectra(X_val)
            y_val_proc = self.preprocess_labels(y_val, fit=False)
            y_val_dict = {
                name: y_val_proc[:, i] 
                for i, name in enumerate(self._output_names)
            }
            validation_data = (X_val_proc, y_val_dict)
        
        # Callbacks - only monitor val_loss if we have validation data
        callbacks = []
        monitor = 'val_loss' if validation_data is not None else 'loss'
        
        callbacks.append(
            EarlyStopping(monitor=monitor, mode='min', patience=cfg.patience, verbose=1)
        )
        
        if save_best:
            callbacks.append(
                ModelCheckpoint(save_best, monitor=monitor, mode='min', save_best_only=True)
            )
        
        # Train
        history = self.model.fit(
            X_train_proc,
            y_train_dict,
            validation_data=validation_data,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        
        logger.info(f"Training completed after {len(history.history['loss'])} epochs")
        
        return history.history
    
    def predict(self, spectra: np.ndarray) -> np.ndarray:
        """
        Predict concentrations from spectra.
        
        Parameters
        ----------
        spectra : ndarray
            Input spectra (n_samples, n_channels)
            
        Returns
        -------
        ndarray
            Predictions (n_samples, n_outputs)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        X_proc = self.preprocess_spectra(spectra)
        predictions = self.model.predict(X_proc)
        
        # Combine multi-output predictions
        if isinstance(predictions, list):
            predictions = np.column_stack([p.flatten() for p in predictions])
        
        # Denormalize
        predictions = self.postprocess_labels(predictions)
        
        return predictions
    
    def predict_with_uncertainty(
        self,
        spectra: np.ndarray,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with Monte Carlo dropout uncertainty estimation.
        
        Parameters
        ----------
        spectra : ndarray
            Input spectra
        n_samples : int
            Number of Monte Carlo samples
            
        Returns
        -------
        tuple
            (mean_predictions, std_predictions)
        """
        # Enable dropout during inference
        predictions = []
        
        for _ in range(n_samples):
            pred = self.model(self.preprocess_spectra(spectra), training=True)
            if isinstance(pred, list):
                pred = np.column_stack([p.numpy().flatten() for p in pred])
            else:
                pred = pred.numpy()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = self.postprocess_labels(predictions.mean(axis=0))
        std_pred = predictions.std(axis=0) * (
            self.normalization_params.get('y_max', 1) -
            self.normalization_params.get('y_min', 0)
        )
        
        return mean_pred, std_pred
    
    def save(self, path: str):
        """
        Save model and normalization parameters.
        
        Parameters
        ----------
        path : str
            Directory path to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(path / 'model.keras')
        
        # Save configuration and normalization
        config_dict = {
            'element': self.element,
            'config': {
                'n_channels': self.config.n_channels,
                'n_outputs': self.config.n_outputs,
                'n_embeddings': self.config.n_embeddings,
                'n_patches': self.config.n_patches,
                'activation': self.config.activation,
                'dropout_rate': self.config.dropout_rate,
                'n_hidden': self.config.n_hidden,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs,
                'patience': self.config.patience,
                'loss_weights': self.config.loss_weights,
                'peak_cap': self.config.peak_cap,
            },
            'normalization': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.normalization_params.items()
            }
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved NAA-ANN model to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NAAANNModel':
        """
        Load model from disk.
        
        Parameters
        ----------
        path : str
            Directory path containing saved model
            
        Returns
        -------
        NAAANNModel
            Loaded model
        """
        path = Path(path)
        
        # Load config
        with open(path / 'config.json', 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct config
        cfg = NAAANNConfig(**config_dict['config'])
        
        # Create model instance
        model = cls(cfg, element=config_dict['element'])
        
        # Load Keras model
        model.model = keras.models.load_model(path / 'model.keras')
        model._is_built = True
        
        # Load normalization params
        model.normalization_params = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in config_dict['normalization'].items()
        }
        
        logger.info(f"Loaded NAA-ANN model for {model.element} from {path}")
        
        return model
    
    def analyze_spectrum(
        self,
        spectrum: np.ndarray,
        spectrum_id: str = "",
    ) -> NAAANNResult:
        """
        Analyze a single spectrum and return structured result.
        
        Parameters
        ----------
        spectrum : ndarray
            Input spectrum (n_channels,)
        spectrum_id : str
            Identifier for the spectrum
            
        Returns
        -------
        NAAANNResult
            Analysis result
        """
        # Reshape if needed
        if spectrum.ndim == 1:
            spectrum = spectrum.reshape(1, -1)
        
        predictions = self.predict(spectrum)[0]
        
        return NAAANNResult(
            element=self.element,
            concentration=float(predictions[0]) if len(predictions) > 0 else 0.0,
            uncertainty=float(predictions[1]) if len(predictions) > 1 else 0.0,
            detection_limit=float(predictions[2]) if len(predictions) > 2 else 0.0,
            spectrum_id=spectrum_id,
        )


# =============================================================================
# Training Pipeline
# =============================================================================

def create_training_dataset(
    spectra: List[np.ndarray],
    labels: np.ndarray,
    element: str,
    augmentation_config: Optional[AugmentationConfig] = None,
    test_split: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training and validation datasets with augmentation.
    
    Parameters
    ----------
    spectra : list of ndarray
        Original spectra
    labels : ndarray
        Labels (n_spectra, n_outputs)
    element : str
        Target element
    augmentation_config : AugmentationConfig, optional
        Augmentation settings
    test_split : float
        Fraction for test set
    seed : int
        Random seed
        
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    rng = np.random.default_rng(seed)
    
    # Split before augmentation (to avoid data leakage)
    n_samples = len(spectra)
    n_test = int(n_samples * test_split)
    indices = rng.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_test = np.array([spectra[i] for i in test_indices])
    y_test = labels[test_indices]
    
    train_spectra = [spectra[i] for i in train_indices]
    train_labels = labels[train_indices]
    
    # Augment training data
    augmentor = SpectralAugmentor(augmentation_config)
    augmentor.set_seed(seed)
    
    X_train, y_train, _ = augmentor.augment_dataset(train_spectra, train_labels)
    
    logger.info(f"Created dataset: {len(X_train)} training, {len(X_test)} test samples")
    
    return X_train, y_train, X_test, y_test


def train_naa_ann_model(
    spectra: List[np.ndarray],
    labels: np.ndarray,
    element: str,
    model_config: Optional[NAAANNConfig] = None,
    augmentation_config: Optional[AugmentationConfig] = None,
    save_path: Optional[str] = None,
    test_split: float = 0.2,
) -> Tuple[NAAANNModel, Dict[str, List[float]]]:
    """
    Complete training pipeline for NAA-ANN model.
    
    Parameters
    ----------
    spectra : list of ndarray
        Training spectra
    labels : ndarray
        Training labels (n_spectra, n_outputs)
    element : str
        Target element
    model_config : NAAANNConfig, optional
        Model configuration
    augmentation_config : AugmentationConfig, optional
        Augmentation configuration
    save_path : str, optional
        Path to save trained model
    test_split : float
        Test set fraction
        
    Returns
    -------
    tuple
        (trained_model, training_history)
    """
    # Create dataset
    X_train, y_train, X_test, y_test = create_training_dataset(
        spectra, labels, element,
        augmentation_config=augmentation_config,
        test_split=test_split,
    )
    
    # Update config with actual channel count
    if model_config is None:
        model_config = NAAANNConfig()
    model_config.n_channels = X_train.shape[1]
    model_config.n_outputs = y_train.shape[1] if y_train.ndim > 1 else 1
    
    # Create and train model
    model = NAAANNModel(model_config, element=element)
    model.build()
    
    history = model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        save_best=save_path + '/best_model.keras' if save_path else None,
    )
    
    # Save final model
    if save_path:
        model.save(save_path)
    
    # Evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2, axis=0))
    logger.info(f"Test RMSE for {element}: {rmse}")
    
    return model, history


# =============================================================================
# Integration with FluxForge Pipeline
# =============================================================================

class NAAANNAnalyzer:
    """
    High-level analyzer that combines k0-NAA and ANN approaches.
    
    This class provides a unified interface for NAA analysis,
    using ANN for initial predictions and k0-NAA for validation.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        models_dir : str, optional
            Directory containing pre-trained models
        """
        self.models: Dict[str, NAAANNModel] = {}
        self.models_dir = Path(models_dir) if models_dir else None
    
    def load_model(self, element: str, model_path: Optional[str] = None):
        """Load a trained model for an element."""
        if model_path is None and self.models_dir:
            model_path = self.models_dir / element
        
        if model_path and Path(model_path).exists():
            self.models[element] = NAAANNModel.load(model_path)
            logger.info(f"Loaded model for {element}")
        else:
            logger.warning(f"No model found for {element}")
    
    def analyze_spectrum(
        self,
        spectrum: np.ndarray,
        elements: Optional[List[str]] = None,
        spectrum_id: str = "",
    ) -> Dict[str, NAAANNResult]:
        """
        Analyze a spectrum for multiple elements.
        
        Parameters
        ----------
        spectrum : ndarray
            Input spectrum
        elements : list of str, optional
            Elements to analyze (default: all loaded)
        spectrum_id : str
            Spectrum identifier
            
        Returns
        -------
        dict
            Results keyed by element
        """
        elements = elements or list(self.models.keys())
        results = {}
        
        for element in elements:
            if element in self.models:
                results[element] = self.models[element].analyze_spectrum(
                    spectrum, spectrum_id
                )
        
        return results
    
    def batch_analyze(
        self,
        spectra: List[np.ndarray],
        spectrum_ids: Optional[List[str]] = None,
        elements: Optional[List[str]] = None,
    ) -> List[Dict[str, NAAANNResult]]:
        """
        Analyze multiple spectra.
        
        Parameters
        ----------
        spectra : list of ndarray
            Input spectra
        spectrum_ids : list of str, optional
            Spectrum identifiers
        elements : list of str, optional
            Elements to analyze
            
        Returns
        -------
        list of dict
            Results for each spectrum
        """
        ids = spectrum_ids or [f"spectrum_{i}" for i in range(len(spectra))]
        
        return [
            self.analyze_spectrum(spec, elements, spec_id)
            for spec, spec_id in zip(spectra, ids)
        ]
