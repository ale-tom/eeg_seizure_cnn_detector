"""
Tests segment_signal for correct segmentation and labeling, and spectrogram conversion.
"""
import numpy as np
import pytest

# Assuming preprocess.py is at project root or scripts/preprocess.py
from src.preprocessing import segment_signal


def create_dummy_signal(n_samples: int, n_channels: int) -> np.ndarray:
    """
    Create a dummy multi-channel signal with all zeros except a pulse to simulate a seizure.
    """
    signal = np.zeros((n_samples, n_channels))
    signal[n_samples // 2, 0] = 1.0  # spike in channel 0
    return signal


def test_segment_signal_no_seizure_raw():
    fs = 100.0
    window_sec = 1.0
    overlap = 0.0
    n_channels = 2
    n_samples = int(window_sec * fs * 3)
    signal = np.zeros((n_samples, n_channels))

    segments, labels, metadata = segment_signal(
        signal, fs, window_sec, overlap,
        onsets=np.array([]), offsets=np.array([]),
        to_spectrogram=False,
        record_name="test_rec",
        channel_names=["C1", "C2"]
    )
    assert segments.shape == (3, n_channels, int(fs * window_sec))
    assert all(label == 0 for label in labels)
    assert len(metadata) == 3
    assert metadata[0]["record"] == "test_rec"
    assert metadata[0]["label"] == 0


def test_segment_signal_with_seizure_raw():
    fs = 100.0
    window_sec = 1.0
    overlap = 0.0
    n_channels = 1
    n_samples = int(window_sec * fs * 5)
    signal = create_dummy_signal(n_samples, n_channels)

    segments, labels, metadata = segment_signal(
        signal, fs, window_sec, overlap,
        onsets=np.array([250]), offsets=np.array([250]),
        to_spectrogram=False,
        record_name="test_rec",
        channel_names=["C1"]
    )
    assert sum(labels) == 1
    idxs = [m['window_index'] for m in metadata if m['label'] == 1]
    assert idxs == [2]


def test_segment_signal_spectrogram():
    fs = 50.0
    window_sec = 2.0
    overlap = 0.0
    n_channels = 1
    n_samples = int(window_sec * fs * 2)
    signal = np.random.randn(n_samples, n_channels)

    segments, labels, metadata = segment_signal(
        signal, fs, window_sec, overlap,
        onsets=np.array([]), offsets=np.array([]),
        to_spectrogram=True,
        record_name="test_rec",
        channel_names=["C1"]
    )
    # For spectrogram mode, segments should be 4D: (n_windows, channels, freq_bins, time_bins)
    assert segments.ndim == 4
    assert labels.shape[0] == segments.shape[0]
    assert len(metadata) == segments.shape[0]


def test_segment_signal_invalid_overlap():
    fs = 100.0
    with pytest.raises(ValueError):
        segment_signal(
            signal=np.zeros((1000, 1)),
            fs=fs,
            window_sec=1.0,
            overlap=1.5,
            onsets=np.array([]), offsets=np.array([]),
            to_spectrogram=False,
            record_name="rec",
            channel_names=["C1"]
        )
