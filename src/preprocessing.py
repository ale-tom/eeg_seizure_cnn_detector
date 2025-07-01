"""
Segment EEG recordings into fixed-length windows with labels for seizure detection.
Uses MNE to load EDF files and to apply continuous EEG preprocessing (band-pass, notch filtering, re-referencing),
WFDB to read seizure annotations, applies optional spectrogram transform,
and saves each window and its label as a compressed NumPy file. Also records window metadata in CSV, including channel
names.
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import mne
import wfdb
from scipy.signal import spectrogram


def extract_annotations(record_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read seizure annotations for a given EEG record using WFDB.
    Returns arrays of onset and offset sample indices.
    """
    ann = wfdb.rdann(str(record_path), extension="seizure")
    onset = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym == "["])
    offset = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym == "]"])
    return onset, offset


def segment_signal(
    signal: np.ndarray,
    fs: float,
    window_sec: float,
    overlap: float,
    onsets: np.ndarray,
    offsets: np.ndarray,
    to_spectrogram: bool,
    record_name: str,
    channel_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Segment multi-channel signal into windows, assign labels, and collect metadata.
    Returns arrays of windowed data, binary labels, and a list of metadata dicts.
    """
    window_len = int(window_sec * fs)
    step = int(window_len * (1 - overlap))
    segments, labels = [], []
    metadata: List[Dict] = []

    n_samples = signal.shape[0]
    for idx, start in enumerate(range(0, n_samples - window_len + 1, step)):
        end = start + window_len
        segment = signal[start:end].T  # shape: channels x samples
        mid = start + window_len // 2
        label = int(np.any((onsets <= mid) & (offsets >= mid)))

        if to_spectrogram:
            freqs, times, Sxx = spectrogram(
                segment, fs=fs, axis=1, nperseg=window_len // 4
            )
            data = Sxx
        else:
            data = segment

        segments.append(data)
        labels.append(label)
        metadata.append(
            {
                "record": record_name,
                "window_index": idx,
                "start_sample": start,
                "end_sample": end,
                "start_sec": start / fs,
                "end_sec": end / fs,
                "label": label,
                "channel_names": ",".join(channel_names),
            }
        )

    return np.array(segments), np.array(labels), metadata


def preprocess(
    input_dir: Path,
    output_dir: Path,
    metadata_path: Path,
    window_sec: float,
    overlap: float,
    to_spectrogram: bool,
) -> None:
    """
    Process all EDF files in input_dir: apply filtering and re-referencing,
    segment them, save windows, and write metadata CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metadata: List[Dict] = []

    for edf_path in input_dir.rglob("*.edf"):
        record_name = edf_path.stem
        # Load data with MNE
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        # Continuous preprocessing
        raw.filter(l_freq=0.5, h_freq=70.0, fir_design="firwin")  # band-pass 0.5–70 Hz
        raw.notch_filter(freqs=[50.0, 60.0], fir_design="firwin")  # notch at power-line
        raw.set_eeg_reference("average", projection=False)  # average reference

        fs = raw.info["sfreq"]  # sampling frequency
        signal = raw.get_data().T  # samples x channels
        channel_names = raw.ch_names

        # Extract annotations
        try:
            onsets, offsets = extract_annotations(edf_path)
        except Exception:
            onsets, offsets = np.array([]), np.array([])

        segments, labels, metadata = segment_signal(
            signal,
            fs,
            window_sec,
            overlap,
            onsets,
            offsets,
            to_spectrogram,
            record_name,
            channel_names,
        )

        # Save segment files
        for idx, (data, label) in enumerate(zip(segments, labels)):
            fname = f"{record_name}_win{idx}.npz"
            np.savez_compressed(output_dir / fname, data=data, label=label)
        print(f"Processed {edf_path.name}: {len(segments)} windows")

        all_metadata.extend(metadata)

    # Write metadata CSV
    df = pd.DataFrame(all_metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess EEG EDF files into labeled windows with filtering and referencing"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw EDF files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Directory to save preprocessed windows",
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("data/metadata"),
        help="Path to save metadata CSV",
    )
    parser.add_argument(
        "--window_sec", type=float, default=5.0, help="Length of each window in seconds"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.3,
        help="Fractional overlap between windows (0-1)",
    )
    parser.add_argument(
        "--to_spectrogram", action="store_true", help="Convert windows to spectrograms"
    )
    args = parser.parse_args()

    preprocess(
        args.input_dir,
        args.output_dir,
        args.metadata_csv,
        args.window_sec,
        args.overlap,
        args.to_spectrogram,
    )


if __name__ == "__main__":
    main()
