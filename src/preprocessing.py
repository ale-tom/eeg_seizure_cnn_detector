"""
Segment EEG recordings into fixed-length windows with labels for seizure detection.
Uses MNE to load EDF files and to apply continuous EEG preprocessing (band-pass, notch filtering, and baseline normalisation),
WFDB to read seizure annotations, applies optional spectrogram transform,
and saves each window and its label as a compressed NumPy file. Also records window metadata in CSV, including channel
names.
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import mne
import wfdb
from scipy.signal import spectrogram


def extract_annotations(record_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read seizure annotations for a given EEG record using WFDB and return onset/offset sample indices.
    """
    ann = wfdb.rdann(str(record_path), extension="seizures")
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
    Segment multi-channel signal into windows, normalise each window across channels,
    assign binary seizure labels, and collect metadata for each segment.
    """
    if not (0 <= overlap < 1):
        raise ValueError(f"overlap must be in [0,1), got {overlap}")

    window_len = int(window_sec * fs)
    step = int(window_len * (1 - overlap))

    segments, labels = [], []
    metadata: List[Dict] = []
    n_samples = signal.shape[0]

    for idx, start in enumerate(range(0, n_samples - window_len + 1, step)):
        end = start + window_len
        window = signal[start:end]  # shape: samples x channels
        mid = start + window_len // 2
        label = int(np.any((onsets <= mid) & (offsets >= mid)))

        # Normalise window to zero mean, unit variance per channel
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        std[std == 0] = 1.0
        normed = (window - mean) / std

        if to_spectrogram:
            freqs, times, Sxx = spectrogram(
                normed.T, fs=fs, axis=1, nperseg=window_len // 4
            )
            data = Sxx
        else:
            data = normed.T  # channels x samples

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
    sel_channels: Optional[List[str]],
    rewrite: Optional[bool] = True,
) -> None:
    """
    Process all EDF files in input_dir: apply filtering and re-referencing,
    segment them, save windows, and write metadata CSV.

    If rewrite is False, entire records that already have any output .npz
    files are skipped. Individual windows are also skipped if they exist.
    Metadata CSV accumulates only new windows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    new_metadata: List[Dict] = []

    # If rewriting, remove existing metadata file so we start fresh
    if rewrite and metadata_path.exists():
        metadata_path.unlink()

    for edf_path in input_dir.rglob("*.edf"):
        record_name = edf_path.stem
        # Check for existing outputs for this record
        pattern = f"{record_name}_win*.npz"
        existing = list(output_dir.glob(pattern))
        if existing and not rewrite:
            print(f"[SKIP RUN] {record_name} (outputs exist)")
            continue

        # Load and preprocess raw data
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        raw.filter(l_freq=0.5, h_freq=60.0, fir_design="firwin")
        raw.notch_filter(freqs=[60.0], fir_design="firwin")
        if sel_channels:
            picks = [ch for ch in sel_channels if ch in raw.ch_names]
        if not picks:
            raise ValueError(f"No matching channels found in {edf_path.name}")

        raw.pick(picks)

        fs = raw.info["sfreq"]
        signal = raw.get_data().T
        channel_names = raw.ch_names

        # Extract annotations
        try:
            onsets, offsets = extract_annotations(edf_path)
        except Exception:
            onsets, offsets = np.array([]), np.array([])

        # Segment and save windows
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

        for idx, (data, label, meta) in enumerate(zip(segments, labels, metadata)):
            fname = f"{record_name}_win{idx}.npz"
            out_file = output_dir / fname
            if out_file.exists() and not rewrite:
                continue
            np.savez_compressed(out_file, data=data, label=label)
            new_metadata.append(meta)

        print(f"[PROCESSED] {record_name}: {len(segments)} windows")

    # Write or append metadata for new windows only
    if new_metadata:
        df_new = pd.DataFrame(new_metadata)
        if metadata_path.exists():
            df_old = pd.read_csv(metadata_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(metadata_path, index=False)
        print(f"[METADATA] Saved {len(new_metadata)} new entries to {metadata_path}")
    else:
        print("[METADATA] No new windows; metadata unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess EEG EDF files into normalised, labeled windows"
    )
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--metadata_csv", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--to_spectrogram", action="store_true")
    parser.add_argument(
        "--channels",
        nargs="+",
        type=str,
        help="Optional list of channels to include; defaults to all",
    )
    args = parser.parse_args()

    preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata_csv,
        window_sec=args.window_sec,
        overlap=args.overlap,
        to_spectrogram=args.to_spectrogram,
        sel_channels=args.channels,
        rewrite=True,
    )


if __name__ == "__main__":
    main()
