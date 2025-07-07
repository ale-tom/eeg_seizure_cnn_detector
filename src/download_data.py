"""
Download CHB-MIT and Siena EEG datasets from PhysioNet’s public AWS S3 bucket.
Uses anonymous S3 access (no AWS credentials) and verifies EDF readability.
"""

import argparse
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import wfdb
import mne

# Public S3 bucket and dataset prefixes
BUCKET_NAME = "physionet-open"  # PhysioNet Open Data on AWS
DATASETS = {"chbmit": "chbmit/1.0.0/", "siena": "siena-scalp-eeg/1.0.0/"}


def download_prefix(bucket: str, prefix: str, out_dir: Path) -> None:
    """
    List and download all objects under the given prefix from the S3 bucket
    into the local out_dir, skipping existing files.
    """
    s3 = boto3.resource(
        "s3", config=Config(signature_version=UNSIGNED)  # anonymous access
    )
    bucket = s3.Bucket(bucket)
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):
            continue  # skip directory placeholders
        target_path = out_dir / obj.key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            print(f"[SKIP] {obj.key}")
            continue
        print(f"[DOWNLOAD] {obj.key}")
        bucket.download_file(obj.key, str(target_path))


def validate_edf(edf_path: Path) -> bool:
    """
    Validates whether an EDF file is readable by attempting to parse its header using MNE.
    Returns True if the file can be read without loading the full signal data, otherwise False.
    """
    try:
        mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        return True
    except Exception as e:
        print(f"[ERROR] Could not read {edf_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and validate EEG datasets from PhysioNet AWS S3"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
        help="Datasets to download (default: all)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/raw"),
        help="Local directory for downloaded data",
    )
    args = parser.parse_args()

    for name in args.datasets:
        prefix = DATASETS[name]
        print(f"==> Fetching {name} from s3://{BUCKET_NAME}/{prefix}")
        download_prefix(BUCKET_NAME, prefix, args.out_dir)

        # Validate all EDFs we just downloaded
        print(f"==> Validating EDF files in {name}")
        for edf_file in (args.out_dir / prefix).rglob("*.edf"):
            ok = validate_edf(edf_file)
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {edf_file.relative_to(args.out_dir)}")


if __name__ == "__main__":
    main()
