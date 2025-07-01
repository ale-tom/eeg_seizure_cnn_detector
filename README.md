# EEG Seizure Detection using CNNs in PyTorch

[![Build Status](https://github.com/ale-tom/linkedin-job-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/ale-tom/linkedin-job-analysis/actions)
[![Coverage Status](https://coveralls.io/repos/github/ale-tom/linkedin-job-analysis/badge.svg?branch=master)](https://coveralls.io/github/ale-tom/linkedin-job-analysis?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org)
[![Repo Size](https://img.shields.io/github/repo-size/ale-tom/linkedin-job-analysis)](https://github.com/ale-tom/linkedin-job-analysis)


## Overview

This project implements a cutting-edge CNN-based pipeline using PyTorch to detect EEG seizures from publicly available multi-channel scalp recordings. It compares both raw signal and spectrogram-based approaches, addresses class imbalance, and emphasizes interpretability with saliency and Grad‑CAM methods.

## Objectives

- Build robust 1D and 2D CNNs that process raw EEG signals and spectrograms.
- Tackle class imbalance with custom samplers and cost-sensitive training.
- Validate model performance with subject-wise cross-validation.
- Use interpretability techniques to highlight EEG patterns associated with seizures.

## Repository structure
```
eeg_seizure_detection/
├── data/               # preprocessed data and metadata files
├── notebooks/          # Jupyter notebooks 
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_cnn1d.ipynb
│   ├── 04_train_cnn2d.ipynb
│   └── 05_interpretability.ipynb
│
├── src/                # Source code
│   ├── __init__.py
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualise.py
├── models/             
│   ├── cnn1d.py
│   ├── cnn2d.py
│   └── imbalance_sampler.py
├── tests/              # Unit tests (using pytest)
│   └── test_preprocess.py
├── .github/            # GitHub configuration files 
│   ├── workflows/
│   │   └── ci.yml      # Continuous Integration (CI) configuration using GitHub Actions
│   └── ISSUE_TEMPLATE.md
├── LICENSE             # License file 
├── README.md           # Project overview, badges, setup instructions, etc.
├── requirements.txt    # Project dependencies
└── setup.py            # Packaging configuration
```

## Datasets

### Children's Hospital Boston(CHB) MIT Scalp EEG Dataset  
- 22 pediatric subjects with long interictal and ictal EEG recordings, 182 annotated seizures, EDF format, sampled at 256 Hz.  
- 22 channels (10-20 montage) in each EDF file.
- Approximately 42 GB; includes detailed seizure onset/offset annotations in `.seizure` files.  


### Siena Scalp EEG Database  
- 14 adult subjects (ages 20–71) with long interictal and ictal EEG recordings, 47 annotated seizures, sampled at 512 Hz.  
- Variable number of channels (10-20 monatage) between subjects.
- Approximately 20 GB; includes EEG + ECG (1 subject) + Vagal nerve stimulation (1 subject) and expert-annotated events.

## Getting started

### Prerequisites
- Python 3.10+  
- PyTorch, numpy, scipy, matplotlib, wfdb, MNE (see `requirements.txt`)

### Setup
```bash
git clone https://github.com/your-username/eeg_seizure_detection.git
cd eeg_seizure_detection
pip install -r requirements.txt
docker build -t eeg-seizure .
```

### Data download
This script uses boto3 to fetch CHB‑MIT and Siena datasets from PhysioNet public AWS S3 bucket. 
Then it employs the WFDB Python toolkit to verify that each EDF file is readable.
```bash
python scripts/download_data.py
```
### Preprocessing
Segment EEG recordings into fixed-length windows with labels for seizure detection.
Uses MNE to load EDF files and to apply continuous EEG preprocessing:
* band-pass filters (e.g., 0.5–1 Hz high-pass to remove slow drift, 35–70 Hz low-pass to remove muscle noise)
* Notch filters (50/60 Hz) to suppress power-line interference.
* Re-reference signals to an average reference to enhance signal localisation. 
WFDB is used to read seizure annotations. The script applies spectrogram transform,
and saves each window and its label as a compressed NumPy file. Window metadata is recorded in CSV, including channel
names.

*Note*:I chose not to apply artifact removal during the seizure recordings because any attempt to filter out muscle, 
blink, or cardiac noise risks erasing true ictal features since these artifacts share the same frequency bands and 
waveforms as genuine epileptic activity, so aggressive cleaning could strip away critical seizure dynamics.

```bash
python scripts/preprocess.py \
  --input_dir data/raw/ \
  --output_dir data/preprocessed/ \
  --window_sec 5 \
  --overlap 0.5 \
  --to_spectrogram
```

### Model training
#### 1D CNN (raw time-series)
```bash
python scripts/train.py \
  --model cnn1d \
  --data_dir data/preprocessed \
  --epochs 50 \
  --batch_size 64 \
  --use_sampler
```
#### 2D CNN (spectrograms)
Change --model cnn2d_spectrogram.

Both support weighted loss or imbalance-aware sampling.

### Evaluation
```bash
python scripts/evaluate.py \
  --model_path models/best_model.pth \
  --data_dir data/preprocessed
```
### Interpretability
Outputs saliency and Grad‑CAM visualizations for both raw and spectrogram inputs.
python scripts/visualise.py \
  --model_path models/best_model.pth \
  --data_dir data/preprocessed \
  --method grad_cam

### Results
* Compare 1D vs. 2D CNNs.
* Report AUROC, sensitivity, specificity.
* Visualize Grad‑CAM highlighting seizure-critical EEG segments.
* Subject-wise cross-validation to verify generalization.

### Testing
Includes unit tests for preprocessing and model consistency.
```bash
pytest
```
## Contributing
Contributions welcome! Please submit issues or pull requests for:

* New augmentations or imbalance methods
* Alternative CNN architectures or hybrid CNN+LSTM models
* Extended evaluation or additional datasets

## License
MIT License

## References
* Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/cgi/content/full/101/23/e215]; 2000 (June 13).
* Detti, P. (2020). Siena Scalp EEG Database (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/5d4a-j060
* Detti, P., Vatti, G., Zabalo Manrique de Lara, G. EEG Synchronization Analysis for Seizure Prediction: A Study on Data of Noninvasive Recordings. Processes 2020, 8(7), 846; https://doi.org/10.3390/pr8070846
* Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
