from setuptools import setup, find_packages

setup(
    name="eeg_seizure_cnn_detector",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
