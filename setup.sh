#!/bin/bash
# Exit on error
set -e

# Create virtual environment
python3 -m venv mykernel
source mykernel/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools ipykernel
python -m ipykernel install --user --name=mykernel --display-name "Python (mykernel)"

# Install required packages
pip install --use-pep517 torch torchvision torchmetrics pytorch-gradcam matplotlib lime timm grad-cam pandas tqdm albumentations scikit-learn opencv-python kaggle keras yacs einops munch termcolor huggingface_hub tensorboard -q 

# Setup Kaggle credentials (ensure kaggle.json is present in current directory)
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download and unzip datasets (uncomment apt commands if on a fresh system)
sudo apt update -qq
sudo apt install -qq unzip

# Uncomment below lines to download from Kaggle (requires valid kaggle.json)
kaggle datasets download nirmalsankalana/sugarcane-leaf-disease-dataset
unzip -q sugarcane-leaf-disease-dataset.zip -d data

kaggle datasets download pungliyavithika/sugarcane-leaf-disease-classification
unzip -q sugarcane-leaf-disease-classification.zip -d data2

# Unzip Mendeley dataset (make sure mendeley.zip exists)
huggingface-cli login
huggingface-cli download omkar334/agri mendeley.zip Visualization.zip --local-dir .
unzip -q mendeley.zip -d data3
unzip -q Visualization.zip -d vizdata

# Clean up ZIP files after extraction
rm -f mendeley.zip
rm -f sugarcane-leaf-disease-dataset.zip
rm -f sugarcane-leaf-disease-classification.zip
rm -f Visualization.zip
