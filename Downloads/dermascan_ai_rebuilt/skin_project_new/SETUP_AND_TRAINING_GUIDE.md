# install python
install Python 3.10 / 3.11

# Check Python version (need 3.10–3.11)
python --version

# Check if you have a GPU
nvidia-smi

# Create the environment
python -m venv venv

# Activate it — Windows:
.venv\Scripts\activate

# Activate it — Mac/Linux:
source venv/bin/activate

# Core ML stack
pip install tensorflow==2.13.0
pip install keras==2.13.0

# Data & utilities
pip install numpy pandas scikit-learn
pip install matplotlib seaborn Pillow
pip install tqdm requests

# Install packages
python download_datasets.py --install

# Kaggle downloader
pip install kaggle

# Verify GPU is detected by TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# set up kaggle api key
Go to kaggle.com → Settings → API → Create New Token. It downloads kaggle.json.

# Windows — place kaggle.json here:
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\

# Mac/Linux — place kaggle.json here:
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Verify it works
python download_datasets.py --status

# Download everything + auto-organize folders
python download_datasets.py --all --organize

#  Check final dataset
python organize_kaggle.py --verify

# Quick sanity check — import test
python -c "import tensorflow as tf; import numpy as np; print('All OK')"

# Check dataset folders are readable
python -c "import os; print([d for d in os.listdir('dataset/train')])"

# Final Step Train model
python train_model.py


