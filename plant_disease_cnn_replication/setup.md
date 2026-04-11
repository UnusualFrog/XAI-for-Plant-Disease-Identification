## Project Setup
NOTE: For windows installation, WSL will need to be used, as Linux-speicfic packages are requried, all following commands should be performed using WSL. Install using the following command in powershell
```
wsl --install
```

## Step 1 - Clone repo from Github
NOTE: For windows users, ensure this repo is cloned using WSL within the Linux filesystem

```
git clone https://github.com/UnusualFrog/XAI-for-Plant-Disease-Identification
```


### Step 2 — Manually Download datasets
The rice disease and cassava disease datasets must be downloaded manually and placed in the following directories:
- `.\plant_disease_cnn_replication\data\cassava-disease`
- `.\plant_disease_cnn_replication\data\rice_data`

Links to datasets:
- Rice: https://data.mendeley.com/datasets/fwcj7stb8r/1
- Cassava: https://www.kaggle.com/competitions/cassava-disease/data 

### Step 3 — Fix Directory Formatting
Datasets should be fomratted as follows. 
- Rice dataset parent folder will need to be renamed to `rice_data`
- Rice classes will need to be renamed as below:

```
# RICE DATASET
# Expected folder structure:
#   rice_data/
#     bacterial_blight/   (1584 images)
#     blast/              (1440 images)
#     brown_spot/         (1600 images)
#     tungro/             (1308 images)
```

- Cassava dataset uses `train.zip` only
- Parent folder `train` within `train.zip` must be renamed to `cassava_data`
- Cassava classes should be renamed as below:

```
# CASSAVA DATASET
# Expected folder structure:
#   cassava_data/
#     Healthy/                        (316 images)
#     Cassava_Bacterial_Blight/       (466 images)
#     Cassava_Brown_Streak_Disease/   (1443 images)
#     Cassava_Green_Mite/             (773 images)
#     Cassava_Mosaic_Disease/         (2658 images)
```
 
## Step 4 - Install Python
Python version 3.12.3 must be used for this project. Install using the following commands

```
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-pip python3-venv build-essential
```

### Step 5 — Create and activate a virtual environment
 
```bash
python3 -m venv venv
source venv/bin/activate
```
 
### Step 6 — Install dependencies
 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
 
This installs TensorFlow 2.21 along with bundled CUDA 12.x and cuDNN 9.x libraries inside the venv. No system-wide CUDA installation is required.
 
### Step 7 — Fix GPU library visibility (required if no system CUDA)
 
TensorFlow needs to find the CUDA libraries that were installed into the venv. By default, the system's dynamic linker doesn't look inside the venv, so GPU devices will not be detected without this step.
 
The fix appends an `LD_LIBRARY_PATH` export to the venv's `activate` script so it is set automatically every time the venv is activated:
 
```bash
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusparse/lib"' >> venv/bin/activate
```
 
### Step 8 — Reload the venv to apply the path
 
```bash
deactivate
source venv/bin/activate
```
 
### Step 9 — Verify GPU is detected
 
```bash
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print('TF version:', tf.__version__)
print('GPU devices:', gpus)
print('Built with CUDA:', tf.test.is_built_with_cuda())
"
```
 
Expected output:
```
TF version: 2.21.0
GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Built with CUDA: True
```

### Step 10 — Set TFDS directory in data_loading_01.py
The script `data_loading_01.py` must have its `TFDS_DATA_DIR` variable path set explicitly to the desired path for downloading the plantvillage dataset

```
# Optionally create TFDS directory
mkdir -p ~/tensorflow_datasets

# CHANGE THIS TO DESIRED DOWNLOAD DIRECTORY ON LOCAL MACHINE
TFDS_DATA_DIR = "/home/regan/tensorflow_datasets"
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR
```

### Step 11 — Running Scripts
Scripts are intended to be run in the order denoted by the suffix of filename (i.e. data_loading_**01**.py). On initial setup, run each script once to verify working setup. Subseqeuent runs can skip `data_loading_01.py` and `model_02.py` and proceed directly to `train_03.py` for baseline model replication and then `explain_04.py` for the enhancement of the baseline through the added XAI techniques

NOTE: For windows users, ensure scripts are run through WSL