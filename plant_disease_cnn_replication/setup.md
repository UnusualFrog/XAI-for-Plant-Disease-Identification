## Project Setup

### Step 1 — Manually Download datasets
The rice disease and cassava disease datasets must be downloaded manually and placed in the following directories:
- `./plant_disease_cnn_replication/data/cassava_data`
- `.plant_disease_cnn_replication/data/rice_data`

Links to datasets:
- Rice: https://data.mendeley.com/datasets/fwcj7stb8r/1
- Cassava: https://www.kaggle.com/competitions/cassava-disease/data 
 
### Step 2 — Create and activate a virtual environment
 
```bash
python3 -m venv venv
source venv/bin/activate
```
 
### Step 3 — Install dependencies
 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
 
This installs TensorFlow 2.21 along with bundled CUDA 12.x and cuDNN 9.x libraries inside the venv. No system-wide CUDA installation is required.
 
### Step 4 — Fix GPU library visibility (required if no system CUDA)
 
TensorFlow needs to find the CUDA libraries that were installed into the venv. By default, the system's dynamic linker doesn't look inside the venv, so GPU devices will not be detected without this step.
 
The fix appends an `LD_LIBRARY_PATH` export to the venv's `activate` script so it is set automatically every time the venv is activated:
 
```bash
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusparse/lib"' >> venv/bin/activate
```
 
### Step 5 — Reload the venv to apply the path
 
```bash
deactivate
source venv/bin/activate
```
 
### Step 6 — Verify GPU is detected
 
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

### Step 7 — Set TFDS directory in data_loading_01.py
The script `data_loading_01.py` must have its `TFDS_DATA_DIR` variable path set explicitly to the desired path for downloading the plantvillage dataset

```
# CHANGE THIS TO DESIRED DOWNLOAD DIRECTORY ON LOCAL MACHINE
TFDS_DATA_DIR = "C:\HDD\example\tensorflow_datasets"
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR
```

### Step 8 — Running Scripts
Scripts are intended to be run in the order denoted by the suffix of filename (i.e. data_loading_**01**.py). On initial setup, run each script once to verify working setup. Subseqeuent runs can skip `data_loading_01.py` and `model_02.py` and proceed directly to `train_03.py` for baseline model replication and then `explain_04.py` for the enhancement of the baseline through the added XAI techniques