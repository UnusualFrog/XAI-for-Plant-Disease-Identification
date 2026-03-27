## Environment Setup
 
### Step 1 — Create and activate a virtual environment
 
```bash
python3 -m venv venv
source venv/bin/activate
```
 
### Step 2 — Install dependencies
 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
 
This installs TensorFlow 2.21 along with bundled CUDA 12.x and cuDNN 9.x libraries inside the venv. No system-wide CUDA installation is required.
 
### Step 3 — Fix GPU library visibility (required if no system CUDA)
 
TensorFlow needs to find the CUDA libraries that were installed into the venv. By default, the system's dynamic linker doesn't look inside the venv, so GPU devices will not be detected without this step.
 
The fix appends an `LD_LIBRARY_PATH` export to the venv's `activate` script so it is set automatically every time the venv is activated:
 
```bash
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusparse/lib"' >> venv/bin/activate
```
 
### Step 4 — Reload the venv to apply the path
 
```bash
deactivate
source venv/bin/activate
```
 
### Step 5 — Verify GPU is detected
 
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