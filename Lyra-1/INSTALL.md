### Environment setup

Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
Cosmos requires the Python version to be `3.10.x`. Please also make sure you have `conda` installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

The below commands creates the `lyra` conda environment and installs the dependencies for inference:
```bash
# Create the lyra conda environment.
conda env create --file lyra.yaml
# Activate the lyra conda environment.
conda activate lyra
# Install the dependencies.
pip install -r requirements_gen3c.txt
pip install -r requirements_lyra.txt
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0
# Install Apex for inference.
git clone https://github.com/NVIDIA/apex
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex
# Install MoGe for inference.
pip install git+https://github.com/microsoft/MoGe.git
# Install Mamba for reconstruction model.
pip install --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"
```

You can test the environment setup for inference with
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
```

### Download Cosmos-Predict1 tokenizer

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the Cosmos Tokenize model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
```bash
python3 -m scripts.download_tokenizer_checkpoints --checkpoint_dir checkpoints/cosmos_predict1 --tokenizer_types CV8x8x8-720p
```

The downloaded files should be in the following structure:
```
checkpoints/
├── Cosmos-Tokenize1-CV8x8x8-720p
├── Cosmos-Tokenize1-DV8x16x16-720p
├── Cosmos-Tokenize1-CI8x8-360p
├── Cosmos-Tokenize1-CI16x16-360p
├── Cosmos-Tokenize1-CV4x8x8-360p
├── Cosmos-Tokenize1-DI8x8-360p
├── Cosmos-Tokenize1-DI16x16-360p
└── Cosmos-Tokenize1-DV4x8x8-360p
```

Under the checkpoint repository `checkpoints/<model-name>`, we provide the encoder, decoder, the full autoencoder in TorchScript (PyTorch JIT mode) and the native PyTorch checkpoints. For instance for `Cosmos-Tokenize1-CV8x8x8-720p` model:
```bash
├── checkpoints/
│   ├── Cosmos-Tokenize1-CV8x8x8-720p/
│   │   ├── encoder.jit
│   │   ├── decoder.jit
│   │   ├── autoencoder.jit
│   │   ├── model.pt
```

### Download GEN3C checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the GEN3C model weights from [Hugging Face](https://huggingface.co/nvidia/GEN3C-Cosmos-7B):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_gen3c_checkpoints.py --checkpoint_dir checkpoints
   ```

### Download Lyra checkpoints

1. Download the Lyra model weights from [Hugging Face](https://huggingface.co/nvidia/Lyra):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_lyra_checkpoints.py --checkpoint_dir checkpoints
   ```
