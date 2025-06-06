# Conditional DDPM for Wildfire Spread Prediction

## Goal
This project aims to forecast future wildfire spread masks using a Conditional Denoising Diffusion Probabilistic Model (DDPM). The model is trained on 1–5 days of multichannel environmental inputs from the WildfireSpreadTS dataset to predict the fire mask for a subsequent day.

## Setup

1.  **Create Python Environment**:
    It's highly recommended to use a virtual environment.
    ```bash
    python3 -m venv wildfire_env
    source wildfire_env/bin/activate  # macOS/Linux
    # wildfire_env\Scripts\activate   # Windows
    ```
    Alternatively, using conda:
    ```bash
    conda create -n wildfire_env python=3.9
    conda activate wildfire_env
    ```

2.  **Install Dependencies**:
    ```bash
    pip install torch torchvision torchaudio
    pip install numpy Pillow tqdm
    pip install rasterio  # For processing .tif files in prepare_data.py
    # pip install matplotlib # Optional, for visualization
    ```
    For specific PyTorch versions (e.g., with CUDA support), refer to the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Data Preparation

The model expects data in a specific format: conditioning inputs as `.npy` files and target masks as `.png` files.

1.  **Download Raw Dataset**:
    * Download the WildfireSpreadTS dataset (e.g., from Zenodo, as `WildfireSpreadTS.zip`).
    * Unzip it to a known location on your system.

2.  **Configure `prepare_data.py`**:
    * Open `scripts/prepare_data.py`.
    * Set `RAW_DATA_DIR` to the path where you unzipped the raw WildfireSpreadTS dataset.
    * **Crucially, you MUST modify the `process_raw_input_data` and `process_raw_target_data` functions within this script.** These functions need to:
        * Correctly iterate through the raw dataset's directory structure (e.g., `YEAR/fire_EVENTID/YYYY-MM-DD.tif`).
        * Use `rasterio` to read the multi-band `.tif` files.
        * Consult the WildfireSpreadTS documentation (`WildfireSpreadTS_Documentation.pdf`) to identify:
            * Band indices for environmental variables (for conditioning input).
            * The band index for the active fire mask (for the target).
        * Update `NUM_CHANNELS_PER_DAY`, `TARGET_FIRE_MASK_BAND_INDEX`, and `LIST_OF_ENV_BAND_INDICES` in `prepare_data.py` based on the documentation and your choices.
        * Implement logic to extract, resize (to `IMAGE_SIZE`), normalize (inputs, e.g., to `[-1, 1]`), and binarize (targets) the data.
        * Save conditioning inputs as `(C, H, W)` `.npy` arrays and target masks as single-channel binary `.png` images in the `data/` subdirectories.
    * Define a strategy for splitting fire events into `train`, `val`, and `test` sets within `prepare_data.py`.

3.  **Run Data Preparation Script**:
    Navigate to the `scripts/` directory and run:
    ```bash
    cd wildfire_ddpm/scripts/
    python prepare_data.py
    ```
    This will populate `wildfire_ddpm/data/` with the processed data.

## Training

1.  **Configure Training Script**:
    * Open `scripts/train.py`.
    * Review the `CONFIG` dictionary at the top.
    * **Important**: Ensure `CONFIG["condition_channels"]` matches the `TOTAL_INPUT_CHANNELS` (i.e., `NUM_CONDITIONING_DAYS * NUM_CHANNELS_PER_DAY`) derived from your `prepare_data.py` setup.
    * Adjust `batch_size`, `epochs`, `learning_rate`, `image_size`, etc., as needed.
    * Specify `device` (e.g., "cuda", "mps", "cpu").

2.  **Run Training**:
    Navigate to the `scripts/` directory and run:
    ```bash
    cd wildfire_ddpm/scripts/ # if not already there
    python train.py --run_name wildfire_exp1 --epochs 100 --batch_size 4 --condition_channels 24
    ```
    *(Adjust `--condition_channels` and other arguments as per your configuration.)*
    * Checkpoints will be saved in `checkpoints/{run_name}/`.
    * Sample images generated during training will be in `outputs/training_samples/`.

## Inference (Sampling)

1.  **Configure Inference Script**:
    * Open `scripts/inference.py`.
    * Review `INFERENCE_CONFIG`.
    * Ensure model parameters (like `condition_channels`, `image_size`) match the trained model. These are often loaded from the checkpoint's config if saved.

2.  **Run Inference**:
    Provide the path to a trained checkpoint and a sample conditioning input (`.npy` file from your processed test/validation set).
    ```bash
    cd wildfire_ddpm/scripts/ # if not already there
    python inference.py \
        --checkpoint ../checkpoints/wildfire_exp1/ckpt_epoch_100.pt \
        --condition_input ../data/test/inputs/some_sample.npy \
        --output_dir ../outputs/predictions_run1 \
        --num_samples 4 \
        --condition_channels 24
    ```
    *(Adjust paths and parameters accordingly.)*
    * Predicted masks will be saved in the specified output directory.

## Key Configuration Points

* **`condition_channels`**: This is the total number of channels in your input conditioning data (`.npy` files). It's calculated as `NUM_CONDITIONING_DAYS * NUM_CHANNELS_PER_DAY` (from `prepare_data.py`). This value must be consistent across `prepare_data.py`, `train.py` (CONFIG), `inference.py` (INFERENCE_CONFIG), and the U-Net model instantiation.
* **`image_size`**: Must be consistent across data preparation, training, and inference.
* **Normalization**: Ensure your input conditioning data is normalized (e.g., to `[-1, 1]`) during the `prepare_data.py` step. Target masks are expected to be binary `[0, 1]` by the `WildfireDataset` loader, and are internally scaled to `[-1, 1]` for the diffusion process during training.

## Compute Requirements

* **GPU Recommended**: Training DDPMs is computationally intensive. An NVIDIA GPU with CUDA support (recommended >= 8-12GB VRAM for 64x64 images, more for larger sizes) is highly recommended for reasonable training times.
* **Mac MPS**: Macs with M-series chips can use "mps" for GPU acceleration, which is faster than CPU but generally slower than CUDA.
* **CPU**: Possible for debugging or very small experiments, but will be extremely slow for full training.
