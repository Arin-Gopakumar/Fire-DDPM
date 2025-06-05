import os
import numpy as np
from PIL import Image
import shutil # For cleaning up dummy data

# --- Configuration ---
# These paths should point to your raw WildfireSpreadTS dataset
RAW_DATA_DIR = "path/to/your/raw_wildfirespreadts_dataset" # IMPORTANT: Update this
OUTPUT_DIR = "../data" # Relative to the script's location in wildfire_ddpm/scripts/
IMAGE_SIZE = (64, 64) # Target H, W for your model
NUM_CONDITIONING_DAYS = 3 # Example: 1-5 days
# Example: List of channel indices to extract from raw data, or how you identify them
# This is highly dependent on your raw data structure.
# For WildfireSpreadTS, you'll need to map variable names to channel indices.
# Let's assume your raw data, after some initial processing, gives you N channels per day.
# And you want to use all of them.
# For this dummy script, we'll generate random channels.
NUM_CHANNELS_PER_DAY = 8 # Example: 8 environmental variables per day
TOTAL_INPUT_CHANNELS = NUM_CONDITIONING_DAYS * NUM_CHANNELS_PER_DAY

# Normalization: Choose your strategy. For this example, min-max to [-1, 1]
# In a real scenario, calculate mean/std from your training set for standardization.
NORMALIZE_INPUT = True

# --- Helper Functions ---
def normalize_array(arr):
    """Normalizes a NumPy array to the range [-1, 1]."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val > 0:
        return 2 * (arr - min_val) / (max_val - min_val) - 1
    else:
        return arr - min_val # Or just zeros if all values are the same

def process_raw_input_data(sample_id, raw_data_path_template):
    """
    Placeholder function to simulate loading and processing raw input data.
    Replace this with your actual data loading logic for .tif files.

    Args:
        sample_id (str): Identifier for the sample.
        raw_data_path_template (str): A template for finding raw files.

    Returns:
        np.array: Processed multichannel input data (C, H, W) or None if error.
    """
    # In a real scenario, you would:
    # 1. Identify the sequence of N day files for this sample_id.
    # 2. Load each GeoTIFF file for each day.
    # 3. Extract relevant channels/bands.
    # 4. Ensure all data is co-registered and on the same grid.
    # 5. Resize to IMAGE_SIZE if necessary (e.g., using rasterio.warp.reproject).
    # 6. Stack the channels from all N days: (NUM_CONDITIONING_DAYS * NUM_CHANNELS_PER_DAY, H, W).
    # For this dummy script, generate random data:
    print(f"Simulating processing for input: {sample_id}")
    try:
        # Example: this might load data like `RAW_DATA_DIR/region_X/day_Y_vars.tif`
        # stacked_channels = []
        # for day_idx in range(NUM_CONDITIONING_DAYS):
        #     # daily_data = load_and_process_geotiff(f"{raw_data_path_template}_day{day_idx+1}.tif") # (channels_per_day, H, W)
        #     # daily_data_resized = resize_raster(daily_data, IMAGE_SIZE)
        #     # stacked_channels.append(daily_data_resized)
        # final_input_data = np.concatenate(stacked_channels, axis=0) # (TOTAL_INPUT_CHANNELS, H, W)
        
        # Dummy data generation:
        final_input_data = np.random.rand(TOTAL_INPUT_CHANNELS, IMAGE_SIZE[0], IMAGE_SIZE[1]).astype(np.float32)

        if NORMALIZE_INPUT:
            # Channel-wise normalization is often better, but here's a global example
            final_input_data = normalize_array(final_input_data)
        
        return final_input_data
    except Exception as e:
        print(f"Error processing input for {sample_id}: {e}")
        return None

def process_raw_target_data(sample_id, raw_data_path_template):
    """
    Placeholder function to simulate loading and processing raw target mask data.
    Replace this with your actual data loading logic for .tif fire masks.

    Args:
        sample_id (str): Identifier for the sample.
        raw_data_path_template (str): A template for finding raw files.

    Returns:
        Image.Image: Processed binary mask image (H, W) or None if error.
    """
    # In a real scenario, you would:
    # 1. Load the target day's fire mask GeoTIFF.
    # 2. Convert it to a binary mask (0s and 1s, or 0s and 255s).
    # 3. Resize to IMAGE_SIZE if necessary.
    # 4. Ensure it's a single channel grayscale image.
    print(f"Simulating processing for target: {sample_id}")
    try:
        # Example: this might load `RAW_DATA_DIR/region_X/mask_day_N+1.tif`
        # mask_data = load_and_process_geotiff_mask(f"{raw_data_path_template}_mask.tif") # (1, H, W)
        # mask_data_resized = resize_raster_mask(mask_data, IMAGE_SIZE) # ensure binary
        # mask_image = Image.fromarray(mask_data_resized.squeeze().astype(np.uint8), mode='L') # Grayscale
        
        # Dummy data generation:
        dummy_mask_array = np.random.randint(0, 2, size=IMAGE_SIZE, dtype=np.uint8) * 255 # 0 or 255
        mask_image = Image.fromarray(dummy_mask_array, mode='L')
        return mask_image
    except Exception as e:
        print(f"Error processing target for {sample_id}: {e}")
        return None

def create_dataset(split_name, num_samples):
    """
    Creates the dataset split (train, val, test) with dummy data.
    You'll need to replace the sample_ids generation with your actual file list
    from the WildfireSpreadTS dataset.
    """
    inputs_dir = os.path.join(OUTPUT_DIR, split_name, "inputs")
    targets_dir = os.path.join(OUTPUT_DIR, split_name, "targets")

    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)

    print(f"\n--- Creating {split_name} data ---")
    # In a real scenario, iterate over your actual dataset identifiers (e.g., from a CSV or file scan)
    # For WildfireSpreadTS, this might be combinations of region and start date.
    for i in range(num_samples):
        sample_id = f"sample_{i:04d}"
        
        # This raw_data_path_template is a placeholder. You'll need to define how to map
        # sample_id to your actual raw file paths.
        # e.g., f"{RAW_DATA_DIR}/some_pattern_based_on_{sample_id}"
        raw_input_path_stub = os.path.join(RAW_DATA_DIR, f"stub_for_{sample_id}_input")
        raw_target_path_stub = os.path.join(RAW_DATA_DIR, f"stub_for_{sample_id}_target")

        # Process and save input
        input_data = process_raw_input_data(sample_id, raw_input_path_stub)
        if input_data is not None:
            np.save(os.path.join(inputs_dir, f"{sample_id}.npy"), input_data)
        else:
            print(f"Skipping input for {sample_id} due to processing error.")

        # Process and save target
        target_mask_image = process_raw_target_data(sample_id, raw_target_path_stub)
        if target_mask_image is not None:
            target_mask_image.save(os.path.join(targets_dir, f"{sample_id}.png"))
        else:
            print(f"Skipping target for {sample_id} due to processing error.")

    print(f"Finished creating {split_name} data with {num_samples} samples.")
    print(f"Input data saved in: {inputs_dir}")
    print(f"Target data saved in: {targets_dir}")


if __name__ == "__main__":
    print("Starting dataset preparation...")
    print(f"IMPORTANT: This script uses DUMMY data generation.")
    print(f"You MUST adapt 'process_raw_input_data' and 'process_raw_target_data' functions")
    print(f"and the sample ID generation to work with your actual WildfireSpreadTS .tif files.")
    print(f"Make sure RAW_DATA_DIR ('{RAW_DATA_DIR}') points to your dataset.")
    print(f"Output will be in: {os.path.abspath(OUTPUT_DIR)}")

    # Clean up previous dummy data if any (optional)
    # for split in ["train", "val", "test"]:
    #     if os.path.exists(os.path.join(OUTPUT_DIR, split)):
    #         print(f"Cleaning up existing dummy data in {split}...")
    #         # shutil.rmtree(os.path.join(OUTPUT_DIR, split, "inputs"))
    #         # shutil.rmtree(os.path.join(OUTPUT_DIR, split, "targets"))


    # Define number of samples for each split (adjust as needed)
    num_train_samples = 20 # Small number for quick dummy generation
    num_val_samples = 5
    num_test_samples = 5

    create_dataset("train", num_train_samples)
    create_dataset("val", num_val_samples)
    create_dataset("test", num_test_samples)

    print("\nDataset preparation script finished.")
    print("Please check the 'data' directory for the generated structure.")
    print("Remember to replace placeholder functions with actual data processing logic.")