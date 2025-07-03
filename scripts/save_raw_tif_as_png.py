# save_raw_tif_as_png.py
import rasterio
from PIL import Image
import numpy as np
import os
from pathlib import Path # Added for easier path manipulation

def save_tif_band_as_png(tif_path, output_png_path, band_index=0, scale_to_255=True):
    """
    Loads a specific band from a TIFF file, scales it to 0-255, and saves as PNG.
    Also prints basic statistics of the raw data.
    """
    try:
        with rasterio.open(tif_path) as src:
            # Read the specified band (rasterio bands are 1-indexed)
            data = src.read(band_index + 1) 
            
            # --- NEW: Print raw data statistics ---
            print(f"\n--- Raw Data Stats for {os.path.basename(tif_path)} (Band {band_index + 1}) ---")
            print(f"Shape: {data.shape}")
            print(f"Data Type: {data.dtype}")
            print(f"Min Value: {np.min(data)}")
            print(f"Max Value: {np.max(data)}")
            print(f"Mean Value: {np.mean(data):.4f}")
            # Print a small slice of the data (e.g., top-left corner)
            print("Top-left 5x5 pixel values:")
            print(data)
            print("--------------------------------------------------")
            # --- END NEW ---

            # Normalize and scale to 0-255 for visualization
            if scale_to_255:
                print("scale to 255")
                # Handle potential non-binary data (e.g., timestamps)
                # If it's a binary mask (0 or non-zero for fire), simply binarize and scale
                if np.max(data) > 0: # Avoid division by zero if all pixels are 0
                    # Simple binarization: anything > 0 is fire
                    print(np.max(data))
                    data = (data > 0).astype(np.uint8) * 255
                else:
                    data = np.zeros_like(data, dtype=np.uint8) # All black if no fire
                    print("else")
            # Ensure it's uint8 for PIL
            img_array = data.astype(np.uint8)
            
            # Create PIL Image and save
            img = Image.fromarray(img_array, mode='L') # 'L' for grayscale
            img.save(output_png_path)
            print(f"Saved {output_png_path}")
    except Exception as e:
        print(f"Error processing {tif_path}: {e}")

if __name__ == "__main__":
    print("name = main")
    # --- Define paths ---
    # Get the parent directory of the current working directory (which is /workspace/ in your case)
    # Assuming you run this script from /workspace/Fire-DDPM/
    base_workspace_root = Path(os.getcwd()).parent 
    
    # Output directory for these raw TIFF visualizations
    output_viz_dir = base_workspace_root / "outputs" / "raw_tif_visualizations"
    output_viz_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

    # Path to the raw target .tif file that corresponds to the all-white target you saw
    # This is the target for Day N+1
    raw_target_tif_file = "/workspace/WildfireSpreadTS_Data/2020/fire_23654679/2020-01-04.tif" 
    
    # Path to the raw input .tif file for the day before (Day N)
    raw_input_tif_file = "/workspace/WildfireSpreadTS_Data/2020/fire_23654679/2020-01-03.tif"

    # Output paths for the PNGs within your project's outputs folder
    output_target_png_file = output_viz_dir / "raw_target_fire_2020-01-04.png"
    output_input_png_file = output_viz_dir / "raw_input_fire_2020-01-03.png"

    print(f"Attempting to save raw target TIFF: {raw_target_tif_file}")
    save_tif_band_as_png(raw_target_tif_file, output_target_png_file, band_index=0, scale_to_255=True)

    print(f"Attempting to save raw input TIFF: {raw_input_tif_file}")
    save_tif_band_as_png(raw_input_tif_file, output_input_png_file, band_index=0, scale_to_255=True)
