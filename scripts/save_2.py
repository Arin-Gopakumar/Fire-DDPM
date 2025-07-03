# save_raw_tif_as_png.py
import rasterio
from PIL import Image
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt # Added for plotting

def save_tif_band_as_png(tif_path, output_png_path, band_index=0, scale_to_255=True):
    """
    Loads a specific band from a TIFF file, scales it to 0-255, and saves as PNG.
    Also prints basic statistics and generates a histogram of the raw data.
    """
    try:
        with rasterio.open(tif_path) as src:
            # Read the specified band (rasterio bands are 1-indexed)
            data = src.read(band_index + 1) 
            
            # --- Print raw data statistics ---
            print(f"\n--- Raw Data Stats for {os.path.basename(tif_path)} (Band {band_index + 1}) ---")
            print(f"Shape: {data.shape}")
            print(f"Data Type: {data.dtype}")
            print(f"Min Value: {np.min(data)}")
            print(f"Max Value: {np.max(data)}")
            print(f"10th Max Value: {np.partition(data.flatten(), -10)[-10]}")
            print(f"Mean Value: {np.mean(data):.4f}")
            print(f"Standard Deviation value: {np.std(data)}")
            data_new = data.flatten()
            data_new.sort()
            data_sorted = data_new[::-1]
            print(data_sorted[:485])
            print(len(data_new))
            # Print a small slice of the data (e.g., top-left corner)
            print("Top-left 5x5 pixel values:")
            print(data[:5, :5])
            print("--------------------------------------------------")
            print(f"--- Iterating through top-left 5x5 pixels of {os.path.basename(tif_path)} ---")
            mean = np.mean(data)
            std = np.std(data)
            for row_idx in range(data.shape[0]): # Iterate up to 5 rows or max rows
                for col_idx in range(data.shape[1]): # Iterate up to 5 columns or max cols
                    pixel_value = data[row_idx, col_idx]
                    z_score = (pixel_value - mean)/std
                    if pixel_value > 2359:
                        pixel_value = 255
                    else: 
                        pixel_value = 0
                    data[row_idx, col_idx] = pixel_value

            # --- NEW: Generate and save histogram ---
            try:
                flat_data = data.flatten()
                
                plt.figure(figsize=(8, 6))
                plt.hist(flat_data, bins=50, color='skyblue', edgecolor='black')
                plt.title(f"Pixel Value Distribution for {os.path.basename(tif_path)} (Band {band_index + 1})")
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                plt.grid(axis='y', alpha=0.75)
                
                # Define output path for the histogram
                histogram_output_path = output_png_path.parent / f"{output_png_path.stem}_histogram.png"
                plt.savefig(histogram_output_path)
                plt.close() # Close the plot to free memory
                print(f"Saved histogram to {histogram_output_path}")
            except ImportError:
                print("Matplotlib not found. Skipping histogram generation. Please install it: pip install matplotlib")
            except Exception as e:
                print(f"Error generating histogram for {os.path.basename(tif_path)}: {e}")
            # --- END NEW ---
            
            img_array = data.astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(output_png_path)
            print(f"Saved {output_png_path}")
    except Exception as e:
        print(f"Error processing {tif_path}: {e}")

if __name__ == "__main__":
    # --- Define paths ---
    # Get the parent directory of the current working directory (which is /workspace/ in your case)
    # Assuming you run this script from /workspace/Fire-DDPM/
    project_root = Path(os.getcwd()).parent 
    
    # Output directory for these raw TIFF visualizations
    output_viz_dir = project_root / "outputs" / "raw_tif_visualizations"
    output_viz_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

    # Path to the raw target .tif file that corresponds to the all-white target you saw
    # This is the target for Day N+1
    raw_target_tif_file = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-06.tif" 
    
    # Path to the raw input .tif file for the day before (Day N)
    raw_input_tif_file = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-07.tif"

    raw_input_tif_file1 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-08.tif"
    
    raw_input_tif_file2 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-09.tif"

    raw_input_tif_file3 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-10.tif"
    
    raw_input_tif_file4 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-11.tif"

    raw_input_tif_file5 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-12.tif"
    
    raw_input_tif_file6 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-13.tif"

    raw_input_tif_file7 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-14.tif"
    
    raw_input_tif_file8 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-15.tif"

    raw_input_tif_file9 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-16.tif"
    
    raw_input_tif_file10 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-17.tif"

    raw_input_tif_file11 = "/workspace/WildfireSpreadTS_Data/2019/fire_22938749/2019-06-18.tif"
    
    # Output paths for the PNGs within your project's outputs folder
    output_target_png_file = output_viz_dir / "raw_target_fire_2019-06-06.png"
    output_input_png_file = output_viz_dir / "raw_input_fire_2019-06-07.png"
    output_input_png_file1 = output_viz_dir / "raw_target_fire_2019-06-08.png"
    output_input_png_file2= output_viz_dir / "raw_input_fire_2019-06-09.png"
    output_input_png_file3 = output_viz_dir / "raw_target_fire_2019-06-10.png"
    output_input_png_file4 = output_viz_dir / "raw_input_fire_2019-06-11.png"
    output_input_png_file5= output_viz_dir / "raw_target_fire_2019-06-12.png"
    output_input_png_file6 = output_viz_dir / "raw_input_fire_2019-06-13.png"
    output_input_png_file7 = output_viz_dir / "raw_target_fire_2019-06-14.png"
    output_input_png_file8 = output_viz_dir / "raw_input_fire_2019-06-15.png"
    output_input_png_file9 = output_viz_dir / "raw_target_fire_2019-06-16.png"
    output_input_png_file10 = output_viz_dir / "raw_input_fire_2019-06-17.png"
    output_input_png_file11 = output_viz_dir / "raw_target_fire_2019-06-18.png"

    print(f"Attempting to save raw target TIFF: {raw_target_tif_file}")
    save_tif_band_as_png(raw_target_tif_file, output_target_png_file, band_index=0, scale_to_255=True)

    print(f"Attempting to save raw input TIFF: {raw_input_tif_file1}")
    save_tif_band_as_png(raw_input_tif_file1, output_input_png_file1, band_index=0, scale_to_255=True)

    print(f"Attempting to save raw input TIFF: {raw_input_tif_file2}")
    save_tif_band_as_png(raw_input_tif_file2, output_input_png_file2, band_index=0, scale_to_255=True)
   
    print(f"Attempting to save raw input TIFF: {raw_input_tif_file3}")
    save_tif_band_as_png(raw_input_tif_file3, output_input_png_file3, band_index=0, scale_to_255=True)
  
    print(f"Attempting to save raw input TIFF: {raw_input_tif_file4}")
    save_tif_band_as_png(raw_input_tif_file4, output_input_png_file4, band_index=0, scale_to_255=True)
  
    print(f"Attempting to save raw input TIFF: {raw_input_tif_file5}")
    save_tif_band_as_png(raw_input_tif_file5, output_input_png_file5, band_index=0, scale_to_255=True)
  
    print(f"Attempting to save raw input TIFF: {raw_input_tif_file6}")
    save_tif_band_as_png(raw_input_tif_file6, output_input_png_file6, band_index=0, scale_to_255=True)
  
    print(f"Attempting to save raw input TIFF: {raw_input_tif_file7}")
    save_tif_band_as_png(raw_input_tif_file7, output_input_png_file7, band_index=0, scale_to_255=True)
  
    print(f"Attempting to save raw input TIFF: {raw_input_tif_file8}")
    save_tif_band_as_png(raw_input_tif_file8, output_input_png_file8, band_index=0, scale_to_255=True)
  
    print(f"Attempting to save raw input TIFF: {raw_input_tif_file9}")
    save_tif_band_as_png(raw_input_tif_file9, output_input_png_file9, band_index=0, scale_to_255=True)
  
    print(f"Attempting to save raw input TIFF: {raw_input_tif_file10}")
    save_tif_band_as_png(raw_input_tif_file10, output_input_png_file10, band_index=0, scale_to_255=True)
  
    print(f"Attempting to save raw input TIFF: {raw_input_tif_file11}")
    save_tif_band_as_png(raw_input_tif_file11, output_input_png_file11, band_index=0, scale_to_255=True)
  