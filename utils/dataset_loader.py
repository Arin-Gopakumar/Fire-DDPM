#gpts dataloader
import os
import numpy as np
from PIL import Image # Still imported, but less used for .npy targets
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime, timedelta
import logging # Added for better logging

class WildfireDataset(Dataset):
    """
    Custom Dataset for loading wildfire conditioning inputs and next-day target masks.
    Assumes conditioning inputs and targets are .npy files.
    Filenames are expected to contain a unique event ID and a date (YYYY-MM-DD or YYYYMMDD)
    or a sequential day index to allow for temporal matching.
    Example: 'fire_event_A_2023-01-01.npy'
    """
    def __init__(self, data_dir, split="train", image_size=(64,64), target_transform=None, input_transform=None):
        """
        Args:
            data_dir (str): Path to the base data directory (e.g., "../data").
            split (str): "train", "val", or "test".
            image_size (tuple): Target (H,W) for images.
            target_transform (callable, optional): Optional transform to be applied on a target.
                                                  Note: Binarization is done internally.
            input_transform (callable, optional): Optional transform to be applied on an input.
        """
        self.inputs_dir = os.path.join(data_dir, split, "inputs")
        self.targets_dir = os.path.join(data_dir, split, "targets")
        self.image_size = image_size

        self.samples = [] # This will store tuples of (input_path_day_N, target_path_day_N+1, sample_id)

        # Get all input and target files
        all_input_files = sorted([f for f in os.listdir(self.inputs_dir) if f.endswith(".npy")])
        all_target_files = sorted([f for f in os.listdir(self.targets_dir) if f.endswith(".npy")])

        # Map base filenames to their full paths for quick lookup
        input_paths_map = {os.path.splitext(f)[0]: os.path.join(self.inputs_dir, f) for f in all_input_files}
        target_paths_map = {os.path.splitext(f)[0]: os.path.join(self.targets_dir, f) for f in all_target_files}

        # Group input files by fire event ID
        # This function needs to be adapted if your naming convention is different.
        def _parse_filename(filename):
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split('_')
            
            if len(parts) < 3: # Expect at least 'fire', 'event_id', 'date_str'
                raise ValueError(f"Filename '{filename}' does not have enough parts for parsing.")

            # Assuming the date string is the last part
            date_part = parts[-1]
            event_id_parts = parts[:-1] # All parts before the date

            try:
                # Try parsing as YYYY-MM-DD (e.g., 2018-01-03)
                parsed_date = datetime.strptime(date_part, '%Y-%m-%d')
                event_id = '_'.join(event_id_parts)
                return event_id, parsed_date
            except ValueError:
                pass # Not YYYY-MM-DD, try other formats

            try:
                # Try parsing as YYYYMMDD (original logic)
                if len(date_part) == 8 and date_part.isdigit():
                    parsed_date = datetime.strptime(date_part, '%Y%m%d')
                    event_id = '_'.join(event_id_parts)
                    return event_id, parsed_date
            except ValueError:
                pass # Not YYYYMMDD, try other formats

            try:
                # Try parsing as sequential day index (e.g., _day_001)
                if date_part.isdigit():
                    day_index = int(date_part)
                    event_id = '_'.join(event_id_parts)
                    return event_id, day_index
            except ValueError:
                pass # Not a sequential day index

            raise ValueError(f"Filename '{filename}' does not match any expected naming convention (e.g., eventID_YYYY-MM-DD.npy, eventID_YYYYMMDD.npy, eventID_day_XXX.npy).")

        # Group files by event_id and sort them temporally
        event_grouped_inputs = {}
        for f in all_input_files:
            try:
                event_id, time_id = _parse_filename(f)
                if event_id not in event_grouped_inputs:
                    event_grouped_inputs[event_id] = []
                event_grouped_inputs[event_id].append((time_id, f))
            except ValueError as e:
                logging.warning(f"Skipping input file {f} due to parsing error: {e}")
                continue

        # Create (input_day_N, target_day_N+1) pairs
        for event_id, files_in_event in event_grouped_inputs.items():
            # Sort files within each event by their time_id (date or day_index)
            files_in_event.sort(key=lambda x: x[0])

            for i in range(len(files_in_event) - 1): # Iterate up to the second to last file
                current_day_time_id, current_day_filename = files_in_event[i]
                next_day_time_id, next_day_filename = files_in_event[i+1]

                # Ensure the 'next_day_filename' is indeed the *next* chronological step for the target
                # This is crucial for time-series prediction
                is_consecutive = False
                if isinstance(current_day_time_id, datetime):
                    # Check if next_day_time_id is exactly one day after current_day_time_id
                    is_consecutive = (next_day_time_id - current_day_time_id).days == 1
                elif isinstance(current_day_time_id, int):
                    # Check if next_day_time_id is exactly one index after current_day_time_id
                    is_consecutive = (next_day_time_id - current_day_time_id) == 1

                if not is_consecutive:
                    logging.warning(f"Non-consecutive days detected for event {event_id} between {current_day_filename} and {next_day_filename}. Skipping this pair.")
                    continue # Skip if not strictly consecutive for target

                current_day_basename = os.path.splitext(current_day_filename)[0]
                next_day_basename = os.path.splitext(next_day_filename)[0]

                # Check if the corresponding target file exists for the *next day*
                if next_day_basename in target_paths_map:
                    self.samples.append((
                        input_paths_map[current_day_basename],
                        target_paths_map[next_day_basename],
                        current_day_basename # Use current day's ID for reference
                    ))
                else:
                    logging.warning(f"No target file found for next day '{next_day_basename}' corresponding to input '{current_day_basename}'. Skipping.")

        if not self.samples:
            raise RuntimeError(f"No valid (input_day_N, target_day_N+1) pairs found in {split} directory. "
                               "Please check file naming convention, data structure, and ensure targets for N+1 exist.")
        logging.info(f"Loaded {len(self.samples)} valid (input, next-day target) pairs for {split} split.")


        if target_transform is None:
            # For .npy masks, we directly load, binarize, and convert to tensor.
            # No need for PIL-based transforms here.
            self.target_transform = None # Will handle directly in __getitem__
        else:
            self.target_transform = target_transform

        if input_transform is None:
            self.input_transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.from_numpy(x.astype(np.float32)))
            ])
        else:
            self.input_transform = input_transform


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_path, sample_id = self.samples[idx]

        try:
            # Load conditioning input (.npy)
            conditioning_input = np.load(input_path)
            
            # Ensure conditioning_input is (C, H, W) and float32
            if conditioning_input.ndim == 2: # (H,W) -> (1,H,W)
                conditioning_input = np.expand_dims(conditioning_input, axis=0)
            
            # Apply input transforms
            if self.input_transform:
                conditioning_input = self.input_transform(conditioning_input)
            
            # Ensure it's FloatTensor
            if not isinstance(conditioning_input, torch.FloatTensor):
                 conditioning_input = conditioning_input.float()


            # Load target fire mask (.npy)
            target_mask_np = np.load(target_path).astype(np.float32)

            # Binarize and ensure [0, 1] range for the target mask
            # Assuming target mask values are either 0 or some positive value for fire
            target_mask = (target_mask_np > 0.5).astype(np.float32) # Use a small threshold to catch any non-zero fire pixels

            # Add channel dimension (1, H, W) and convert to tensor
            target_mask = torch.from_numpy(target_mask).unsqueeze(0)

            # Apply optional target_transform if provided (e.g., for further resizing or specific ops)
            if self.target_transform:
                target_mask = self.target_transform(target_mask)

            return {"condition": conditioning_input, "target": target_mask, "id": sample_id}
        except Exception as e:
            logging.error(f"Error loading data for sample ID {sample_id} (input: {input_path}, target: {target_path}): {e}")
            # For robustness in evaluation, return dummy data for problematic samples.
            # In a real training scenario, you might want to raise the error or log more aggressively.
            dummy_cond = torch.zeros((24, self.image_size[0], self.image_size[1]), dtype=torch.float32) # Assuming 24 input channels
            dummy_target = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)
            
            return {"condition": dummy_cond, "target": dummy_target, "id": f"error_sample_{sample_id}"}




"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WildfireDataset(Dataset):
"""
    #Custom Dataset for loading wildfire conditioning inputs and target masks.
    #Assumes conditioning inputs are .npy files and targets are .png files.
"""
    def __init__(self, data_dir, split="train", image_size=(64,64), target_transform=None, input_transform=None):
        """
        #Args:
            #data_dir (str): Path to the base data directory (e.g., "../data").
            #split (str): "train", "val", or "test".
            #image_size (tuple): Target (H,W) for images.
            #target_transform (callable, optional): Optional transform to be applied on a target.
            #input_transform (callable, optional): Optional transform to be applied on an input.
"""
        self.inputs_dir = os.path.join(data_dir, split, "inputs")
        self.targets_dir = os.path.join(data_dir, split, "targets")
        self.image_size = image_size

        self.input_files = sorted([f for f in os.listdir(self.inputs_dir) if f.endswith(".npy")])
        self.target_files = sorted([f for f in os.listdir(self.targets_dir) if f.endswith(".npy")])

        # Basic check for consistency
        if len(self.input_files) != len(self.target_files):
            print(f"Warning: Mismatch in number of input and target files in {split} directory.")
            # Optionally, try to match them by base name
            input_basenames = {os.path.splitext(f)[0] for f in self.input_files}
            target_basenames = {os.path.splitext(f)[0] for f in self.target_files}
            common_basenames = sorted(list(input_basenames.intersection(target_basenames)))
            
            self.input_files = [f"{bn}.npy" for bn in common_basenames]
            self.target_files = [f"{bn}.png" for bn in common_basenames]
            print(f"Found {len(common_basenames)} matching samples for {split}.")


        if target_transform is None:
            self.target_transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST), # Use NEAREST for masks
                transforms.ToTensor(), # Converts to [0, 1] range for single channel PIL image
                transforms.Lambda(lambda x: (x > 0.5).float()) # Ensure binary 0 or 1
            ])
        else:
            self.target_transform = target_transform

        if input_transform is None:
            # For .npy files, normalization should ideally be done during data prep (prepare_data.py)
            # If inputs are already normalized to [-1, 1] or [0, 1], ToTensor() is enough.
            # If they are images (e.g. PNG stacks), more transforms might be needed here.
            self.input_transform = transforms.Compose([
                # Assuming input .npy is (C,H,W) and already normalized
                transforms.Lambda(lambda x: torch.from_numpy(x.astype(np.float32)))
                # If your .npy files are not (C,H,W) or need resizing/normalization here:
                # transforms.Lambda(lambda x: torch.from_numpy(x.astype(np.float32))),
                # transforms.ToPILImage() # if numpy array is not in C, H, W and needs PIL transforms
                # transforms.Resize(self.image_size),
                # transforms.ToTensor(), # This scales [0,255] to [0,1]. If data is already [0,1] or [-1,1] it's fine.
                # transforms.Normalize(mean=[...], std=[...]) # If you want to normalize here based on dataset stats
            ])
        else:
            self.input_transform = input_transform


    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_filename = self.input_files[idx]
        target_filename = self.target_files[idx] # Assumes direct correspondence by sort order

        # Check filename match (optional, but good for sanity)
        input_base = os.path.splitext(input_filename)[0]
        target_base = os.path.splitext(target_filename)[0]
        if input_base != target_base:
            raise ValueError(f"Mismatched file names at index {idx}: {input_filename} vs {target_filename}")

        input_path = os.path.join(self.inputs_dir, input_filename)
        target_path = os.path.join(self.targets_dir, target_filename)

        try:
            # Load conditioning input (.npy)
            # Expected shape: (C, H, W), float, ideally normalized in prepare_data.py
            conditioning_input = np.load(input_path)
            
            # Ensure conditioning_input is (C, H, W) and float32
            if conditioning_input.ndim == 2: # (H,W) -> (1,H,W)
                conditioning_input = np.expand_dims(conditioning_input, axis=0)
            # If H, W are first, permute: (H, W, C) -> (C, H, W)
            # This depends on how you save your .npy files in prepare_data.py
            # Example: if np.save(..., arr) where arr is (H,W,C)
            # if conditioning_input.shape[-1] == num_channels:
            #    conditioning_input = np.transpose(conditioning_input, (2,0,1))

            # Apply input transforms
            if self.input_transform:
                conditioning_input = self.input_transform(conditioning_input)
            
            # Ensure it's FloatTensor
            if not isinstance(conditioning_input, torch.FloatTensor):
                 conditioning_input = conditioning_input.float()


            # Load target fire mask (.png)
            # Expected: Grayscale, uint8. Will be converted to binary tensor [0,1] by target_transform
            target_mask = np.load(target_path).astype(np.float32)  # Load .npy

            # Binarize: assume mask is 0 or 255 and normalize to [0, 1]
            target_mask = (target_mask > 0.5).astype(np.float32)

            # Add channel dimension (1, H, W)
            target_mask = torch.from_numpy(target_mask).unsqueeze(0)


            return {"condition": conditioning_input, "target": target_mask, "id": input_base}
        except Exception as e:
            print(f"Error loading data for {input_filename} or {target_filename}: {e}")
            # Return dummy data or skip by raising an error that DataLoader can handle (if configured)
            # For simplicity, returning None or raising error might be best.
            # Here, we'll try to return a placeholder to avoid crashing the batch if one file is bad.
            # This is not ideal for training.
            dummy_cond = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32) # Adjust C if known
            dummy_target = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)
            if self.input_transform: dummy_cond = self.input_transform(np.zeros((1, self.image_size[0], self.image_size[1])))
            if self.target_transform: dummy_target = self.target_transform(Image.new("L", self.image_size))

            return {"condition": dummy_cond, "target": dummy_target, "id": "error_sample"}
"""