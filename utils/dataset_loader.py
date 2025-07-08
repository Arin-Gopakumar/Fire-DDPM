import os
import numpy as np
from PIL import Image # Still imported, but less used for .npy targets
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime, timedelta
import logging

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
        def _parse_filename(filename):
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split('_')
            
            if len(parts) < 3: 
                raise ValueError(f"Filename '{filename}' does not have enough parts for parsing.")

            date_part = parts[-1]
            event_id_parts = parts[:-1]

            try:
                parsed_date = datetime.strptime(date_part, '%Y-%m-%d')
                event_id = '_'.join(event_id_parts)
                return event_id, parsed_date
            except ValueError:
                pass

            try:
                if len(date_part) == 8 and date_part.isdigit():
                    parsed_date = datetime.strptime(date_part, '%Y%m%d')
                    event_id = '_'.join(event_id_parts)
                    return event_id, parsed_date
            except ValueError:
                pass

            try:
                if date_part.isdigit():
                    day_index = int(date_part)
                    event_id = '_'.join(event_id_parts)
                    return event_id, day_index
            except ValueError:
                pass

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
            files_in_event.sort(key=lambda x: x[0])

            for i in range(len(files_in_event) - 1):
                current_day_time_id, current_day_filename = files_in_event[i]
                next_day_time_id, next_day_filename = files_in_event[i+1]

                is_consecutive = False
                if isinstance(current_day_time_id, datetime):
                    is_consecutive = (next_day_time_id - current_day_time_id).days == 1
                elif isinstance(current_day_time_id, int):
                    is_consecutive = (next_day_time_id - current_day_time_id) == 1

                if not is_consecutive:
                    logging.warning(f"Non-consecutive days detected for event {event_id} between {current_day_filename} and {next_day_filename}. Skipping this pair.")
                    continue

                current_day_basename = os.path.splitext(current_day_filename)[0]
                next_day_basename = os.path.splitext(next_day_filename)[0]

                if next_day_basename in target_paths_map:
                    self.samples.append((
                        input_paths_map[current_day_basename],
                        target_paths_map[next_day_basename],
                        current_day_basename
                    ))
                else:
                    logging.warning(f"No target file found for next day '{next_day_basename}' corresponding to input '{current_day_basename}'. Skipping.")

        if not self.samples:
            raise RuntimeError(f"No valid (input_day_N, target_day_N+1) pairs found in {split} directory. "
                               "Please check file naming convention, data structure, and ensure targets for N+1 exist.")
        logging.info(f"Loaded {len(self.samples)} valid (input, next-day target) pairs for {split} split.")


        if target_transform is None:
            # Target is already (1, H, W) from prepare_data.py, so no need for transforms.
            self.target_transform = None 
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
            # Expected shape: (C, H, W), float, ideally normalized in prepare_data.py
            conditioning_input = np.load(input_path)
            
            # Ensure conditioning_input is (C, H, W) and float32
            # No need for expand_dims here if prepare_data.py already saves (C,H,W)
            # if conditioning_input.ndim == 2: # (H,W) -> (1,H,W)
            #     conditioning_input = np.expand_dims(conditioning_input, axis=0)

            # Apply input transforms
            if self.input_transform:
                conditioning_input = self.input_transform(conditioning_input)
            
            # Ensure it's FloatTensor
            if not isinstance(conditioning_input, torch.FloatTensor):
                 conditioning_input = conditioning_input.float()


            # Load target mask (.npy)
            # Expected: (1, H, W) with values 0.0 or 255.0 from prepare_data.py
            target_mask = np.load(target_path).astype(np.float32)

            # --- FIX: Ensure target_mask is (1, H, W) and convert to tensor directly ---
            # If prepare_data.py saves (H, W), then add channel dim here.
            # If prepare_data.py saves (1, H, W), then just convert to tensor.
            if target_mask.ndim == 2: # If it's (H,W)
                target_mask = np.expand_dims(target_mask, axis=0) # Make it (1, H, W)
            # No binarization or scaling to [0,1] here, as it's done in prepare_data.py
            # and train/evaluate.py will handle 0.0/255.0 to 0/1 conversion.
            target_mask_tensor = torch.from_numpy(target_mask)
            # --- END FIX ---

            # Apply optional target_transform if provided (e.g., for further resizing or specific ops)
            if self.target_transform:
                target_mask_tensor = self.target_transform(target_mask_tensor)

            return {"condition": conditioning_input, "target": target_mask_tensor, "id": sample_id}
        except Exception as e:
            logging.error(f"Error loading data for sample ID {sample_id} (input: {input_path}, target: {target_path}): {e}")
            dummy_cond = torch.zeros((24, self.image_size[0], self.image_size[1]), dtype=torch.float32) # Adjust C if known
            dummy_target = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)
            
            return {"condition": dummy_cond, "target": dummy_target, "id": f"error_sample_{sample_id}"}

