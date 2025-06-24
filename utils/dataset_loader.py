import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WildfireDataset(Dataset):
    """
    Custom Dataset for loading wildfire conditioning inputs and target masks.
    Assumes conditioning inputs are .npy files and targets are .png files.
    """
    def __init__(self, data_dir, split="train", image_size=(64,64), target_transform=None, input_transform=None):
        """
        Args:
            data_dir (str): Path to the base data directory (e.g., "../data").
            split (str): "train", "val", or "test".
            image_size (tuple): Target (H,W) for images.
            target_transform (callable, optional): Optional transform to be applied on a target.
            input_transform (callable, optional): Optional transform to be applied on an input.
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