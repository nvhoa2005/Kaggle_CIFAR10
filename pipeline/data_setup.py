import os
import torch
import pathlib
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from typing import Tuple

from pipeline import const

import importlib
importlib.reload(const)

NUM_WORKERS = const.NUM_WORKERS
BATCH_SIZE = const.BATCH_SIZE
CLASSES = const.CLASSES

# Write a custom dataset class (inherits from torch.utils.data.Dataset)
class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}
# 1. Subclass torch.utils.data.Dataset
class TrainDatasetCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, labels: pd.DataFrame, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all .png images and sort them by filename (numerically)
        self.paths = sorted(pathlib.Path(targ_dir).glob("*.png"), key=lambda x: int(x.stem))
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = CLASSES, class_to_idx
        # Set up labels
        self.labels = labels

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.labels["label"][index]
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)

# 1. Subclass torch.utils.data.Dataset
class TestDatasetCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, labels: pd.DataFrame, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all .png images and sort them by filename (numerically)
        self.paths = sorted(pathlib.Path(targ_dir).glob("*.png"), key=lambda x: int(x.stem))
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = CLASSES, class_to_idx
        # Set up labels
        self.labels = labels

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_id = int(self.labels.iloc[index]['id'])
        image_path = self.paths[index]
        img = Image.open(image_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, image_id  # trả về ảnh và id



def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    train_transform: transforms.Compose, 
    test_transform: transforms.Compose,
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    
    # Use TrainDatasetCustom to create dataset(s)
    train_data_custom = TrainDatasetCustom(targ_dir=train_dir,
                                      labels=train_labels,
                                      transform=train_transform)
    test_data_custom = TestDatasetCustom(targ_dir=test_dir,
                                      labels=test_labels,
                                      transform=test_transform)
    
    # Get class names
    class_names = train_data_custom.classes
    
    # Turn images into data loaders
    train_dataloader = DataLoader(
      train_data_custom,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
    )
    
    test_dataloader = DataLoader(
      test_data_custom,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names