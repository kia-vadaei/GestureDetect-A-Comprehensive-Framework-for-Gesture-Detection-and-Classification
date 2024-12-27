import json
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GestureDetectDataset(Dataset):
    def __init__(self, json_file, mode='train'):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.image_paths = [item['image_path'] for item in data]
        self.labels = [item['class'] for item in data]
        self.bboxes = [item['bbox'] for item in data]
        self.mode = mode
        
        # Split the data
        train_paths, test_paths, train_labels, test_labels, train_bboxes, test_bboxes = train_test_split(
            self.image_paths, self.labels, self.bboxes, test_size=0.2, random_state=42, stratify=self.labels)
        
        train_paths, val_paths, train_labels, val_labels, train_bboxes, val_bboxes = train_test_split(
            train_paths, train_labels, train_bboxes, test_size=0.2, random_state=42, stratify=train_labels)
        
        if mode == 'train':
            self.image_paths, self.labels, self.bboxes = train_paths, train_labels, train_bboxes
            self.transform = A.Compose([
                A.Resize(640, 640),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.RandomResizedCrop(640, 640, scale=(0.8, 1.0), p=0.5),
                A.MotionBlur(p=0.2),
                A.GaussianBlur(p=0.1),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        elif mode == 'val':
            self.image_paths, self.labels, self.bboxes = val_paths, val_labels, val_bboxes
            self.transform = A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        elif mode == 'test':
            self.image_paths, self.labels, self.bboxes = test_paths, test_labels, test_bboxes
            self.transform = A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            raise ValueError("Mode should be 'train', 'val', or 'test'")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        bbox = self.bboxes[idx]  # Get the bounding box for the current image
        label = self.labels[idx]  # Get the class label for the current image
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Pass the bbox and label to the transform
        transformed = self.transform(
            image=image,
            bboxes=[bbox],
            class_labels=[label]
        )
        
        image = transformed['image']
        bboxes = transformed['bboxes']
        class_labels = transformed['class_labels']
        
        return image, bboxes, class_labels
    