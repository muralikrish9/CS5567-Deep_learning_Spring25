
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MOT16Dataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        with open(annotations_file, 'rb') as f:
            self.annotations = pickle.load(f)
        
        self.image_paths = []
        self.targets = []

        for seq, data in self.annotations.items():
            seq_dir = os.path.join(data_dir, seq, 'img1')
            for frame_data in data:
                frame_num = int(frame_data[0])
                img_path = os.path.join(seq_dir, f'{frame_num:06d}.jpg')
                self.image_paths.append(img_path)
                self.targets.append(frame_data)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        target_data = self.targets[idx]
        x_min = target_data[2]
        y_min = target_data[3]
        width = target_data[4]
        height = target_data[5]
        x_max = x_min + width
        y_max = y_min + height
        boxes = [[x_min, y_min, x_max, y_max]]
        labels = [1] # 1 for pedestrian

        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32), "labels": torch.as_tensor(labels, dtype=torch.int64)}

        if self.transform:
            image = self.transform(image)

        return image, target

def get_transform(train):
    trans = []
    trans.append(transforms.ToTensor())
    if train:
        trans.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5))
        trans.append(transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
    return transforms.Compose(trans)
