import os

import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
import pandas
import kagglehub


class RetinalDiseaseDetectionDataset(Dataset):
    
    def __init__(self, img_dir, annotations_file, transforms=None):
        self.img_dir = img_dir
        self.labels = pandas.read_csv(annotations_file)
        self.transforms = transforms
        
    
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, idx):
        img_name = self.labels.iat[idx, 0]
        labels =  torch.tensor([self.labels.iat[idx, 1], self.labels.iat[idx, 2]])
        
        img = None
        try:
            with Image.open(os.path.join(self.img_dir, img_name), "r") as img_buf:
                img = img_buf.convert("RGB")
        except FileNotFoundError | PIL.UnidentifiedImageError as e:
            print(f"Error loading image {img_name} at index {idx}: {e}")
            img = Image.new("RGB", (2000, 1328), color="black")
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, labels
    
    
    def class_dist(self):
        retino_class_dist = (self.labels.iloc[:,1].value_counts().sort_index() / len(self.labels)).to_numpy()
        edema_class_dist = (self.labels.iloc[:,2].value_counts().sort_index() / len(self.labels)).to_numpy()
        return retino_class_dist, edema_class_dist
    
    
    def raw_class_dist(self):
        retino_class_dist = self.labels.iloc[:,1].value_counts().sort_index().to_numpy()
        edema_class_dist = self.labels.iloc[:,2].value_counts().sort_index().to_numpy()
        return retino_class_dist, edema_class_dist


def download_dataset(batch_size, num_workers):
    
    # Download Dataset
    os.environ['KAGGLEHUB_CACHE'] = 'data'

    dataset_path = kagglehub.dataset_download("mohamedabdalkader/retinal-disease-detection")
    print('Dataset Downloaded! Path: {}'.format(dataset_path))


    # Create DataLoader
    
    transforms = tfs.Compose([
        tfs.Resize((800, 528)),
        tfs.ToTensor()
    ])
    
    # Train DataLoader
    train_img_dir =  os.path.join(dataset_path, r"Diabetic Retinopathy\train\images")
    train_annotations_path =  os.path.join(dataset_path, r"Diabetic Retinopathy\train\annotations.csv")

    train_data = RetinalDiseaseDetectionDataset(train_img_dir, train_annotations_path, transforms=transforms)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=True)

    # Test DataLoader
    test_img_dir =  os.path.join(dataset_path, r"Diabetic Retinopathy\test\images")
    test_annotations_path = os.path.join(dataset_path, r"Diabetic Retinopathy\test\annotations.csv")

    test_data = RetinalDiseaseDetectionDataset(test_img_dir, test_annotations_path, transforms=transforms)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
    
    # Validation DataLoader
    val_img_dir = os.path.join(dataset_path, r'Diabetic Retinopathy\valid\images')
    val_annotations_path = os.path.join(dataset_path, r'Diabetic Retinopathy\valid\annotations.csv')

    val_data = RetinalDiseaseDetectionDataset(val_img_dir, val_annotations_path, transforms=transforms)

    val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
    
    return train_dataloader, test_dataloader, val_dataloader