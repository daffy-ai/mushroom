import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.model_selection import train_test_split

class MushroomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(data_dir='data', batch_size=32):
    # Collect all images and labels from nested directories if needed.
    images = []
    labels = []
    category_map = {
        'healthy': 0,
        'single infected': 1,
        'single_infected': 1,
        'mixed infected': 1,
        'mixed_infected': 1,
    }

    for root, dirs, _ in os.walk(data_dir):
        for directory in dirs:
            key = directory.replace('-', ' ').strip().lower()
            if key in category_map:
                folder_path = os.path.join(root, directory)
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append(os.path.join(folder_path, img_name))
                        labels.append(category_map[key])

    if len(images) == 0:
        raise FileNotFoundError(
            f"No image files found under {data_dir}. Make sure the dataset is extracted and contains Healthy / Single_Infected / Mixed_Infected folders."
        )

    # Split
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=0.15, random_state=42, stratify=labels
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, test_size=0.176, random_state=42, stratify=train_val_labels
    )  # ~15% of total for val

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MushroomDataset(train_images, train_labels, train_transform)
    val_dataset = MushroomDataset(val_images, val_labels, val_test_transform)
    test_dataset = MushroomDataset(test_images, test_labels, val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader