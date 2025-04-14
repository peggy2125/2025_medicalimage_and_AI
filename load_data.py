import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from glob import glob
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm 


class MuraDataset(Dataset):
    ''' mode - train or valid '''
    def __init__(self, csv_path, data_path, mode='train', transform=None):
        self.data_path = data_path # project 資料夾
        self.data = pd.read_csv(csv_path, header=None, names=['path', 'label'])
        self.mode = mode
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
    def __getitem__(self, index):
        study_folder = os.path.join(self.data_path, self.data.iloc[index]['path'])
        label = self.data.iloc[index]['label']
        image_paths = glob(os.path.join(study_folder, '*.png'))
        images = []
        img_paths = [] # 同時load path，如evaluate時需要用到
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('L')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
                img_paths.append(img_path)
            except Exception as e:
                print(f"Failed to load image: {img_path} | Error: {e}")
        images = torch.stack(images)
        label = torch.tensor(label, dtype=torch.long)

        # 如果是驗證模式，返回圖片、標籤和圖片路徑
        if self.mode == 'valid':
            return images, label, img_paths
        else:
            return images, label

    def __len__(self):
        return len(self.data)

def mura_collate_fn(batch):
    image_stacks = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    if len(batch[0]) == 3:  # 模式為 'valid' 時，返回圖片路徑
        image_paths = [item[2] for item in batch]
        return image_stacks, labels, image_paths
    else:  # 'train' 模式
        return image_stacks, labels

    
def load_mura_data(data_path, mode='train', batch_size=8, num_workers=0):
    assert mode in ['train', 'valid'], "mode must be 'train' or 'valid'"
    csv_path = os.path.join(data_path, f'MURA-v1.1', f'{mode}_labeled_studies.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    dataset = MuraDataset(csv_path, data_path, mode=mode)
    # create data loaders
    if mode == 'train':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=mura_collate_fn)
    elif mode == 'valid':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=mura_collate_fn)
    print(f"Number of {mode} samples: {len(dataset)}") 
    return data_loader
    