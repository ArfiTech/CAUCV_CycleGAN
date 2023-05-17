import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils

class CycleGAN_Dataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.dir_A = os.path.join(root_path, 'trainA')
        self.dir_B = os.path.join(root_path, 'trainB')
        self.A_paths = sorted(os.listdir(self.dir_A))
        self.B_paths = sorted(os.listdir(self.dir_B))

    def __getitem__(self, index):
        A_path = os.path.join(self.dir_A, self.A_paths[index % len(self.A_paths)])
        B_path = os.path.join(self.dir_B, self.B_paths[random.randint(0, len(self.B_paths) - 1)])
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img}

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

def get_loader(root_path, image_size, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize(int(image_size*1.12), Image.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CycleGAN_Dataset(root_path, transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)

    return data_loader

def plot_images(images):
    # 이미지 배치를 그리드로 변환
    grid = vutils.make_grid(images, nrow=8, padding=2, normalize=True)
    
    # 이미지를 numpy 배열로 변환
    grid = grid.cpu().numpy().transpose((1, 2, 0))
    
    # 이미지 플롯
    plt.figure(figsize=(12, 6))
    plt.axis("off")
    plt.imshow(grid)
    plt.show()

if __name__ == "__main__":
    data_loader = get_loader(root_path="D:\CAU\eclass\컴퓨터비전\\archive (2)",
                         image_size=256,
                         batch_size=32,
                         num_workers=4)
    
    data_loader_iter = iter(data_loader)
    batch = next(data_loader_iter)
    
    plot_images(batch['B'])
    