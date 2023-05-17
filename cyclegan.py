import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.utils import save_image
from data_loader import get_loader
from data_loader import plot_images
from discriminator import Discriminator
from generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 생성된 image plot
def plot_generated_images(fake_A, fake_B, real_A, real_B, epoch, batch_idx):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(fake_A[0].permute(1, 2, 0).cpu().detach().numpy())
    axes[0, 0].set_title('Generated A')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(fake_B[0].permute(1, 2, 0).cpu().detach().numpy())
    axes[0, 1].set_title('Generated B')
    axes[0, 1].axis('off')
    axes[1, 0].imshow(real_A[0].permute(1, 2, 0).cpu().detach().numpy())
    axes[1, 0].set_title('Real A')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(real_B[0].permute(1, 2, 0).cpu().detach().numpy())
    axes[1, 1].set_title('Real B')
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_images/epoch_{epoch+1}_batch_{batch_idx+1}.png")
    plt.close()


def train(num_epochs, dataloader, G_AB, G_BA, D_A, D_B, criterion_GAN, criterion_cycle, optimizer_G, optimizer_D):
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            # 실제 이미지 데이터 가져오기
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # 가짜 이미지 생성
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            # Train Discriminator A
            optimizer_D.zero_grad()
            
            # Real images
            pred_real = D_A(real_A)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            # Fake images
            pred_fake = D_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) / 2
            loss_D_A.backward()
            optimizer_D.step()

            # Train Discriminator B
            optimizer_D.zero_grad()
            # Real images
            pred_real = D_B(real_B)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            # Fake images
            pred_fake = D_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) / 2
            loss_D_B.backward()
            optimizer_D.step()

            # Train Generators
            optimizer_G.zero_grad()
            # Adversarial loss
            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
            loss_GAN = (loss_GAN_BA + loss_GAN_AB) / 2
            # Cycle consistency loss
            reconstructed_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(reconstructed_A, real_A)
            reconstructed_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(reconstructed_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            # Total loss
            loss_G = loss_GAN + (lambda_cycle * loss_cycle)
            loss_G.backward()
            optimizer_G.step()

            # 일정 주기로 이미지 저장 및 출력
            if (i + 1) % 10 == 0:
                # 생성된 이미지 저장
                save_dir = "generated_images"
                os.makedirs(save_dir, exist_ok=True)  # 존재하지 않으면 directory 생성
                save_image(fake_A, os.path.join(save_dir, f"fake_A_{epoch+1}_{i+1}.png"), normalize=True)
                save_image(fake_B, os.path.join(save_dir, f"fake_B_{epoch+1}_{i+1}.png"), normalize=True)
                
                # Plot generated images at regular intervals
                plot_generated_images(fake_A, fake_B, real_A, real_B, epoch, i)
    
                # 진행 상황 출력
                print(f"[Epoch {epoch+1}/{num_epochs}], "
                      f"Batch {i+1}/{len(data_loader)}, "
                      f"Discriminator A Loss: {loss_D_A.item():.4f}, "
                      f"Discriminator B Loss: {loss_D_B.item():.4f}, "
                      f"Generator Loss: {loss_G.item():.4f}")
    
        # 일정 주기로 모델 저장
        if (epoch + 1) % 10 == 0:
            save_dir = "saved_models"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(G_AB.state_dict(), os.path.join("saved_models", f"generator_G_AB_{epoch+1}.pth"))
            torch.save(G_BA.state_dict(), os.path.join("saved_models", f"generator_G_BA_{epoch+1}.pth"))
            torch.save(D_A.state_dict(), os.path.join("saved_models", f"discriminator_D_A_{epoch+1}.pth"))
            torch.save(D_B.state_dict(), os.path.join("saved_models", f"discriminator_D_B_{epoch+1}.pth"))
        
        # GPU 메모리 비우기
        torch.cuda.empty_cache()
            
def test(G_AB, G_BA, D_A, D_B, test_dataloader, device):
    # 모델을 device에 할당
    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)
    
    # 테스트 모드로 설정
    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()
    
    # 테스트 실행
    with torch.no_grad():
        for i, (real_A, _) in enumerate(test_dataloader):
            real_A = real_A.to(device)
            
            # 가짜 이미지 생성
            fake_B = G_AB(real_A)
            fake_A = G_BA(fake_B)
            
            # 이미지 저장
            save_image(fake_B, f"test_results/fake_B_{i+1}.png")
            save_image(fake_A, f"test_results/fake_A_{i+1}.png")
    
    print("Test complete.")
    
if __name__ == "__main__":
    data_loader = get_loader(root_path="D:\CAU\eclass\컴퓨터비전\\archive (2)",
                         image_size=256,
                         batch_size=8,
                         num_workers=4)
    
    # 하이퍼파라미터 세팅
    num_epochs = 200
    lr = 0.0002
    lambda_cycle = 10
    
    # 모델 및 데이터 로더 초기화
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    
    # 손실 함수 및 최적화 알고리즘 설정
    criterion_GAN = nn.BCELoss()
    criterion_cycle = nn.L1Loss()
    
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    
    train(num_epochs, data_loader, G_AB, G_BA, D_A, D_B, criterion_GAN, criterion_cycle, optimizer_G, optimizer_D)
    
    #test_dataloader = get_dataloader("test_data", batch_size=1, num_workers=4)
    

    
    