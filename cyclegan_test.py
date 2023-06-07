import os
import torch
from generator import Generator
from data_loader import get_loader
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(G_AB, G_BA, test_dataloader, device):
    # 모델을 device에 할당
    G_AB.to(device)
    G_BA.to(device)
    
    # 테스트 모드로 설정
    G_AB.eval()
    G_BA.eval()
    
    # 테스트 실행
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # 실제 이미지 데이터 가져오기
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            
            # 이미지 이름 출력
            batch_size_A = real_A.size(0)  # 배치 크기 가져오기
            batch_size_B = real_B.size(0)
            
            for j in range(batch_size_A):
                print(f"Image name: {test_dataloader.dataset.A_paths[i * test_dataloader.batch_size + j]}")
            print(f"{i}th A finished")
            
            for j in range(batch_size_B):
                print(f"Image name: {test_dataloader.dataset.B_paths[i * test_dataloader.batch_size + j]}")
            print(f"{i}th B finished")
            
            # 가짜 이미지 생성
            fake_B = G_AB(real_A) # A->B 변환 확인
            fake_A = G_BA(real_B) # B->A 변환 확인
            
            # 이미지 저장
            save_dir = "test_results"
            os.makedirs(save_dir, exist_ok=True)  # 존재하지 않으면 directory 생성
            save_image(fake_A, os.path.join(save_dir, f"fake_A_{i+1}.png"), normalize=True)
            save_image(fake_B, os.path.join(save_dir, f"fake_B_{i+1}.png"), normalize=True)
    
    print("Test complete.")

def load_models(G_AB, G_BA, checkpoint_dir):
    G_AB.load_state_dict(torch.load(os.path.join(checkpoint_dir, "generator_G_AB_190.pth")))
    G_BA.load_state_dict(torch.load(os.path.join(checkpoint_dir, "generator_G_BA_190.pth")))

if __name__ == "__main__":
    
    test_data_loader = get_loader(root_path=r"D:\CAU\eclass\컴퓨터비전\\archive (2)",
                                  phase='test',
                                  image_size=256,
                                  batch_size=8,
                                  num_workers=4)
    # 모델 체크포인트 디렉토리 경로
    checkpoint_dir = "D:\CAU\eclass\컴퓨터비전\saved_models_sum_win"
    
    
    test_data_loader = get_loader(root_path=r"D:\CAU\eclass\컴퓨터비전\\archive (4)",
                                  phase='test',
                                  image_size=256,
                                  batch_size=8,
                                  num_workers=4)
    checkpoint_dir = "D:\CAU\eclass\컴퓨터비전\saved_models"
    
    
    
    
    # 모델 초기화
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    
    # 저장된 모델 불러오기
    load_models(G_AB, G_BA, checkpoint_dir)
    
    # 테스트 실행
    test(G_AB, G_BA, test_data_loader, device)
