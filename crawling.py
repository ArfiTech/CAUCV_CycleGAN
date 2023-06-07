import requests
from bs4 import BeautifulSoup
import urllib
import os

# 검색어와 이미지 저장 경로 설정
search_query = "중앙대학교 건물"
save_directory = "D:\CAU\eclass\컴퓨터비전\\archive (4)\\testA"
max_images = 300  # 수집할 이미지 개수 설정

# Google 이미지 검색 URL 생성
search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"

# 검색 결과 페이지 요청
response = requests.get(search_url)
response.raise_for_status()

# HTML 파싱
soup = BeautifulSoup(response.text, "html.parser")

# 이미지 태그 추출
image_tags = soup.find_all("img")

# 이미지 다운로드
image_count = 0
for i, image_tag in enumerate(image_tags):
    if image_count >= max_images:
        break
    
    image_url = image_tag.get("src")
    
    # 이미지 다운로드 시도
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        # 이미지 저장 경로 설정
        image_path = os.path.join(save_directory, f"image{i}.png")
        print(image_path)
        
        # 이미지 저장
        with open(image_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        
        print(f"Image {i+1} downloaded successfully.")
        image_count += 1
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to download Image {i+1}. Error: {str(e)}")
