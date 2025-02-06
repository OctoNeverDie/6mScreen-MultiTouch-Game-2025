import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(imageName, scale):
     # 첫 번째 이미지: 멀리서 보일 타겟 (노이즈 파라미터: 평균 0, 분산 25)
    input = f"{imgName}.png"
    output = f"{imgName}_{scale}_output.png"
    image = cv2.imread(input, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {input}")
        return
    
    height, width = 1500, 6000
    #base_color = 128  # 배경색 평균
    #background = np.ones((height, width)) * base_color#회색 이미지

    # 0평균 노이즈 생성 (예: 표준편차 20)
    noise = np.random.normal(loc=128, scale=scale, size=(height, width, 1))  # (1500, 6000, 1)로 확장

    # 노이즈를 배경에 더함
    result = image.astype(np.float32) + noise
    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(output, result)
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    imgName = "transparent"
    scale = 800
    add_gaussian_noise(imgName, scale)
    