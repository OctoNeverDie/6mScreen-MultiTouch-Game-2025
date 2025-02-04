import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
size = 64
name = 't3'
image = cv2.imread(f"C:/Users/JSY/Desktop/img/{name}.png")  # 이미지 파일 경로

if not os.path.exists(image):
    print(f"이미지 파일을 찾을 수 없습니다: {image}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 로드하므로 RGB로 변환

# 이미지 크기 확인
height, width, channels = image.shape
print(f"Original Image Size: {width}x{height}")

# 1. 이미지를 64x64로 다운샘플링
downsampled = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

# 2. 다운샘플링된 이미지를 원본 크기로 업스케일링
mosaic_image = cv2.resize(downsampled, (width, height), interpolation=cv2.INTER_NEAREST)

# 3. 가우시안 노이즈 추가 (옵션)
#noise = np.random.normal(0, 25, mosaic_image.shape).astype(np.uint8)  # 노이즈 강도 조절 (25는 표준편차)
#mosaic_image = cv2.add(mosaic_image, noise)

# 4. 결과 출력
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
print("Mosaic Image with Gaussian Noise")
plt.imshow(mosaic_image)

plt.show()
outpath = f'{name}_{size}_m.jpg'

# 5. 결과 저장 (옵션)
cv2.imwrite(outpath, cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR))