import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(imageName, loc, scale):
    # 파일 경로 설정
    input_path = f"{imageName}.png"
    output_path = f"{imageName}_{loc}_{scale}_rgb_noise.png"

    # 이미지 로드 (알파 채널 포함 가능)
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {input_path}")
        return

    # 이미지 크기 확인
    height, width = image.shape[:2]

    if image.shape[2] == 4:  # RGBA 이미지인 경우
        print("RGBA 이미지 감지됨, RGB 채널에만 노이즈 추가")

        # RGB와 알파 채널 분리
        rgb, alpha = image[:, :, :3], image[:, :, 3]

        # RGB 채널에만 노이즈 추가 (평균 128, 표준편차 scale)
        noise = np.random.normal(loc, scale, size=(height, width, 3))  # (1500, 6000, 3)
        result_rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 알파 채널은 그대로 유지하면서 다시 합치기
        result = np.dstack((result_rgb, alpha))

    else:  # RGB 이미지인 경우
        print("RGB 이미지 감지됨, 모든 채널에 노이즈 추가")

        # RGB 채널에만 노이즈 추가 (평균 0, 표준편차 scale)
        noise = np.random.normal(loc=0, scale=scale, size=(height, width, 3))
        result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 저장 및 출력
    cv2.imwrite(output_path, result)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA) if image.shape[2] == 4 else result)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    imgName = "128marker"
    loc = 0  # 노이즈 평균값 조절 가능
    scale = 10  # 노이즈 강도 조절 가능
    add_gaussian_noise(imgName, loc, scale)
