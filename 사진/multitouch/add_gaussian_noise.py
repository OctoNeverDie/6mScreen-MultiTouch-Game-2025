# 필요한 라이브러리 불러오기
import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, var=10):
    """
    이미지에 Gaussian 노이즈를 추가하는 함수

    Parameters:
    - image: 원본 이미지 (numpy 배열)
    - mean: 노이즈의 평균
    - var: 노이즈의 분산

    Returns:
    - 노이즈가 추가된 이미지 (numpy 배열)
    """
    # 이미지의 데이터 타입을 float으로 변환하여 계산 정밀도 향상
    image = image.astype(np.float32)

    # 각 채널에 대해 노이즈 생성
    noise = np.zeros_like(image)
    for channel in range(3):  # B, G, R 채널
        # 평균과 표준편차 계산 (표준편차는 분산의 제곱근)
        sigma = var ** 0.5
        # Gaussian 노이즈 생성
        gaussian = np.random.normal(mean, sigma, image[:,:,channel].shape)
        noise[:,:,channel] = gaussian

    # 노이즈를 원본 이미지에 추가
    noisy_image = image + noise

    # 픽셀 값이 0~255 범위를 벗어나지 않도록 클리핑
    noisy_image = np.clip(noisy_image, 0, 255)

    # 다시 uint8 타입으로 변환
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image

def main():
    # 이미지 파일 경로 지정 (여기에 본인의 이미지 파일 경로를 입력하세요)
    image_path = "C:/Users/JSY/Desktop/multitouch/stars.png"  # 예: 'C:/ImageProcessing/input_image.jpg'

    # 이미지 읽기
    image = cv2.imread(image_path)

    # 이미지가 제대로 로드되었는지 확인
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    # Gaussian 노이즈 추가
    noisy_image = add_gaussian_noise(image, mean=-225, var=800)

    # 노이즈 추가된 이미지 저장
    cv2.imwrite('stars_noisy.jpg', noisy_image)

    # 결과 이미지 보여주기 (윈도우 창에서)
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
