import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, var=25):
    """
    Gaussian 노이즈를 추가하는 함수
    Parameters:
    - image: 입력 이미지
    - mean: 노이즈 평균
    - var: 노이즈 분산
    Returns:
    - 노이즈가 추가된 이미지
    """
    noise = np.random.normal(mean, var**0.5, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def split_high_low_frequency(image, blur_ksize=15):
    """
    고주파와 저주파로 이미지를 분리하는 함수
    Parameters:
    - image: 입력 이미지
    - blur_ksize: Gaussian 블러 커널 크기 (저주파 필터링 강도)
    Returns:
    - low_freq: 저주파 이미지
    - high_freq: 고주파 이미지
    """
    # Gaussian 블러를 적용해 저주파 구성 요소 추출
    low_freq = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    # 고주파 구성 요소는 원본 이미지에서 저주파 이미지를 뺀 것
    high_freq = cv2.subtract(image, low_freq)
    return low_freq, high_freq

def process_image(image_path, output_path, blur_ksize=15, mean_low=0, var_low=50, mean_high=0, var_high=400):
    """
    고주파와 저주파에 각각 다른 노이즈를 추가하는 함수
    Parameters:
    - image_path: 입력 이미지 경로
    - output_path: 출력 이미지 경로
    - blur_ksize: Gaussian 블러 커널 크기
    - mean_low, var_low: 저주파에 추가할 노이즈의 평균과 분산
    - mean_high, var_high: 고주파에 추가할 노이즈의 평균과 분산
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 로드할 수 없습니다.")
        return

    # 고주파와 저주파로 분리
    low_freq, high_freq = split_high_low_frequency(image, blur_ksize)

    # 저주파에 노이즈 추가
    noisy_low_freq = add_gaussian_noise(low_freq, mean=mean_low, var=var_low)

    # 고주파에 노이즈 추가
    noisy_high_freq = add_gaussian_noise(high_freq, mean=mean_high, var=var_high)

    # 저주파와 고주파를 합성
    combined_image = cv2.add(noisy_low_freq, noisy_high_freq)

    # 결과 저장
    cv2.imwrite(output_path, combined_image)
    print(f"결과 이미지가 저장되었습니다: {output_path}")

    # 결과 보기
    cv2.imshow("Original Image", image)
    cv2.imshow("Noisy Image", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행 코드
if __name__ == "__main__":
    input_image_path = "C:/Users/JSY/Desktop/multitouch/businessMan.png"  # 입력 이미지 경로
    output_image_path = "C:/Users/JSY/Desktop/multitouch/businessMan_noisy_combined.png"  # 출력 이미지 경로
    process_image(
        input_image_path,
        output_image_path,
        blur_ksize=15,  # 저주파 필터 강도
        mean_low=0, var_low=0,  # 저주파 노이즈
        mean_high=-225, var_high=40000  # 고주파 노이즈
    )
