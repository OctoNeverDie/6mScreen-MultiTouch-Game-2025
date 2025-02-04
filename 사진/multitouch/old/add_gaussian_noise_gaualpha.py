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
    low_freq = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    high_freq = cv2.subtract(image, low_freq)
    return low_freq, high_freq

def process_image_with_alpha(image_path, output_dir, blur_ksize=15, mean_low=0, var_low=50, mean_high=0, var_high=400):
    """
    고주파와 저주파에 각각 다른 노이즈를 추가하되, 투명 영역은 유지하는 함수
    Parameters:
    - image_path: 입력 이미지 경로
    - output_dir: 출력 이미지 디렉토리
    - blur_ksize: Gaussian 블러 커널 크기
    - mean_low, var_low: 저주파에 추가할 노이즈의 평균과 분산
    - mean_high, var_high: 고주파에 추가할 노이즈의 평균과 분산
    """
    # 이미지 로드 (알파 채널 포함)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("이미지를 로드할 수 없습니다.")
        return

    # RGBA인지 확인
    if image.shape[2] != 4:
        print("이미지에 알파 채널이 없습니다.")
        return

    # RGB와 알파 채널 분리
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]

    # 고주파와 저주파로 분리
    low_freq, high_freq = split_high_low_frequency(rgb, blur_ksize)

    # 저주파에 노이즈 추가
    noisy_low_freq = add_gaussian_noise(low_freq, mean=mean_low, var=var_low)

    # 고주파에 노이즈 추가
    noisy_high_freq = add_gaussian_noise(high_freq, mean=mean_high, var=var_high)

    # 저주파와 고주파 합성
    combined_rgb = cv2.add(noisy_low_freq, noisy_high_freq)

    # 투명 영역 유지: 알파 채널이 0인 부분은 RGB를 0으로 설정
    combined_rgb[alpha == 0] = 0

    # RGB와 알파 채널 결합
    result_image = cv2.merge((combined_rgb, alpha))

    # 출력 파일 이름 생성
    output_filename = f"stars_bg_trans_noisy_alpha_{mean_low}_{var_low}_{mean_high}_{var_high}.png"
    output_path = f"{output_dir}/{output_filename}"

    # 결과 저장
    cv2.imwrite(output_path, result_image)
    print(f"결과 이미지가 저장되었습니다: {output_path}")

    # 결과 보기
    cv2.imshow("Original Image", image)
    cv2.imshow("Noisy Image with Transparency", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행 코드
if __name__ == "__main__":
    input_image_path = "C:/Users/JSY/Desktop/img/stars_bg_trans.png"  # 입력 이미지 경로
    output_directory = "C:/Users/JSY/Desktop/img"  # 출력 이미지 디렉토리
    process_image_with_alpha(
        input_image_path,
        output_directory,
        blur_ksize=15,  # 저주파 필터 강도
        mean_low=0, var_low=800,  # 저주파 노이즈
        mean_high=-225, var_high=100  # 고주파 노이즈
    )
