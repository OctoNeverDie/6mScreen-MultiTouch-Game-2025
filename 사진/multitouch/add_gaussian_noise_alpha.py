import cv2
import numpy as np

def add_gaussian_noise(image, mean=100, var=0):
    """
    RGB 채널에만 Gaussian 노이즈를 추가하는 함수
    Parameters:
    - image: RGB 이미지 (numpy 배열)
    - mean: 노이즈의 평균
    - var: 노이즈의 분산
    Returns:
    - 노이즈가 추가된 RGB 이미지 (numpy 배열)
    """
    # RGB 채널에 대해 노이즈 생성
    noise = np.random.normal(mean, var**0.5, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def process_image_with_alpha(image_path, output_path, mean=0, var=25):
    """
    알파 채널을 유지하며 RGB 채널에만 노이즈를 추가하는 함수
    Parameters:
    - image_path: 입력 PNG 이미지 경로
    - output_path: 출력 PNG 이미지 경로
    - mean: Gaussian 노이즈의 평균
    - var: Gaussian 노이즈의 분산
    """
    # 알파 채널 포함하여 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("이미지를 로드할 수 없습니다.")
        return
    
    # 이미지가 4채널 (RGBA)인지 확인
    if image.shape[2] != 4:
        print("이미지에 알파 채널이 없습니다.")
        return
    
    # RGB와 알파 채널 분리
    rgb = image[:, :, :3]  # RGB 채널
    alpha = image[:, :, 3]  # 알파 채널
    
    # RGB 채널에 노이즈 추가
    noisy_rgb = add_gaussian_noise(rgb, mean, var)
    
    # 노이즈가 적용된 RGB와 알파 채널 다시 결합
    result_image = cv2.merge((noisy_rgb, alpha))
    
    # 결과 이미지 저장
    cv2.imwrite(output_path, result_image)
    print(f"결과 이미지가 저장되었습니다: {output_path}")
    
    # 결과 이미지 표시 (옵션)
    cv2.imshow("Original Image", image)
    cv2.imshow("Noisy Image with Alpha", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    input_image_path = "C:/Users/JSY/Desktop/multitouch/starMap.png"  # 입력 이미지 경로
    output_image_path = "C:/Users/JSY/Desktop/multitouch/stars_noisy_alpha.png"  # 출력 이미지 경로
    process_image_with_alpha(input_image_path, output_image_path, mean=-225, var=800)
    
    