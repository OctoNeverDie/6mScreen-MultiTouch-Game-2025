import cv2
import numpy as np

def add_gaussian_noise(image, mean, var):
    """
    RGB 채널에만 Gaussian 노이즈를 추가하는 함수
    Parameters:
      - image: RGB 이미지 (numpy 배열)
      - mean: 노이즈의 평균
      - var: 노이즈의 분산 (표준편차는 var**0.5)
    Returns:
      - 노이즈가 추가된 RGB 이미지 (numpy 배열)
    """
    noise = np.random.normal(mean, var**0.5, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def process_image_with_alpha(image_path, output_path, mean=0, var=25):
    """
    알파 채널은 유지하면서 RGB 채널에만 노이즈를 추가하는 함수.
    Parameters:
      - image_path: 입력 PNG 이미지 경로 (알파 채널 포함)
      - output_path: 결과 PNG 이미지 저장 경로
      - mean: Gaussian 노이즈의 평균
      - var: Gaussian 노이즈의 분산
    """
    # 알파 채널 포함하여 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return
    # 4채널(RGBA) 이미지인지 확인
    #if image.shape[2] != 4:
    #    print(f"이미지에 알파 채널이 없습니다: {image_path}")
    #    return
    
    # RGB와 알파 채널 분리
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]
    
    # RGB 채널에 노이즈 추가
    noisy_rgb = add_gaussian_noise(rgb, mean, var)
    
    # 노이즈가 적용된 RGB와 원래의 알파 채널 결합
    result_image = cv2.merge((noisy_rgb, alpha))
    
    # 결과 이미지 저장
    cv2.imwrite(output_path, result_image)
    print(f"결과 이미지가 저장되었습니다: {output_path}")
    
    # 결과 이미지 표시 (옵션)
    cv2.imshow("Original Image", image)
    cv2.imshow("Noisy Image with Alpha", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result_image

def composite_images(far_image_path, near_image_path, output_path):
    """
    두 이미지를 알파 블렌딩하여 합성합니다.
    far_image: 멀리서 보일 타겟, 노이즈 효과가 덜하여 앞쪽에 보임 (즉, 위쪽에 위치)
    near_image: 가까이서 보일 타겟, 노이즈 효과가 강해 배경처럼 보임
    두 이미지 모두 4채널(RGBA)이어야 하며, 같은 크기라고 가정합니다.
    """
    # 두 이미지 모두 알파 채널 포함 읽기
    far_img = cv2.imread(far_image_path, cv2.IMREAD_UNCHANGED)
    near_img = cv2.imread(near_image_path, cv2.IMREAD_UNCHANGED)
    
    if far_img is None or near_img is None:
        print("합성할 이미지를 불러올 수 없습니다.")
        return
    
    # 크기가 다를 경우, far_img 크기에 맞게 near_img를 리사이즈 (필요 시)
    if far_img.shape != near_img.shape:
        near_img = cv2.resize(near_img, (far_img.shape[1], far_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # far_img의 알파 채널을 정규화 (0~1)
    alpha_far = far_img[:, :, 3].astype(np.float32) / 255.0
    alpha_far = cv2.merge([alpha_far, alpha_far, alpha_far])  # 3채널로 확장

    # 알파 블렌딩: far_img를 위에, near_img를 아래에 배치
    composite_rgb = far_img[:, :, :3].astype(np.float32) * alpha_far + \
                    near_img[:, :, :3].astype(np.float32) * (1 - alpha_far)
    
    composite_rgb = np.clip(composite_rgb, 0, 255).astype(np.uint8)
    
    # 결과 알파 채널은 far_img의 알파 채널 (또는 상황에 따라 조정)
    composite_alpha = far_img[:, :, 3]
    composite = cv2.merge((composite_rgb, composite_alpha))
    
    cv2.imwrite(output_path, composite)
    print(f"합성 이미지가 저장되었습니다: {output_path}")
    
    cv2.imshow("Composite Image", composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return composite

if __name__ == "__main__":
    lowImgName = "b_mung_star"
    highImgName = "c_bg"
     # 첫 번째 이미지: 멀리서 보일 타겟 (노이즈 파라미터: 평균 0, 분산 25)
    low_output = f"{lowImgName}_output.png"
    process_image_with_alpha(f"{lowImgName}.png", f"{lowImgName}_output.png", mean=0, var=800)
    
    # 두 번째 이미지: 가까이서 보일 타겟 (노이즈 파라미터: 평균 -225, 분산 800)
    high_output = f"{highImgName}_output.png"
    process_image_with_alpha(f"{highImgName}.png", f"{highImgName}_output.png", mean=-225, var=100)
    
    # 두 이미지를 합성 (멀리서 보일 타겟(far_target)이 위에 오도록)
    #composite_output = "composite_output.png"
    #composite_images(low_output, high_output, composite_output)
