import cv2
import numpy as np

def adjust_brightness_alpha(image, alpha_factor=0.5, brightness_threshold=200):
    """
    밝기와 알파 채널을 조정하여 별이 가까운 거리에서 보이지 않게 처리
    Parameters:
    - image: RGBA 이미지 (4채널)
    - alpha_factor: 밝기 조정 비율 (0.0~1.0)
    - brightness_threshold: 밝기 임계값 (255 중)
    Returns:
    - 수정된 이미지 (RGBA)
    """
    # RGB와 알파 채널 분리
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]

    # 밝기 계산 (RGB 평균값 사용)
    brightness = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # 밝기에 따라 알파 값 조정
    adjusted_alpha = np.where(
        brightness > brightness_threshold,  # 밝기가 특정 임계값보다 크면
        alpha * alpha_factor,               # 투명도를 낮춤
        alpha                               # 그대로 유지
    ).astype(np.uint8)

    # 새로운 이미지 결합
    result_image = cv2.merge((rgb, adjusted_alpha))
    return result_image

def process_image(image_path, output_directory, alpha_factor=0.5, brightness_threshold=200):
    """
    이미지 처리 함수: 밝기와 투명도를 조정
    Parameters:
    - image_path: 입력 이미지 경로
    - output_path: 출력 이미지 경로
    - alpha_factor: 밝기 조정 비율
    - brightness_threshold: 밝기 임계값
    """
    # 이미지 로드 (알파 채널 포함)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("이미지를 로드할 수 없습니다.")
        return

    # 알파와 밝기 조정
    processed_image = adjust_brightness_alpha(image, alpha_factor, brightness_threshold)

    # 결과 저장
    output_filename = f"teduri_processed_{alpha_factor}_{brightness_threshold}.png"
    output_path = f"{output_directory}/{output_filename}"
    cv2.imwrite(output_path, processed_image)

    # 결과 보기
    cv2.imshow("Original Image", image)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행 코드
if __name__ == "__main__":
    input_image_path = "C:/Users/JSY/Desktop/multitouch/teduri.png"  # 입력 이미지 경로
    output_directory =  "C:/Users/JSY/Desktop/multitouch"  # 출력 이미지 경로
    process_image(
        input_image_path,
        output_directory,
        alpha_factor=0.2,  # 알파 조정 비율 (멀리서 보이게)
        brightness_threshold=200  # 밝기 임계값
    )
