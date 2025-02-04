import cv2
import numpy as np

def process_image_with_alpha_and_mosaic(image_path, output_path, mean=0, var=25):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("이미지를 로드할 수 없습니다.")
        return
    
    if image.shape[2] != 4:
        print("이미지에 알파 채널이 없습니다.")
        return
    
    rgb = image[:, :, :3]  # RGB 채널
    alpha = image[:, :, 3]  # 알파 채널
    
    # 모자이크 효과 적용
    downsampled = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
    mosaic_rgb = cv2.resize(downsampled, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 노이즈 추가
    noise = np.random.normal(mean, var**0.5, mosaic_rgb.shape).astype(np.float32)
    noisy_rgb = mosaic_rgb.astype(np.float32) + noise
    noisy_rgb = np.clip(noisy_rgb, 0, 255).astype(np.uint8)
    
    # 결과 이미지 생성
    result_image = cv2.merge((noisy_rgb, alpha))
    cv2.imwrite(output_path, result_image)
    print(f"결과 이미지가 저장되었습니다: {output_path}")
    
    # 결과 이미지 표시
    cv2.imshow("Result Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_image_path = "C:/Users/JSY/Desktop/multitouch/starMap.png"
    output_image_path = "C:/Users/JSY/Desktop/multitouch/stars_mosaic_noisy_alpha.png"
    process_image_with_alpha_and_mosaic(input_image_path, output_image_path, mean=0, var=100)