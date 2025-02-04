import cv2
import numpy as np

def generate_low_freq_noise(height, width, mean, variance, downscale_factor=0.1, blur_sigma=5):
    """
    저주파 노이즈 생성 함수:
    - 낮은 해상도의 노이즈를 생성 후 업샘플링과 블러를 적용하여 부드러운 저주파 노이즈를 생성합니다.
    """
    # 낮은 해상도로 생성
    low_h = max(1, int(height * downscale_factor))
    low_w = max(1, int(width * downscale_factor))
    noise_low = np.random.normal(mean, np.sqrt(variance), (low_h, low_w)).astype(np.float32)
    
    # 원래 해상도로 업샘플링
    noise_low_up = cv2.resize(noise_low, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # 추가적인 Gaussian 블러 적용으로 더 부드러운 효과
    #noise_low_up = cv2.GaussianBlur(noise_low_up, (0, 0), sigmaX=blur_sigma)
    
    return noise_low_up

def generate_high_freq_noise(height, width, mean, variance):
    """
    고주파 노이즈 생성 함수:
    - 전체 해상도에서 바로 생성하여 세밀한 고주파 노이즈를 만듭니다.
    """
    noise_high = np.random.normal(mean, np.sqrt(variance), (height, width)).astype(np.float32)
    return noise_high

def normalize_and_save(filename, noise):
    """
    이미지를 0~255 범위로 정규화하여 저장합니다.
    """
    norm = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    name2 = f"{filename}_norm.png"
    cv2.imwrite(name2, norm)

# 이미지 크기 설정
height, width = 6000, 1500

# 사용자가 조절할 수 있는 파라미터 (평균, 분산)
mean_low = 0.0
variance_low = 0.05   # 저주파 노이즈 분산

mean_high = 0.0
variance_high = 0.1   # 고주파 노이즈 분산

# 노이즈 생성
low_freq_noise = generate_low_freq_noise(height, width, mean_low, variance_low, downscale_factor=0.1, blur_sigma=5)
high_freq_noise = generate_high_freq_noise(height, width, mean_high, variance_high)

# 각각의 노이즈 이미지를 저장 ("1.png" - 저주파, "2.png" - 고주파)
lowImgName = "bw_teduri_star"
highImgName = "c_bg"
normalize_and_save(f"{lowImgName}.png", low_freq_noise)
normalize_and_save(f"{highImgName}.png", high_freq_noise)

# 두 노이즈를 단순 합성 (필요에 따라 가중치를 조절할 수 있음)
combined_noise = low_freq_noise + high_freq_noise

# 합성 이미지 저장
name = f"{lowImgName}_{highImgName}_{mean_low}_{variance_low}_{mean_high}_{variance_high}"
normalize_and_save(f"{name}.png", combined_noise)
