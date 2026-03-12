import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

import os

# 좌/우 이미지 불러오기
base_dir = Path(__file__).parent.resolve()
left_color = cv2.imread(str(base_dir / "images/left.png"))
right_color = cv2.imread(str(base_dir / "images/right.png"))

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")


# 카메라 파라미터
f = 700.0
B = 0.12

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
gray_left = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity_int = stereo.compute(gray_left, gray_right)
disparity = disparity_int.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
valid_mask = disparity > 0
depth_map = np.zeros_like(disparity)
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}
for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    roi_valid = valid_mask[y:y+h, x:x+w]
    
    if np.any(roi_valid):
        avg_disp = np.mean(roi_disp[roi_valid])
        avg_depth = np.mean(roi_depth[roi_valid])
    else:
        avg_disp = 0
        avg_depth = 0
    
    results[name] = {"avg_disp": avg_disp, "avg_depth": avg_depth}

# -----------------------------
# 4. 결과 출력
# -----------------------------
print(f"{'ROI':<10} | {'Avg Disparity':<15} | {'Avg Depth':<10}")
print("-" * 40)
for name, data in results.items():
    print(f"{name:<10} | {data['avg_disp']:<15.4f} | {data['avg_depth']:<10.4f}")

# depth 가 가장 작은 것이 가장 가까움 (depth 값이 실제 거리)
closest_roi = min(results.items(), key=lambda item: item[1]['avg_depth'])
farthest_roi = max(results.items(), key=lambda item: item[1]['avg_depth'])

print(f"\n가장 가까운 객체: {closest_roi[0]} (Avg Depth: {closest_roi[1]['avg_depth']:.4f})")
print(f"가장 먼 객체: {farthest_roi[0]} (Avg Depth: {farthest_roi[1]['avg_depth']:.4f})")


# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)

# -----------------------------
# 9. 출력
# -----------------------------
cv2.imshow("Original Left (with ROIs)", left_vis)
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Depth Map", depth_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
