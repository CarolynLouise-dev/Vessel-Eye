# feature_extract.py
import numpy as np
from skimage.measure import label, regionprops


def extract_features(binary_mask, en_green):
    label_img = label(binary_mask)
    regions = regionprops(label_img)

    vessel_pixels = en_green[binary_mask > 0]
    if len(vessel_pixels) < 10: return [1.0, 1.0, 0.0, 0.0], regions

    v_pix_mean = np.median(vessel_pixels)
    a_diams, v_diams, torts = [], [], []

    for reg in regions:
        if reg.area < 50: continue  # Lọc nhiễu nhẹ để tính toán chính xác

        avg_int = np.mean(en_green[reg.coords[:, 0], reg.coords[:, 1]])
        t_score = (reg.perimeter ** 2) / (4 * np.pi * reg.area)

        # Chỉ lấy các đoạn mạch có độ dài hợp lệ để tính trung bình
        if reg.area > 100:
            torts.append(t_score)

        if avg_int > v_pix_mean:
            a_diams.append(reg.axis_minor_length)
        else:
            v_diams.append(reg.axis_minor_length)

    avg_a = np.mean(a_diams) if a_diams else 1.0
    avg_v = np.mean(v_diams) if v_diams else 1.5
    av_ratio = avg_a / avg_v

    # Lọc giá trị Tortuosity để lấy giá trị trung bình thực tế hơn
    valid_torts = [t for t in torts if t > 1.02]
    avg_tort = np.mean(valid_torts) if valid_torts else 1.0
    std_tort = np.std(valid_torts) if valid_torts else 0.0
    density = np.sum(binary_mask > 0) / (binary_mask.shape[0] * binary_mask.shape[1])

    return [av_ratio, avg_tort, std_tort, density], regions