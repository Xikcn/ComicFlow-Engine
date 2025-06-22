# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
import numpy as np
import re
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

def iou(box1, box2):
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2
    inter_l = max(l1, l2)
    inter_t = max(t1, t2)
    inter_r = min(r1, r2)
    inter_b = min(b1, b2)
    if inter_l >= inter_r or inter_t >= inter_b:
        return 0
    inter_area = (inter_r - inter_l) * (inter_b - inter_t)
    area1 = (r1 - l1) * (b1 - t1)
    area2 = (r2 - l2) * (b2 - t2)
    return inter_area / (area1 + area2 - inter_area)

def cluster_bubbles(bubbles, iou_threshold=0.1, dist_threshold=30):
    clusters = []
    for bubble in bubbles:
        added = False
        for cluster in clusters:
            for b in cluster:
                if iou(b['bbox'], bubble['bbox']) > iou_threshold or \
                   (abs(b['center'][0] - bubble['center'][0]) < dist_threshold and abs(b['center'][1] - bubble['center'][1]) < dist_threshold):
                    cluster.append(bubble)
                    added = True
                    break
            if added:
                break
        if not added:
            clusters.append([bubble])
    return clusters

def is_valid_text(text):
    # 只保留含有中文的内容
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def sort_bubbles_by_position(ocr_data, img_shape):
    rec_texts = ocr_data.get('rec_texts', [])
    rec_scores = ocr_data.get('rec_scores', [])
    rec_polys = ocr_data.get('rec_polys', [])
    bubbles = []
    for i, poly in enumerate(rec_polys):
        if i >= len(rec_scores) or rec_scores[i] < 0.8:
            continue
        if poly is None or len(poly) < 4:
            continue
        flat_poly = poly.flatten() if isinstance(poly, np.ndarray) else poly
        xs = [flat_poly[j] for j in range(0, len(flat_poly), 2)]
        ys = [flat_poly[j] for j in range(1, len(flat_poly), 2)]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        text = rec_texts[i] if i < len(rec_texts) else ""
        if not is_valid_text(text):
            continue
        bubbles.append({'text': text, 'center': (center_x, center_y), 'bbox': (left, top, right, bottom)})

    if not bubbles:
        return []

    img_w, img_h = img_shape[0], img_shape[1]
    x_threshold = img_w * 0.1  # 横向聚类阈值

    # 按X排序
    bubbles.sort(key=lambda b: b['center'][0])

    groups = []
    used = [False] * len(bubbles)

    for i, b in enumerate(bubbles):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        l1, t1, r1, b1 = b['bbox']
        # 横向聚类：只要X区间有重叠或距离小于阈值就归为一组
        for j in range(i + 1, len(bubbles)):
            if used[j]:
                continue
            l2, t2, r2, b2 = bubbles[j]['bbox']
            # 判断X区间是否重叠或距离小于阈值
            if (l2 <= r1 and r2 >= l1) or (l2 - r1 < x_threshold):
                group.append(j)
                used[j] = True
                r1 = max(r1, r2)
        # 组内按Y排序
        group_sorted = sorted(group, key=lambda idx: bubbles[idx]['center'][1])
        groups.append(''.join([bubbles[idx]['text'] for idx in group_sorted]))

    return groups

def ocr_and_order_text(img_path):
    from PIL import Image
    img = Image.open(img_path)
    img_w, img_h = img.size
    result = ocr.predict(input=img_path)
    if not result or not result[0]:
        print(f"未检测到文本: {img_path}")
        return ""
    ocr_data = result[0]
    ordered_texts = sort_bubbles_by_position(ocr_data, (img_w, img_h))
    return '\n'.join(ordered_texts)

# 示例用法
if __name__ == "__main__":
    img_path = r"D:\Python_workspace\ComicFlow-Engine\src\pdf2imgs\model_pre\015\panel_3.jpg"  # 替换为你的图片路径
    final_text = ocr_and_order_text(img_path)
    print(final_text)
