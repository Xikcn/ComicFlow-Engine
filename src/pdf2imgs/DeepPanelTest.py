import os
from jinja2 import Environment, FileSystemLoader
import numpy as np
from utils import labeled_prediction_to_image, files_in_folder
from PIL import Image
import tensorflow as tf
import cv2

def preprocess_image(img_path, image_size):
    image = Image.open(img_path).convert('RGB')
    orig_size = image.size
    image = image.resize((image_size, image_size))
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)  # [1, H, W, 3]
    return image_np, orig_size

def load_true_mask(mask_path, image_size):
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((image_size, image_size))
    mask = np.array(mask, dtype=np.uint8)
    return mask

class PredictionResult:
    def __init__(self, page, pred, mask, panels=None):
        self.page = page
        self.pred = pred
        self.mask = mask
        self.panels = panels if panels is not None else []

def generate_output_template():
    output_images = files_in_folder("./output/")
    test_images = files_in_folder("./imgs")
    template_predictions = list()
    
    # 按数字顺序排序输入图片
    def extract_page_number(filename):
        # 处理page_9.png格式，提取数字9
        if filename.startswith('page_'):
            try:
                return int(filename.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                return float('inf')
        return float('inf')
    
    test_images_sorted = sorted(test_images, key=extract_page_number)
    
    for test_img in test_images_sorted:
        # 从输入文件名获取页面编号
        page_num = extract_page_number(test_img)
        if page_num == float('inf'):
            continue
            
        # 构造对应的输出文件名
        output_img = f"{page_num:03d}.jpg"
        
        # 检查输出文件是否存在
        if not os.path.exists(f"./output/{output_img}"):
            print(f"Warning: 输出文件不存在 {output_img}")
            continue
            
        # 收集分镜图片路径
        panel_dir = f"./model_pre/{page_num:03d}"
        panels = []
        if os.path.exists(panel_dir):
            for fname in sorted(os.listdir(panel_dir)):
                if fname.startswith("panel_") and fname.endswith(".jpg"):
                    panels.append(f"../model_pre/{page_num:03d}/{fname}")
        
        template_predictions.append(PredictionResult(test_img, output_img, None, panels))
    
    loader = FileSystemLoader("./templates")
    env = Environment(loader=loader)
    template = env.get_template('index.html')
    template_output = template.render(predictions=template_predictions)
    reports_path = "./reports"
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    text_file = open(f"{reports_path}/index.html", "w")
    text_file.write(template_output)
    text_file.close()

if __name__ == "__main__":
    print(" - Loading TFLite model")
    tflite_model_path = "./model/model.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    IMAGE_SIZE = input_details[0]['shape'][1]
    print(f" - Model input size: {IMAGE_SIZE}")

    test_images = files_in_folder("./imgs")
    testing_num_files = len(test_images)
    print(f" - Test data loaded for {testing_num_files} images")
    print(" - Prediction started")
    
    # 清空输出文件夹
    output_path = "./output/"
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    # 清空model_pre文件夹
    model_pre_dir = "model_pre"
    if os.path.exists(model_pre_dir):
        import shutil
        shutil.rmtree(model_pre_dir)
    os.makedirs(model_pre_dir)
    
    # 按页面编号排序输入图片
    def extract_page_number(filename):
        if filename.startswith('page_'):
            try:
                return int(filename.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                return float('inf')
        return float('inf')
    
    test_images_sorted = sorted(test_images, key=extract_page_number)
    
    for test_img in test_images_sorted:
        page_num = extract_page_number(test_img)
        if page_num == float('inf'):
            continue
            
        img_path = os.path.join("./imgs", test_img)
        image_np, orig_size = preprocess_image(img_path, IMAGE_SIZE)
        interpreter.set_tensor(input_details[0]['index'], image_np.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if output.ndim == 4:
            if output.shape[-1] == 3:
                pred_mask = np.argmax(output[0], axis=-1)
            elif output.shape[1] == 3:
                pred_mask = np.argmax(output[0].transpose(1,2,0), axis=-1)
            else:
                pred_mask = output[0, :, :, 0]
        elif output.ndim == 3:
            pred_mask = np.argmax(output[0], axis=-1)
        else:
            pred_mask = output[0]
        # 保存预测图片，resize回原图尺寸
        prediction_as_image = labeled_prediction_to_image(pred_mask)
        prediction_as_image = prediction_as_image.resize(orig_size, resample=Image.NEAREST)
        pred_save_path = f"{output_path}{page_num:03d}.jpg"
        prediction_as_image.save(pred_save_path)
        prediction_as_image.close()
        print(f"    - Image {test_img} (page {page_num}) saved.")
        # ========== extract_panels.py功能集成 ==========
        # 1. 读取mask图和原图
        mask_img = cv2.imread(pred_save_path)
        orig_img = cv2.imread(img_path)
        if mask_img is None or orig_img is None:
            print(f"[Warning] 读取图片失败: {pred_save_path} 或 {img_path}")
            continue
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        # 2. 提取灰色区域
        lower = 100
        upper = 240
        mask = cv2.inRange(gray, lower, upper)
        # 3. 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 4. 获取所有框并扩大尺寸
        boxes = []
        img_h, img_w = orig_img.shape[:2]
        expand = 10
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 1000:
                nx = max(x - expand, 0)
                ny = max(y - expand, 0)
                nw = min(x + w + expand, img_w) - nx
                nh = min(y + h + expand, img_h) - ny
                boxes.append((nx, ny, nw, nh))
        # 5. 只保留未被其他框覆盖面积超过70%的框
        def intersection_area(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            iw = max(xi2 - xi1, 0)
            ih = max(yi2 - yi1, 0)
            return iw * ih
        final_boxes = []
        for i_box, box in enumerate(boxes):
            area = box[2] * box[3]
            max_overlap = 0
            for j, other in enumerate(boxes):
                if i_box == j:
                    continue
                overlap = intersection_area(box, other)
                overlap_ratio = overlap / area
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
            if max_overlap <= 0.7:
                # 过滤掉宽或高小于原图宽/高10%的框
                if box[2] < img_w * 0.1 or box[3] < img_h * 0.1:
                    continue
                final_boxes.append(box)
        # 先按中心y排序，分行，再每行内按x排序
        def box_center(box):
            x, y, w, h = box
            return (y + h / 2, x + w / 2)
        sorted_boxes = sorted(final_boxes, key=lambda box: box_center(box)[0])
        lines = []
        line_threshold = 0.5  # 行高阈值比例，可调整
        for box in sorted_boxes:
            cy, _ = box_center(box)
            if not lines:
                lines.append([box])
            else:
                last_line = lines[-1]
                last_cy, _ = box_center(last_line[0])
                avg_h = np.mean([b[3] for b in last_line])
                if abs(cy - last_cy) > avg_h * line_threshold:
                    lines.append([box])
                else:
                    lines[-1].append(box)
        ordered_boxes = []
        for line in lines:
            line_sorted = sorted(line, key=lambda box: box_center(box)[1])
            ordered_boxes.extend(line_sorted)
        boxed = mask_img.copy()
        for idx_panel, (x, y, w, h) in enumerate(ordered_boxes):
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(boxed, str(idx_panel), (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output_img_path = os.path.join(output_path, f"{page_num:03d}.jpg")
        cv2.imwrite(output_img_path, boxed)
        base_name = f"{page_num:03d}"
        panel_dir = os.path.join(model_pre_dir, base_name)
        if not os.path.exists(panel_dir):
            os.makedirs(panel_dir)
        for idx_panel, (x, y, w, h) in enumerate(ordered_boxes):
            crop = orig_img[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(panel_dir, f"panel_{idx_panel}.jpg"), crop)
        # ========== extract_panels.py功能集成结束 ==========
    print(f" - Generating sample output page")
    generate_output_template()
    print(f" - Images saved. Time to check accuracy metrics per label:")

