import os
import json
from paddleocr import PaddleOCR
from PIL import Image
import traceback
from imgocr import ocr_and_order_text

MODEL_PRE_DIR = r"../pdf2imgs/model_pre"
OUTPUT_JSON = "video_script.json"

def main():
    panel_infos = []
    for folder in sorted(os.listdir(MODEL_PRE_DIR)):
        folder_path = os.path.join(MODEL_PRE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.startswith("panel_") and fname.endswith(".jpg"):
                img_path = os.path.join(folder_path, fname)
                print(f"处理图像: {img_path}")
                try:
                    ocr_text = ocr_and_order_text(img_path)
                    print(f"识别结果: {ocr_text}")
                    panel_infos.append({
                        "分镜编号": str(len(panel_infos) + 1),
                        "标题": img_path.replace("\\", "/"),
                        "字幕": {"中文": ocr_text},
                        "分镜图片": img_path.replace("\\", "/")
                    })
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {str(e)}")
                    traceback.print_exc()
                    panel_infos.append({
                        "分镜编号": str(len(panel_infos) + 1),
                        "标题": img_path.replace("\\", "/"),
                        "字幕": {"中文": "OCR处理失败"},
                        "分镜图片": img_path.replace("\\", "/")
                    })
    cover_img = panel_infos[0]["分镜图片"] if panel_infos else ""
    video_script = {
        "分镜结构": {
            "封面提示词": {
                "封面文案": "",
                "封面图片": cover_img
            },
            "分镜列表": panel_infos
        }
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(video_script, f, ensure_ascii=False, indent=2)
    print(f"已生成视频脚本: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()