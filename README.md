# ComicFlow-Engine
利用 OCR 与漫画分镜提取将 PDF 漫画转短视频

---

## 项目效果展示
### 视频合成效果
https://v.douyin.com/pg7BBFOVBUA/

### 漫画分镜分割效果
打开 src/pdf2imgs/reports/index.html

## 环境依赖

- Python 3.8+
- 依赖库（部分）：`paddleocr`、`pillow`、`moviepy`、`edge-tts`、`tqdm`、`opencv-python`、`PyMuPDF`、`tkinter` 等

安装依赖（建议使用虚拟环境）：

```bash
pip install paddleocr pillow moviepy edge-tts tqdm opencv-python PyMuPDF
```


---

## 使用流程

### 1. PDF转图片

首先使用 `src/pdf2imgs/pdf2imgs.py` 将 PDF 文件每页转为图片。

```bash
cd src/pdf2imgs
python pdf2imgs.py
```

- 运行后会弹出窗口，选择 PDF 文件和输出文件夹。
- 可选择要删除的页码（如不需要可留空）。
- 转换后每页图片会以 `page_1.png`、`page_2.png` ... 命名，保存在你选择的输出文件夹。

> **请将这些图片复制到 `src/pdf2imgs/imgs/` 目录下，供后续分镜提取使用。**


### 2. 分镜提取与OCR

在 `src/pdf2imgs/` 目录下，运行分镜提取脚本（如 `DeepPanelTest.py`），自动将每页图片分割为分镜图片，保存在 `model_pre/` 目录。

```bash
python DeepPanelTest.py
```

### 3. 生成视频脚本

进入 `src/imgs2script/`，运行 `gen_video_script.py`，自动对所有分镜图片进行 OCR，并生成视频脚本 `video_script.json`。

```bash
cd ../imgs2script
python gen_video_script.py
```

### 4. 合成短视频

进入 `src/script2video/`，运行 `gen_video_from_script.py`，自动根据 `video_script.json` 合成横屏短视频 `output.mp4`。

```bash
cd ../script2video
python gen_video_from_script.py
```

---

## 目录结构说明

- `src/pdf2imgs/`：PDF转图片、分镜提取与页面图片处理
- `src/imgs2script/`：OCR 识别与视频脚本生成
- `src/script2video/`：视频合成（配音、插画、主题、输出 mp4）

---

## 结果示例

- `src/pdf2imgs/model_pre/`：分镜图片
- `src/imgs2script/video_script.json`：视频脚本
- `src/script2video/output.mp4`：最终生成的视频

---

## 注意事项

- 需提前准备好每页图片（用 `pdf2imgs.py` 转换）
- 需提前准备好分镜图片（用 `DeepPanelTest.py` 自动生成）
- 需联网以使用 edge-tts 语音合成
- 需本地有 `msyh.ttc` 字体文件
- 视频为横屏 1920x1080，插画居中，左上角显示主题

---

如需自定义主题、配音内容、分镜样式等，请修改 `src/script2video/gen_video_from_script.py` 中相关参数。

如需更详细的参数说明或遇到问题，欢迎提 issue 或联系作者。
