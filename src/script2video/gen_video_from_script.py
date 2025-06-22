import json
import os
import asyncio
import edge_tts
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

VIDEO_SCRIPT_PATH = "../imgs2script/video_script.json"
OUTPUT_VIDEO = "output.mp4"
TEMP_AUDIO_DIR = "temp_audio"
TEMP_FRAME_DIR = "temp_frames"
VOICE = "zh-CN-YunxiNeural"
FPS = 24
THEME = "偷星九月天第一卷"  # 可根据需要修改
FONT_PATH = "msyh.ttc"  # 确保本地有此字体

# edge-tts异步配音
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

async def synthesize_audio(text, audio_path, voice=VOICE):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(audio_path)

def create_frame(img_path, theme, frame_path):
    # 横屏 1920x1080
    bg = Image.new("RGBA", (1920, 1080), (255, 255, 255, 255))
    fg = Image.open(img_path).convert('RGBA').resize((800, 800))
    # 居中插画
    fg_x = (1920 - 800) // 2
    fg_y = (1080 - 800) // 2
    bg.paste(fg, (fg_x, fg_y), fg)
    draw = ImageDraw.Draw(bg)
    theme_font = ImageFont.truetype(FONT_PATH, 48)
    # 主题显示在左上角
    draw.text((50, 30), f"本期主题：{theme}", fill=(0, 0, 0), font=theme_font)
    bg.save(frame_path)

def main():
    ensure_dir(TEMP_AUDIO_DIR)
    ensure_dir(TEMP_FRAME_DIR)
    # 读取脚本
    with open(VIDEO_SCRIPT_PATH, "r", encoding="utf-8") as f:
        video_script = json.load(f)
    scenes = video_script["分镜结构"]["分镜列表"]
    cover_img = video_script["分镜结构"]["封面提示词"]["封面图片"]
    # 合成封面音频
    cover_audio_path = os.path.join(TEMP_AUDIO_DIR, "cover.mp3")
    cover_text = f"本期要讲的主题是{THEME}"
    async def synthesize_cover():
        await synthesize_audio(cover_text, cover_audio_path)
    asyncio.run(synthesize_cover())
    # 生成封面帧
    cover_frame_path = os.path.join(TEMP_FRAME_DIR, "cover.png")
    create_frame(cover_img, THEME, cover_frame_path)
    # 合成封面clip
    cover_audio_clip = AudioFileClip(cover_audio_path)
    cover_duration = cover_audio_clip.duration
    cover_clip = ImageClip(cover_frame_path).set_duration(cover_duration).set_audio(cover_audio_clip)
    clips = [cover_clip]
    # 生成所有配音任务
    audio_tasks = []
    for scene in scenes:
        subtitle = scene["字幕"]["中文"].strip()
        if subtitle:
            audio_path = os.path.join(TEMP_AUDIO_DIR, f"{scene['分镜编号']}.mp3")
            audio_tasks.append((subtitle, audio_path))
    # 执行配音，带进度条
    async def run_tts_tasks():
        tasks = []
        for text, path in tqdm(audio_tasks, desc="配音生成", unit="段"):
            tasks.append(synthesize_audio(text, path))
        await asyncio.gather(*tasks)
    if audio_tasks:
        asyncio.run(run_tts_tasks())
    # 合成分镜帧和clip
    for scene in tqdm(scenes, desc="视频片段合成", unit="段"):
        img_path = scene["分镜图片"]
        frame_path = os.path.join(TEMP_FRAME_DIR, f"scene_{scene['分镜编号']}.png")
        create_frame(img_path, THEME, frame_path)
        subtitle = scene["字幕"].get("中文", "").strip()
        if subtitle:
            audio_path = os.path.join(TEMP_AUDIO_DIR, f"{scene['分镜编号']}.mp3")
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            img_clip = ImageClip(frame_path).set_duration(duration).set_audio(audio_clip)
        else:
            img_clip = ImageClip(frame_path).set_duration(2)
        clips.append(img_clip)
    # 合并所有片段
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(OUTPUT_VIDEO, fps=FPS, codec="libx264", audio_codec="aac")
    # 清理临时音频和帧
    for _, audio_path in audio_tasks:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    if os.path.exists(cover_audio_path):
        os.remove(cover_audio_path)
    if os.path.exists(TEMP_AUDIO_DIR) and not os.listdir(TEMP_AUDIO_DIR):
        os.rmdir(TEMP_AUDIO_DIR)
    for f in os.listdir(TEMP_FRAME_DIR):
        os.remove(os.path.join(TEMP_FRAME_DIR, f))
    if os.path.exists(TEMP_FRAME_DIR) and not os.listdir(TEMP_FRAME_DIR):
        os.rmdir(TEMP_FRAME_DIR)
    print(f"视频已生成: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()