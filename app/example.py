"""内置“相机陷阱连拍”示例会话数据。

6 帧 384x288 画面（均不超过 512x512），模拟相机陷阱的一次连拍：

- 帧 0、1、2：同一只动物（橙棕色鹿形剪影）保持同一位置与姿态，
  仅有传感器噪声与轻微 JPEG 重编码差异 —— 期望判为重复，
  建议保留最早帧（帧 0），省略帧 1、2。
- 帧 3：动物走到画面另一侧，姿态由“站立”变为“低头觅食” ——
  动物区域可辨且发生变化，期望保留该帧。
- 帧 4：动物继续移动到画面中部，朝向改变（另一种姿态） ——
  期望保留该帧。
- 帧 5：与帧 4 同位置同姿态，但人为加入明显 JPEG 块效应 + 强噪声 + 模糊，
  使动物区域的相关性被压缩伪影拉低、可辨证据不足 ——
  证据不足，标为“待核”，不自动省略。

所有帧共享同一草地背景（固定相机陷阱视角）。
检测框每帧一个，框住动物；帧 5 的框与帧 4 完全一致。
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageDraw, ImageFilter

WIDTH, HEIGHT = 384, 288
N_FRAMES = 6

# 每帧动物中心（cx, cy）与姿态：
#   stand  站立、head_down 低头、turn 侧身转向
ANIMAL_POSES = [
    ((120, 150), "stand"),
    ((120, 150), "stand"),     # 与帧 0 同位置同姿态（重复）
    ((120, 150), "stand"),     # 与帧 0 同位置同姿态（重复）
    ((300, 170), "head_down"),  # 位置 + 姿态变化（与其他帧检测框不相交）
    ((205, 225), "turn"),       # 位置 + 朝向变化（与其他帧检测框不相交）
    ((205, 225), "turn"),       # 同位置同姿态，但压缩伪影严重 → 证据不足
]

# 检测框 [x, y, w, h]
# 帧 0~2 同位置；帧 3、帧 4 的框与其他帧互不相交（位置变化强证据）；
# 帧 4、5 同框（同位置同姿态，仅压缩质量差异 → 歧义待核）。
DETECT_RECTS = [
    [86, 104, 70, 88],
    [86, 104, 70, 88],
    [86, 104, 70, 88],
    [258, 126, 80, 88],
    [162, 180, 86, 86],
    [162, 180, 86, 86],
]

FUR = (176, 104, 52)
FUR_DARK = (128, 72, 34)


def _grass_background(rng: random.Random) -> Image.Image:
    """生成固定视角的草地背景（所有连拍帧共享，仅每帧加独立传感器噪声）。"""
    base = Image.new("RGB", (WIDTH, HEIGHT), (58, 88, 48))
    draw = ImageDraw.Draw(base)
    # 远处树线
    draw.rectangle([0, 0, WIDTH, 52], fill=(40, 62, 44))
    for x in range(0, WIDTH, 26):
        h = rng.randrange(34, 58)
        draw.ellipse([x - 16, h - 26, x + 16, h + 8], fill=(46, 72, 50))
    # 草地纹理（确定性布局，跨帧相同）
    for _ in range(900):
        x = rng.randrange(WIDTH)
        y = rng.randrange(52, HEIGHT)
        g = rng.randrange(70, 116)
        draw.point((x, y), fill=(g // 3, g, g // 4))
    # 几块石头 / 土斑，提供丰富的背景可辨证据
    for cx, cy, r in ((60, 232, 14), (330, 96, 11), (240, 60, 9), (150, 250, 8)):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(104, 102, 88))
        draw.ellipse([cx - r + 3, cy - r + 3, cx + r - 3, cy + r - 3], fill=(122, 118, 100))
    return base


def _draw_deer(draw: ImageDraw.ImageDraw, cx: int, cy: int, pose: str) -> None:
    """画一个简单的鹿形剪影；不同姿态的轮廓不同，保证姿态变化可被检出。"""
    # 身体
    draw.ellipse([cx - 30, cy - 16, cx + 30, cy + 18], fill=FUR, outline=FUR_DARK)
    # 四条腿
    for lx in (-20, -8, 10, 22):
        draw.rectangle([cx + lx - 3, cy + 14, cx + lx + 3, cy + 40], fill=FUR_DARK)
    # 尾巴
    draw.polygon([(cx - 30, cy - 6), (cx - 38, cy - 2), (cx - 30, cy + 4)], fill=FUR_DARK)

    if pose == "stand":
        # 头颈朝右上
        draw.rectangle([cx + 18, cy - 34, cx + 27, cy - 6], fill=FUR, outline=FUR_DARK)
        draw.ellipse([cx + 15, cy - 44, cx + 33, cy - 30], fill=FUR, outline=FUR_DARK)
        # 耳朵
        draw.polygon([(cx + 18, cy - 42), (cx + 13, cy - 52), (cx + 23, cy - 44)], fill=FUR_DARK)
        draw.polygon([(cx + 29, cy - 42), (cx + 34, cy - 52), (cx + 25, cy - 44)], fill=FUR_DARK)
    elif pose == "head_down":
        # 颈部前伸、头朝下（觅食），整体轮廓明显不同
        draw.polygon(
            [(cx + 20, cy - 10), (cx + 42, cy + 2), (cx + 40, cy + 14), (cx + 18, cy + 6)],
            fill=FUR, outline=FUR_DARK,
        )
        draw.ellipse([cx + 34, cy + 6, cx + 52, cy + 24], fill=FUR, outline=FUR_DARK)
        # 一只前腿抬起
        draw.rectangle([cx + 10 - 3, cy + 14, cx + 10 + 3, cy + 30], fill=FUR_DARK)
    else:  # turn：侧身回头
        # 颈部弯曲、头转向身体左侧
        draw.rectangle([cx - 6, cy - 32, cx + 4, cy - 4], fill=FUR, outline=FUR_DARK)
        draw.ellipse([cx - 18, cy - 40, cx + 2, cy - 24], fill=FUR, outline=FUR_DARK)
        draw.polygon([(cx - 14, cy - 38), (cx - 20, cy - 48), (cx - 8, cy - 40)], fill=FUR_DARK)
        draw.polygon([(cx - 4, cy - 38), (cx + 2, cy - 48), (cx + 4, cy - 38)], fill=FUR_DARK)


def _add_sensor_noise(img: Image.Image, rng: random.Random, sigma: int = 4) -> Image.Image:
    """加入轻微传感器噪声（每帧独立），模拟连拍帧之间的微小差异。"""
    px = img.load()
    for y in range(0, HEIGHT, 1):
        for x in range(0, WIDTH, 1):
            r, g, b = px[x, y]
            n = rng.gauss(0, sigma)
            px[x, y] = (
                max(0, min(255, int(r + n))),
                max(0, min(255, int(g + n))),
                max(0, min(255, int(b + n))),
            )
    return img


def build_example_detections():
    """返回示例检测框列表（dict）。每帧一个框。"""
    return [
        {"id": f"box{f}", "frame": f, "rect": list(DETECT_RECTS[f])}
        for f in range(N_FRAMES)
    ]


def build_example_frames():
    """返回 [(content_type, bytes), ...] 共 6 帧。

    帧 0~4 为高质量 JPEG（质量 92）；帧 5 先加较强噪声与模糊再以极低质量
    （质量 18）重编码，制造“证据不足”的歧义场景。
    """
    bg_rng = random.Random(20240617)
    background = _grass_background(bg_rng)
    frames = []
    for f in range(N_FRAMES):
        rng = random.Random(9000 + f)
        img = background.copy()
        draw = ImageDraw.Draw(img)
        (cx, cy), pose = ANIMAL_POSES[f]
        _draw_deer(draw, cx, cy, pose)
        draw.text((6, 4), f"burst {f}", fill=(235, 235, 220))
        img = _add_sensor_noise(img, rng, sigma=3)

        buf = io.BytesIO()
        if f == N_FRAMES - 1:
            # 严重压缩 + 模糊：动物区域细节被伪影淹没，相关性明显下降
            img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
            img = _add_sensor_noise(img, random.Random(99), sigma=16)
            img.save(buf, format="JPEG", quality=18, subsampling=2)
            ctype = "image/jpeg"
        else:
            img.save(buf, format="JPEG", quality=92, subsampling=0)
            ctype = "image/jpeg"
        frames.append((ctype, buf.getvalue()))
    return frames


EXAMPLE_DESCRIPTION = (
    "连拍示例：帧 0/1/2 为同位置同姿态的重复画面（仅噪声与 JPEG 差异），"
    "建议保留帧 0、省略帧 1/2；帧 3、帧 4 动物位置或姿态发生变化，应保留；"
    "帧 5 与帧 4 同位置同姿态但压缩伪影严重、动物区域证据不足，标为待核，不自动省略。"
)
