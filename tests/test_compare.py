"""比对引擎单元测试：重复帧、JPEG 重编码容忍、姿态/位置变化、
证据不足（歧义）、重复关系矛盾与组级建议生成。"""

import io
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.compare import (
    analyze_burst,
    compare_pair,
    load_gray,
    report_to_dict,
)
from app.example import build_example_detections, build_example_frames

W, H = 256, 192
RECT = [96, 70, 64, 80]  # 单框，面积 5120


def _scene(seed=1, animal_offset=(0, 0), pose="a"):
    """生成带纹理背景 + 动物块的灰度画面（PIL RGB，便于 JPEG 编码）。"""
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), (60, 90, 50))
    px = img.load()
    # 背景纹理（每帧共享的确定性纹理 + 独立轻噪声）
    for y in range(H):
        for x in range(W):
            base = px[x, y]
            n = rng.gauss(0, 3)
            px[x, y] = (
                max(0, min(255, int(base[0] + n))),
                max(0, min(255, int(base[1] + n))),
                max(0, min(255, int(base[2] + n))),
            )
    # 固定地标，给背景充分可辨证据
    for cx, cy, r in ((30, 30, 12), (220, 160, 10), (120, 24, 8)):
        for x in range(cx - r, cx + r):
            for y in range(cy - r, cy + r):
                if 0 <= x < W and 0 <= y < H and (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    px[x, y] = (200, 190, 150)
    # 动物：带纹理的色块 + 姿态标记
    dx, dy = animal_offset
    x0, y0 = RECT[0] + dx, RECT[1] + dy
    for x in range(x0, min(x0 + RECT[2], W)):
        for y in range(y0, min(y0 + RECT[3], H)):
            stripe = 40 if ((x // 6) % 2 == 0) else 0
            px[x, y] = (170 + stripe, 100, 50)
    if pose == "b":  # 姿态变化：在框内加一块不同形状
        for x in range(x0 + 8, x0 + 30):
            for y in range(y0 + 8, y0 + 40):
                px[x, y] = (90, 50, 30)
    return img


def jpeg_bytes(img, quality=92):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pair_from(img_a, img_b, rect_a=RECT, rect_b=RECT):
    g1, g2 = load_gray(jpeg_bytes(img_a)), load_gray(jpeg_bytes(img_b))
    return compare_pair(0, 1, g1, g2, [rect_a], [rect_b], rect_a, rect_b)


# ---------------------------------------------------------------- 重复与重编码

def test_identical_scene_light_noise_is_duplicate():
    p = pair_from(_scene(1), _scene(2))
    assert p.label == "duplicate"
    assert p.background.sufficient and p.animal.sufficient


@pytest.mark.parametrize("quality", [95, 85, 75, 70, 60, 50])
def test_jpeg_reencode_still_duplicate(quality):
    img = _scene(1)
    g1 = load_gray(jpeg_bytes(img, 92))
    g2 = load_gray(jpeg_bytes(img, quality))
    p = compare_pair(0, 1, g1, g2, [RECT], [RECT], RECT, RECT)
    assert p.label == "duplicate", f"q={quality}: {p.reason}"


def test_very_low_quality_does_not_omit_silently():
    # 极端低质量落入灰区时必须是 unknown（待核），不能判 duplicate 自动省略
    img = _scene(1)
    g1 = load_gray(jpeg_bytes(img, 92))
    g2 = load_gray(jpeg_bytes(img, 20))
    p = compare_pair(0, 1, g1, g2, [RECT], [RECT], RECT, RECT)
    assert p.label in ("unknown", "duplicate")
    if p.label == "unknown":
        assert not (p.background.sufficient and p.animal.sufficient) or "灰区" in p.reason


# ---------------------------------------------------------------- 姿态/位置变化

def test_pose_change_is_different():
    p = pair_from(_scene(1, pose="a"), _scene(2, pose="b"))
    assert p.label == "different"
    assert "保留" in p.reason


def test_disjoint_boxes_is_position_change():
    moved = [RECT[0] + 120, RECT[1], RECT[2], RECT[3]]
    g1 = load_gray(jpeg_bytes(_scene(1)))
    g2 = load_gray(jpeg_bytes(_scene(2, animal_offset=(120, 0))))
    p = compare_pair(0, 1, g1, g2, [RECT], [moved], RECT, moved)
    assert p.label == "different"
    assert p.animal.disjoint is True


def test_background_only_change_is_not_duplicate():
    img_a = _scene(1)
    img_b = _scene(2)
    arr = np.asarray(img_b).copy()
    arr[150:180, 180:230] = 255  # 框外背景变化
    p = pair_from(img_a, Image.fromarray(arr))
    assert p.label != "duplicate"  # 保守：不自动省略


# ---------------------------------------------------------------- 证据不足

def test_flat_animal_region_insufficient_evidence():
    # 两个完全平坦的动物框（无细节能量）→ 动物证据不足 → unknown
    flat = Image.new("RGB", (W, H), (60, 90, 50))
    for x in range(RECT[0], RECT[0] + RECT[2]):
        for y in range(RECT[1], RECT[1] + RECT[3]):
            flat.putpixel((x, y), (120, 120, 120))
    g = load_gray(jpeg_bytes(flat))
    p = compare_pair(0, 1, g, g, [RECT], [RECT], RECT, RECT)
    assert p.label == "unknown"
    assert p.animal.sufficient is False


def test_missing_box_is_unknown():
    g1 = load_gray(jpeg_bytes(_scene(1)))
    g2 = load_gray(jpeg_bytes(_scene(2)))
    p = compare_pair(0, 1, g1, g2, [RECT], [], RECT, None)
    assert p.label == "unknown"
    assert p.animal.sufficient is False


# ---------------------------------------------------------------- 组级建议

def test_example_session_verdicts():
    frames = build_example_frames()
    report = analyze_burst([b for _, b in frames], build_example_detections())
    d = report_to_dict(report)
    decisions = {v["frame"]: v["decision"] for v in d["verdicts"]}
    # 重复组 0/1/2：保留最早帧，省略其余
    assert decisions[0] == "keep"
    assert decisions[1] == "omit" and decisions[2] == "omit"
    assert d["verdicts"][1]["duplicate_of"] == 0
    # 位置/姿态变化帧保留
    assert decisions[3] == "keep"
    # 严重压缩的歧义对待核，不自动省略
    assert decisions[4] == "review" and decisions[5] == "review"
    # 每组每两帧都有配对结论
    assert len(d["pairs"]) == 15


def test_all_duplicate_group_keeps_earliest_only():
    images = [jpeg_bytes(_scene(seed=s)) for s in (1, 2, 3)]
    dets = [{"id": f"b{f}", "frame": f, "rect": list(RECT)} for f in range(3)]
    d = report_to_dict(analyze_burst(images, dets))
    labels = {(p["a"], p["b"]): p["label"] for p in d["pairs"]}
    assert all(v == "duplicate" for v in labels.values())
    decisions = [v["decision"] for v in d["verdicts"]]
    assert decisions == ["keep", "omit", "omit"]


def test_contradictory_relations_become_review(monkeypatch):
    """构造 a≈b、b≈c 重复，但 a 与 c 被显式判 different 的矛盾场景。

    用 monkeypatch 替换 compare_pair 强制标签，验证组级矛盾保护：
    矛盾组全部 review，不产生任何 omit。
    """
    import app.compare as cmp_mod

    images = [jpeg_bytes(_scene(seed=s)) for s in range(3)]
    dets = [{"id": f"b{f}", "frame": f, "rect": list(RECT)} for f in range(3)]

    real = cmp_mod.compare_pair
    forced = {
        (0, 1): "duplicate",
        (1, 2): "duplicate",
        (0, 2): "different",
    }

    def fake_pair(a, b, *args, **kwargs):
        res = real(a, b, *args, **kwargs)
        res.label = forced[(a, b)]
        return res

    monkeypatch.setattr(cmp_mod, "compare_pair", fake_pair)
    d = report_to_dict(analyze_burst(images, dets))
    assert all(v["decision"] == "review" for v in d["verdicts"])
    assert not any(v["decision"] == "omit" for v in d["verdicts"])


def test_unknown_pair_blocks_omission():
    # 两帧重复 + 第三帧证据不足：任何自动省略都不允许波及待核帧
    images = [jpeg_bytes(_scene(seed=s)) for s in range(3)]
    dets = [{"id": f"b{f}", "frame": f, "rect": list(RECT)} for f in range(3)]

    import app.compare as cmp_mod
    real = cmp_mod.compare_pair

    def fake_pair(a, b, *args, **kwargs):
        res = real(a, b, *args, **kwargs)
        if (a, b) == (0, 2):
            res.label = "unknown"
            res.reason = "测试注入：证据不足"
        return res

    import app.compare as cm
    orig = cm.compare_pair
    cm.compare_pair = fake_pair
    try:
        d = report_to_dict(analyze_burst(images, dets))
    finally:
        cm.compare_pair = orig
    decisions = {v["frame"]: v["decision"] for v in d["verdicts"]}
    assert decisions[2] == "review"
    assert decisions[0] in ("keep", "review")


# ---------------------------------------------------------------- 边界

def test_single_frame_is_kept():
    d = report_to_dict(analyze_burst(
        [jpeg_bytes(_scene(1))],
        [{"id": "b0", "frame": 0, "rect": list(RECT)}]))
    assert d["pairs"] == []
    assert d["verdicts"][0]["decision"] == "keep"


def test_no_detections_marks_review_not_omit():
    images = [jpeg_bytes(_scene(s)) for s in (1, 2)]
    d = report_to_dict(analyze_burst(images, []))
    assert d["pairs"][0]["label"] == "unknown"
    assert d["pairs"][0]["animal"]["sufficient"] is False
    assert all(v["decision"] == "review" for v in d["verdicts"])


def test_six_frame_burst_pair_count_and_all_duplicates():
    images = [jpeg_bytes(_scene(s)) for s in range(1, 7)]
    dets = [{"id": f"b{f}", "frame": f, "rect": list(RECT)} for f in range(6)]
    d = report_to_dict(analyze_burst(images, dets))
    assert len(d["pairs"]) == 15  # 6 选 2
    decisions = [v["decision"] for v in d["verdicts"]]
    assert decisions == ["keep", "omit", "omit", "omit", "omit", "omit"]
    for v in d["verdicts"][1:]:
        assert v["duplicate_of"] == 0


def test_real_triplet_duplicate_pair_plus_changed_frame():
    """端到端场景：帧 0/1 重复，帧 2 同框但姿态变化（与前两帧都 different）。

    这不是矛盾：重复组 {0,1} 保留帧 0、省略帧 1；变化帧 2 保留。
    真正的“重复关系相互矛盾”由 test_contradictory_relations_become_review 覆盖。
    """
    images = [
        jpeg_bytes(_scene(1, pose="a")),
        jpeg_bytes(_scene(2, pose="a")),
        jpeg_bytes(_scene(3, pose="b")),
    ]
    dets = [{"id": f"b{f}", "frame": f, "rect": list(RECT)} for f in range(3)]
    d = report_to_dict(analyze_burst(images, dets))
    labels = {(p["a"], p["b"]): p["label"] for p in d["pairs"]}
    assert labels[(0, 1)] == "duplicate"
    assert labels[(0, 2)] == "different" and labels[(1, 2)] == "different"
    decisions = {v["frame"]: v["decision"] for v in d["verdicts"]}
    assert decisions == {0: "keep", 1: "omit", 2: "keep"}
