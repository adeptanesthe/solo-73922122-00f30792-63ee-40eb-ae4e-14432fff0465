"""连拍重复帧比对引擎（相机陷阱核查台）。

设计要点
========
1. 两类区域分开比较：
   - 背景区域：两帧检测框的**并集之外**（动物移动会改变框位置，框内变化
     不能当作背景变化），用均值灰度做鲁棒比对；
   - 动物区域：两帧检测框的**交集**（同一批像素才可逐像素比较），
     在灰度上计算 RMSE / 相关系数，并在 8x8 小窗上计算方向梯度直方图
     （HOG-lite）以捕捉姿态变化。
2. 容忍轻微 JPEG 重编码：所有指标先对图像做轻度高斯模糊，去块效应后再算；
   阈值按“高质量重编码帧判同、真实内容变化判异”校准。
3. 证据充分性：区域有效像素太少（框过小）或细节能量太低
   （平坦区域、严重压缩伪影）都判为“证据不足”，该区域不下结论；
   两框无交集则反过来作为动物整体移位的强证据，直接判内容变化。
4. 保守策略：
   - 动物区域只要证据充分且显示位置/姿态变化 → 该帧必须保留（different）；
   - 背景与动物**都**证据充分且都相符才可能判重复（duplicate）；
   - 任一区域证据不足 → unknown（待核），不自动省略；
   - 组内重复关系相互矛盾（如 a≈b、b≈c 但 a≠c）→ 相关帧全部待核。

本模块只做只读分析，不修改原图、检测框、轨迹或计数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------- 阈值
# 这些阈值在 tests/test_compare.py 与校准实验中按三类场景验证：
#   重复帧（含 JPEG 重编码 q>=50）、姿态/位置变化帧、证据不足帧。
BG_RMSE_DUP = 6.0          # 背景 RMSE（0~255）低于此值视为相符
BG_CORR_DUP = 0.97         # 背景相关系数高于此值视为相符
ANIMAL_RMSE_DUP = 10.0     # 动物区域 RMSE 低于此值视为相符
ANIMAL_CORR_DUP = 0.93     # 动物区域相关系数高于此值视为相符
ANIMAL_HOG_DUP = 0.20      # HOG 直方图交距离低于此值视为姿态相符（容忍重编码）
ANIMAL_RMSE_DIFF = 12.0    # RMSE 高于此值直接判定内容变化（重编码通常 < 3）
ANIMAL_HOG_DIFF = 0.24     # HOG 距离高于此值视为姿态可能变化（需 RMSE 佐证）
ANIMAL_RMSE_HOG_PROVE = 6.0  # HOG 超阈时，RMSE 超过此值即可确认姿态变化
# 相关性极低兜底：低于此值说明细节被压缩伪影/模糊整体破坏，
# 即使 HOG 超阈也不判变化（伪影本身会抬高 HOG），转待核。
ANIMAL_CORR_FLOOR = 0.55
BG_CORR_RELIABLE = 0.95    # 背景相关性低于此值时背景证据不可信

MIN_BG_PIXELS = 2000       # 背景有效像素下限
MIN_ANIMAL_PIXELS = 256    # 动物交集像素下限
MIN_BG_ENERGY = 8.0        # 背景细节能量（去均值后的标准差）下限
MIN_ANIMAL_ENERGY = 5.0    # 动物区域细节能量下限（低于此值视为平坦/严重模糊）

GRID = 8                   # HOG-lite 网格
N_BINS = 6                 # 梯度方向桶数（每桶 30°，无符号梯度）


# ---------------------------------------------------------------- 数据结构

@dataclass
class RegionEvidence:
    sufficient: bool
    reason: str = ""
    pixels: int = 0
    rmse: float | None = None
    corr: float | None = None
    energy: float | None = None
    hog_dist: float | None = None
    disjoint: bool = False  # 仅动物区域：两框无交集（位置变化强证据）


@dataclass
class PairResult:
    a: int
    b: int
    label: str               # "duplicate" | "different" | "unknown"
    reason: str
    background: RegionEvidence
    animal: RegionEvidence


@dataclass
class FrameVerdict:
    frame: int
    decision: str            # "keep" | "omit" | "review"
    reason: str
    duplicate_of: int | None = None


@dataclass
class CompareReport:
    n_frames: int
    pairs: list[PairResult] = field(default_factory=list)
    verdicts: list[FrameVerdict] = field(default_factory=list)
    groups: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- 图像工具

def load_gray(image_bytes: bytes) -> np.ndarray:
    """解码图片为 float32 灰度数组；解码失败抛出 ValueError。"""
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"图片无法解码：{exc}") from exc
    gray = img.convert("L")
    # 轻度高斯模糊：抑制 JPEG 块效应与传感器噪声，保留结构
    gray = gray.filter(ImageFilter.GaussianBlur(radius=1.0))
    return np.asarray(gray, dtype=np.float32)


def _hog_lite(tiles: list[np.ndarray]) -> np.ndarray:
    """对一组等尺寸小图计算 HOG-lite 特征（全局 L1 归一化直方图）。"""
    hist = np.zeros(GRID * GRID * N_BINS, dtype=np.float64)
    for tile in tiles:
        h, w = tile.shape
        gx = np.zeros_like(tile)
        gy = np.zeros_like(tile)
        gx[:, 1:-1] = tile[:, 2:] - tile[:, :-2]
        gy[1:-1, :] = tile[2:, :] - tile[:-2, :]
        mag = np.hypot(gx, gy)
        # 无符号方向（0~π）
        angle = np.arctan2(gy, gx) % math.pi
        bin_idx = np.minimum((angle / math.pi * N_BINS).astype(int), N_BINS - 1)
        for gy_i in range(GRID):
            for gx_i in range(GRID):
                y0, y1 = gy_i * h // GRID, (gy_i + 1) * h // GRID
                x0, x1 = gx_i * w // GRID, (gx_i + 1) * w // GRID
                m = mag[y0:y1, x0:x1]
                b = bin_idx[y0:y1, x0:x1]
                cell = np.bincount(b.ravel(), weights=m.ravel(), minlength=N_BINS)
                hist[(gy_i * GRID + gx_i) * N_BINS:(gy_i * GRID + gx_i + 1) * N_BINS] += cell
    # 全局 L1 归一化：直方图各桶和为 1，两直方图交 ∈ [0,1]
    total = hist.sum() + 1e-9
    return hist / total


def _hog_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    """两等尺寸小图的 HOG 距离 = 1 − 直方图交（0~1，越大姿态越不同）。"""
    h1 = _hog_lite([t1])
    h2 = _hog_lite([t2])
    return float(max(0.0, 1.0 - np.minimum(h1, h2).sum()))


def _metrics(t1: np.ndarray, t2: np.ndarray, with_hog: bool) -> dict:
    """计算两块等尺寸区域的 RMSE / 相关系数 / 细节能量 / HOG 距离。"""
    a = t1.ravel().astype(np.float64)
    b = t2.ravel().astype(np.float64)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    sa, sb = a.std(), b.std()
    corr = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb + 1e-9))
    corr = max(-1.0, min(1.0, corr))
    energy = float(min(sa, sb))  # 取两者较小值：任一帧细节不足都算不足
    out = {"rmse": rmse, "corr": corr, "energy": energy}
    if with_hog:
        out["hog_dist"] = _hog_distance(t1, t2)
    return out


# ---------------------------------------------------------------- 区域比对

def _box_union_mask(shape, boxes_a, boxes_b):
    """两帧所有检测框的并集掩码（框内 = 动物可能出现的位置）。"""
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for boxes in (boxes_a, boxes_b):
        for x, y, bw, bh in boxes:
            x0 = max(0, int(math.floor(x)))
            y0 = max(0, int(math.floor(y)))
            x1 = min(w, int(math.ceil(x + bw)))
            y1 = min(h, int(math.ceil(y + bh)))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
    return mask


def _intersection_mask(shape, rect_a, rect_b):
    """两个检测框的交集矩形掩码；无交集时返回 None。"""
    h, w = shape
    ax0 = max(0, int(math.floor(rect_a[0])))
    ay0 = max(0, int(math.floor(rect_a[1])))
    ax1 = min(w, int(math.ceil(rect_a[0] + rect_a[2])))
    ay1 = min(h, int(math.ceil(rect_a[1] + rect_a[3])))
    bx0 = max(0, int(math.floor(rect_b[0])))
    by0 = max(0, int(math.floor(rect_b[1])))
    bx1 = min(w, int(math.ceil(rect_b[0] + rect_b[2])))
    by1 = min(h, int(math.ceil(rect_b[1] + rect_b[3])))
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (slice(y0, y1), slice(x0, x1))


def compare_background(g1: np.ndarray, g2: np.ndarray,
                       boxes_a: list, boxes_b: list) -> RegionEvidence:
    """背景区域比对：检测框并集之外的像素。"""
    union = _box_union_mask(g1.shape, boxes_a, boxes_b)
    bg_mask = ~union
    pixels = int(bg_mask.sum())
    if pixels < MIN_BG_PIXELS:
        return RegionEvidence(False, f"背景有效像素仅 {pixels}（< {MIN_BG_PIXELS}），证据不足", pixels)
    t1, t2 = g1[bg_mask], g2[bg_mask]
    m = _metrics(t1, t2, with_hog=False)
    if m["energy"] < MIN_BG_ENERGY:
        return RegionEvidence(
            False, f"背景过于平坦（细节能量 {m['energy']:.1f} < {MIN_BG_ENERGY}），证据不足",
            pixels, m["rmse"], m["corr"], m["energy"])
    return RegionEvidence(True, "", pixels, m["rmse"], m["corr"], m["energy"])


def compare_animal(g1: np.ndarray, g2: np.ndarray,
                   rect_a, rect_b) -> RegionEvidence:
    """动物区域比对：两帧检测框的交集。

    - 任一侧无框：没有可比较的动物像素，证据不足；
    - 两框无交集：动物整体移动到画面别处，属于位置变化的强证据，
      直接以特殊标记返回（sufficient=True，disjoint=True），由上层判 different。
    """
    if rect_a is None or rect_b is None:
        side = "前一帧" if rect_a is None else "后一帧"
        return RegionEvidence(False, f"{side}该帧无检测框，动物区域证据不足")
    sl = _intersection_mask(g1.shape, rect_a, rect_b)
    if sl is None:
        return RegionEvidence(
            True, "两个检测框无交集：动物已移动到画面别处（位置变化的强证据）。",
            0, None, None, None, None, disjoint=True)
    t1, t2 = g1[sl], g2[sl]
    pixels = int(t1.size)
    if pixels < MIN_ANIMAL_PIXELS:
        return RegionEvidence(
            False, f"动物交集仅 {pixels} 像素（< {MIN_ANIMAL_PIXELS}），证据不足", pixels)
    m = _metrics(t1, t2, with_hog=True)
    if m["energy"] < MIN_ANIMAL_ENERGY:
        return RegionEvidence(
            False,
            f"动物区域细节能量 {m['energy']:.1f} < {MIN_ANIMAL_ENERGY}（疑似严重压缩/模糊），证据不足",
            pixels, m["rmse"], m["corr"], m["energy"], m["hog_dist"])
    return RegionEvidence(
        True, "", pixels, m["rmse"], m["corr"], m["energy"], m["hog_dist"])


# ---------------------------------------------------------------- 逐帧对

def compare_pair(frame_a: int, frame_b: int,
                 g1: np.ndarray, g2: np.ndarray,
                 boxes_a: list, boxes_b: list,
                 rect_a, rect_b) -> PairResult:
    """比较两帧，输出 duplicate / different / unknown 及两区域证据。"""
    bg = compare_background(g1, g2, boxes_a, boxes_b)
    an = compare_animal(g1, g2, rect_a, rect_b)

    # 0) 两框无交集：动物整体移位，位置变化的强证据，直接判 different
    if an.disjoint:
        return PairResult(
            frame_a, frame_b, "different",
            "动物位置有可辨变化，保留该帧：两个检测框无交集，动物已移动到画面别处。",
            bg, an)

    bg_match = (
        bg.sufficient
        and bg.rmse is not None
        and bg.rmse <= BG_RMSE_DUP
        and bg.corr is not None
        and bg.corr >= BG_CORR_DUP
    )
    bg_reliable = (
        bg.sufficient and bg.corr is not None and bg.corr >= BG_CORR_RELIABLE
    )
    an_match = (
        an.sufficient
        and an.rmse is not None
        and an.rmse <= ANIMAL_RMSE_DUP
        and an.corr is not None
        and an.corr >= ANIMAL_CORR_DUP
        and an.hog_dist is not None
        and an.hog_dist <= ANIMAL_HOG_DUP
    )
    # 真实姿态/位置变化（联合条件，避免被单纯压缩伪影误导）：
    # - RMSE 强变化路径：像素级差异已非常大，本身即可靠，不受相关性兜底限制；
    # - HOG 路径：HOG 超阈且 RMSE 同步升高、相关性仍强（伪影会同时打掉相关性）。
    an_corr_ok = an.corr is not None and an.corr >= ANIMAL_CORR_FLOOR
    # 真实姿态变化会破坏逐像素对齐、相关性明显下降；轻微重编码 corr 接近 1。
    # 用高门槛把“HOG 超阈但仍高度相关”的情形识别为伪影而非姿态变化。
    an_corr_strong = an.corr is not None and an.corr >= 0.97
    an_rmse_strong = an.rmse is not None and an.rmse >= ANIMAL_RMSE_DIFF
    an_changed = (
        an.sufficient and (
            an_rmse_strong
            or (
                an_corr_ok and an_corr_strong
                and an.hog_dist is not None and an.hog_dist >= ANIMAL_HOG_DIFF
                and an.rmse is not None and an.rmse >= ANIMAL_RMSE_HOG_PROVE
            )
        )
    )
    # HOG 超阈但相关性不足以支撑“真实姿态变化” → 伪影嫌疑，待核
    an_suspect_artifact = (
        an.sufficient
        and not an_rmse_strong
        and an.hog_dist is not None and an.hog_dist >= ANIMAL_HOG_DIFF
        and not an_corr_strong
    )

    # 1) 动物位置/姿态有可辨变化（证据可信）→ 必须保留（different）
    if an_changed:
        return PairResult(frame_a, frame_b, "different", _change_reason(an), bg, an)

    # 2) 两类区域都有充分证据且都相符 → duplicate
    if bg.sufficient and an.sufficient and bg_match and an_match:
        reason = (
            f"背景相符（RMSE {bg.rmse:.2f} ≤ {BG_RMSE_DUP}，相关 {bg.corr:.3f}）；"
            f"动物区域相符（RMSE {an.rmse:.2f}，相关 {an.corr:.3f}，姿态距离 {an.hog_dist:.3f}）"
        )
        return PairResult(frame_a, frame_b, "duplicate", reason, bg, an)

    # 3) 证据充分但相关性不足（压缩伪影/模糊淹没信号），或指标落在灰区 → 待核
    if not bg.sufficient or not an.sufficient:
        weak = [r.reason for r in (bg, an) if not r.sufficient]
        reason = "证据不足，标为待核：" + "；".join(weak)
    elif an_suspect_artifact or not bg_reliable or not an_corr_ok:
        reason = (
            "区域相关性过低，细节疑似被压缩伪影或模糊破坏，证据不可信，标为待核："
            f"背景相关 {bg.corr:.3f}（门槛 {BG_CORR_RELIABLE}），"
            f"动物相关 {an.corr:.3f}（可信门槛 {ANIMAL_CORR_FLOOR}），"
            f"姿态距离 {an.hog_dist:.3f}"
        )
    else:
        reason = (
            "指标处于灰区，无法确认重复也无法确认变化，标为待核："
            f"背景 RMSE {bg.rmse:.2f}/相关 {bg.corr:.3f}，"
            f"动物 RMSE {an.rmse:.2f}/相关 {an.corr:.3f}/姿态 {an.hog_dist:.3f}"
        )
    return PairResult(frame_a, frame_b, "unknown", reason, bg, an)


def _change_reason(an: RegionEvidence) -> str:
    parts = []
    if an.rmse is not None and an.rmse >= ANIMAL_RMSE_DIFF:
        parts.append(f"动物区域像素变化明显（RMSE {an.rmse:.2f} ≥ {ANIMAL_RMSE_DIFF}）")
    if an.hog_dist is not None and an.hog_dist >= ANIMAL_HOG_DIFF:
        parts.append(
            f"姿态/轮廓变化明显（姿态距离 {an.hog_dist:.3f} ≥ {ANIMAL_HOG_DIFF}，"
            f"RMSE {an.rmse:.2f} ≥ {ANIMAL_RMSE_HOG_PROVE} 佐证）"
        )
    return "动物位置或姿态有可辨变化，保留该帧：" + "；".join(parts)


# ---------------------------------------------------------------- 组级汇总

def _build_verdicts(n_frames: int, pairs: list[PairResult]) -> tuple[list[FrameVerdict], list[dict]]:
    """按重复关系并查集分组，校验组内每两帧是否相符，生成帧级建议。

    决策规则（保守，自动省略门槛最高）：
    - duplicate 边做并查集合并；
    - 重复组件（成员 ≥2）只有在**组内每两帧都显式为 duplicate**、
      且没有任何 unknown/diff 边把成员连向组外时，才建议保留最早帧、
      省略其余帧；
    - 重复组件内部出现 diff（重复关系相互矛盾）或 unknown（证据不足）：
      组件全部标待核，不自动省略；
    - 成员与组外存在 unknown 边：该成员及其重复组标待核；
    - 成员与组外存在 diff 边：内容变化对端是不同画面，变化帧标 keep；
      重复组本身仍可按重复规则省略（变化关系不影响组内重复结论）；
    - 孤立单帧：默认 keep（无重复关系）；若其所有跨组边都是 diff，
      同样 keep；若涉及 unknown 边则 review。
    """
    parent = list(range(n_frames))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pair_index = {(p.a, p.b): p for p in pairs}
    diff_edges, unknown_edges = [], []
    for p in pairs:
        if p.label == "duplicate":
            union(p.a, p.b)
        elif p.label == "different":
            diff_edges.append((p.a, p.b))
        else:
            unknown_edges.append((p.a, p.b))

    comp: dict[int, list[int]] = {}
    for f in range(n_frames):
        comp.setdefault(find(f), []).append(f)

    # 每个组件收集内部/外部的异常边
    review_reason: dict[int, list[str]] = {}
    keep_reason: dict[int, list[str]] = {}

    def add_reason(bucket, f, text):
        bucket.setdefault(f, [])
        if text not in bucket[f]:
            bucket[f].append(text)

    for a, b in diff_edges:
        # 同组件内出现 diff = 重复关系相互矛盾
        if find(a) == find(b):
            add_reason(review_reason, a,
                       f"重复关系相互矛盾：帧 {a}、帧 {b} 既经重复关系同组，该对又检出内容变化。")
            add_reason(review_reason, b,
                       f"重复关系相互矛盾：帧 {a}、帧 {b} 既经重复关系同组，该对又检出内容变化。")
        else:
            p = pair_index[(a, b)]
            add_reason(keep_reason, a, f"帧 {a} 与帧 {b} 内容不同（{_short(p.reason)}），保留。")
            add_reason(keep_reason, b, f"帧 {b} 与帧 {a} 内容不同（{_short(p.reason)}），保留。")

    for a, b in unknown_edges:
        p = pair_index[(a, b)]
        text = f"帧 {a}–帧 {b} 证据不足（{_short(p.reason)}），需人工核查。"
        add_reason(review_reason, a, text)
        add_reason(review_reason, b, text)

    verdicts: dict[int, FrameVerdict] = {}
    groups: list[dict] = []

    for root, members in sorted(comp.items(), key=lambda kv: min(kv[1])):
        members.sort()
        reviewed = [f for f in members if f in review_reason]

        # 多帧重复组件还要检查跨组 unknown 边：成员与组外帧证据不足时，
        # 无法确认该组之外是否还存在重复关系，整组转待核
        if len(members) > 1 and not reviewed:
            cross_unknown = []
            member_set = set(members)
            for a, b in unknown_edges:
                if (a in member_set) != (b in member_set):
                    cross_unknown.append((a, b))
            if cross_unknown:
                text = "；".join(
                    f"帧 {a}–帧 {b} 证据不足（{_short(pair_index[(a, b)].reason)}）"
                    for a, b in cross_unknown)
                reason = f"重复组与组外帧存在证据不足的配对（{text}），整组标为待核，不自动省略。"
                for f in members:
                    verdicts[f] = FrameVerdict(f, "review", reason)
                groups.append({"frames": members, "decision": "review", "reason": reason})
                continue

        if reviewed:
            reason = "；".join(review_reason[reviewed[0]]) + " 不自动省略，标为待核。"
            for f in members:
                verdicts[f] = FrameVerdict(f, "review", reason)
            groups.append({"frames": members, "decision": "review", "reason": reason})
            continue

        if len(members) == 1:
            f = members[0]
            if f in keep_reason:
                reason = "；".join(keep_reason[f])
            else:
                reason = "无重复关系，保留该帧。"
            verdicts[f] = FrameVerdict(f, "keep", reason)
            groups.append({"frames": [f], "decision": "keep", "reason": reason})
            continue

        # 多帧重复组件：传递性复核，组内每两帧都必须显式 duplicate
        earliest = members[0]
        bad_pairs = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                p = pair_index.get((members[i], members[j]))
                if p is None or p.label != "duplicate":
                    bad_pairs.append((members[i], members[j]))
        if bad_pairs:
            reason = ("组内并非每两帧都相符（"
                      + "、".join(f"帧{a}–帧{b}" for a, b in bad_pairs)
                      + "），重复关系不闭合，全部标为待核，不自动省略。")
            for f in members:
                verdicts[f] = FrameVerdict(f, "review", reason)
            groups.append({"frames": members, "decision": "review", "reason": reason})
            continue

        keep_reason_text = (
            f"重复组 {len(members)} 帧（帧 "
            + "、".join(str(f) for f in members)
            + "）的背景与动物区域均有充分可辨证据且两两相符，保留最早帧。"
        )
        extra = keep_reason.get(earliest)
        verdicts[earliest] = FrameVerdict(
            earliest, "keep",
            keep_reason_text + ("；" + "；".join(extra) if extra else ""))
        for f in members[1:]:
            verdicts[f] = FrameVerdict(
                f, "omit",
                f"与最早帧（帧 {earliest}）背景与动物区域均相符，建议省略；原图与检测框不变。",
                duplicate_of=earliest)
        groups.append({"frames": members, "decision": "omit_duplicates",
                       "keep": earliest, "reason": keep_reason_text})

    ordered = [verdicts[f] for f in range(n_frames)]
    return ordered, groups


def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


# ---------------------------------------------------------------- 入口

def analyze_burst(images: list[bytes], detections: list[dict]) -> CompareReport:
    """对一次连拍会话做完整重复帧分析。

    images: 原始图片字节列表（1~6 张，尺寸相同且 ≤512x512，由 API 层校验）
    detections: [{"id","frame","rect":[x,y,w,h]}]
    返回 CompareReport（只读分析，不改动任何输入）。
    """
    n = len(images)
    grays = [load_gray(b) for b in images]

    # 每帧的框（按面积降序），主框取最大者
    boxes_by_frame: dict[int, list] = {f: [] for f in range(n)}
    for d in detections:
        boxes_by_frame[d["frame"]].append(d["rect"])
    for f in boxes_by_frame:
        boxes_by_frame[f].sort(key=lambda r: r[2] * r[3], reverse=True)

    pairs: list[PairResult] = []
    for a in range(n):
        for b in range(a + 1, n):
            rect_a = boxes_by_frame[a][0] if boxes_by_frame[a] else None
            rect_b = boxes_by_frame[b][0] if boxes_by_frame[b] else None
            pairs.append(compare_pair(
                a, b, grays[a], grays[b],
                boxes_by_frame[a], boxes_by_frame[b], rect_a, rect_b))

    verdicts, groups = _build_verdicts(n, pairs)
    return CompareReport(n_frames=n, pairs=pairs, verdicts=verdicts, groups=groups)


def report_to_dict(report: CompareReport) -> dict:
    """序列化为 API JSON。"""
    def region(r: RegionEvidence) -> dict:
        return {
            "sufficient": r.sufficient,
            "reason": r.reason,
            "pixels": r.pixels,
            "rmse": None if r.rmse is None else round(r.rmse, 3),
            "corr": None if r.corr is None else round(r.corr, 4),
            "energy": None if r.energy is None else round(r.energy, 3),
            "hog_dist": None if r.hog_dist is None else round(r.hog_dist, 4),
            "disjoint": r.disjoint,
        }

    return {
        "n_frames": report.n_frames,
        "pairs": [
            {
                "a": p.a, "b": p.b, "label": p.label, "reason": p.reason,
                "background": region(p.background),
                "animal": region(p.animal),
            }
            for p in report.pairs
        ],
        "verdicts": [
            {"frame": v.frame, "decision": v.decision, "reason": v.reason,
             "duplicate_of": v.duplicate_of}
            for v in report.verdicts
        ],
        "groups": report.groups,
    }
