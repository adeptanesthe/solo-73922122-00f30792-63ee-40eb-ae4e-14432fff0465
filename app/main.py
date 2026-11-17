"""FastAPI 后端：相机陷阱连拍重复帧核查台。

会话保存在进程内存中（单用户工具，重启即清空）。
分析任务记录启动时的输入版本号；前端切换会话或重新分析时撤下旧建议，
迟到的旧结果按版本号校验丢弃，不会覆盖新会话/新输入。

本服务只做只读分析：原图片、检测框在会话创建后不再修改，
也不提供任何轨迹或计数写入能力。
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .compare import analyze_burst, report_to_dict
from .example import EXAMPLE_DESCRIPTION, build_example_detections, build_example_frames

MAX_FRAMES = 6
MAX_SIDE = 512
MAX_DETECTIONS = 30
MAX_IMAGE_BYTES = 20 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="相机陷阱连拍重复帧核查台")

SESSIONS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------- 工具

def _err(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


def _get_session(sid: str) -> dict:
    sess = SESSIONS.get(sid)
    if sess is None:
        _err(404, "会话不存在或已过期，请重新上传数据。")
    return sess


def _decode_and_check_image(data: bytes, filename: str, expect_size: tuple | None):
    """解码图片并校验尺寸；返回 (content_type, width, height, normalized_bytes)。

    统一转存为 PNG 字节，保证后续比对解码稳定；原图展示仍使用原始字节。
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        _err(400, f"图片 {filename} 无法解码：{exc}")
    w, h = img.size
    if w > MAX_SIDE or h > MAX_SIDE:
        _err(400, f"图片 {filename} 尺寸 {w}x{h} 超过上限 {MAX_SIDE}x{MAX_SIDE}。")
    if expect_size is not None and (w, h) != expect_size:
        _err(400, f"图片 {filename} 尺寸 {w}x{h} 与首张 {expect_size[0]}x{expect_size[1]} 不一致，"
                  "连拍帧必须尺寸相同。")
    return w, h


def _validate_detections(payload, n_frames: int, img_size: tuple[int, int]):
    """校验并规范化检测 JSON。接受 {"detections": [...]} 或直接 [...]。"""
    if isinstance(payload, dict):
        dets = payload.get("detections")
    else:
        dets = payload
    if not isinstance(dets, list):
        _err(400, "检测 JSON 必须是数组，或形如 {\"detections\": [...]} 的对象。")
    if len(dets) > MAX_DETECTIONS:
        _err(400, f"检测框数量 {len(dets)} 超过上限 {MAX_DETECTIONS}。")

    W, H = img_size
    seen_ids = set()
    out = []
    for k, d in enumerate(dets):
        ctx = f"第 {k} 个检测框"
        if not isinstance(d, dict):
            _err(400, f"{ctx}：必须是对象。")
        bid = d.get("id")
        if not isinstance(bid, (str, int)) or isinstance(bid, bool) or str(bid) == "":
            _err(400, f"{ctx}：框号 id 必须是非空字符串或数字。")
        bid = str(bid)
        if bid in seen_ids:
            _err(400, f"框号 {bid} 重复。")
        seen_ids.add(bid)

        frame = d.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool):
            _err(400, f"框 {bid}：帧号 frame 必须是整数。")
        if not (0 <= frame < n_frames):
            _err(400, f"框 {bid}：帧号 {frame} 超出范围 [0, {n_frames - 1}]。")

        rect = d.get("rect")
        if not isinstance(rect, list) or len(rect) != 4:
            _err(400, f"框 {bid}：rect 必须是 [x, y, w, h] 四元数组。")
        rect_v = []
        for v in rect:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                _err(400, f"框 {bid}：rect 含有非法数值。")
            rect_v.append(float(v))
        x, y, bw, bh = rect_v
        if bw <= 0 or bh <= 0:
            _err(400, f"框 {bid}：rect 的宽、高必须为正数。")
        # 允许检测框略微越界：裁剪到画面内后必须仍有正面积
        x0 = max(0.0, x)
        y0 = max(0.0, y)
        x1 = min(float(W), x + bw)
        y1 = min(float(H), y + bh)
        if x1 <= x0 or y1 <= y0:
            _err(400, f"框 {bid}：与画面无交集，无法用于动物区域比对。")

        out.append({"id": bid, "frame": frame, "rect": rect_v})
    out.sort(key=lambda d: (d["frame"], d["id"]))
    return out


def _new_session(frames, detections) -> dict:
    sid = uuid.uuid4().hex[:12]
    sess = {
        "id": sid,
        "version": 1,
        "frames": frames,  # [(content_type, bytes)]
        "detections": detections,
    }
    SESSIONS[sid] = sess
    return sess


def _payload(sess: dict) -> dict:
    return {
        "session_id": sess["id"],
        "version": sess["version"],
        "n_frames": len(sess["frames"]),
        "detections": sess["detections"],
        "limits": {"frames": MAX_FRAMES, "max_side": MAX_SIDE, "detections": MAX_DETECTIONS},
    }


# ---------------------------------------------------------------- 会话

@app.post("/api/sessions")
async def create_session(images: list[UploadFile] = File(...),
                         detections: UploadFile = File(...)):
    """上传 1~6 张同尺寸（≤512x512）帧图片 + 检测 JSON，创建会话。"""
    if not (1 <= len(images) <= MAX_FRAMES):
        _err(400, f"图片数量须为 1~{MAX_FRAMES}，收到 {len(images)}。")
    frames = []
    expect_size = None
    for img in images:
        ctype = img.content_type or ""
        if not ctype.startswith("image/"):
            _err(400, f"文件 {img.filename} 不是图片类型（{ctype or '未知'}）。")
        data = await img.read()
        if len(data) > MAX_IMAGE_BYTES:
            _err(400, f"图片 {img.filename} 超过大小限制。")
        w, h = _decode_and_check_image(data, img.filename, expect_size)
        expect_size = (w, h)
        frames.append((ctype, data))

    raw = await detections.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _err(400, f"检测 JSON 解析失败：{exc}")
    dets = _validate_detections(payload, len(frames), expect_size)
    return _payload(_new_session(frames, dets))


@app.post("/api/example")
async def load_example():
    """创建内置连拍示例会话。"""
    frames = build_example_frames()
    sess = _new_session(frames, build_example_detections())
    out = _payload(sess)
    out["description"] = EXAMPLE_DESCRIPTION
    return out


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    return _payload(_get_session(sid))


@app.get("/api/sessions/{sid}/frames/{idx}")
async def get_frame(sid: str, idx: int):
    sess = _get_session(sid)
    if not (0 <= idx < len(sess["frames"])):
        _err(404, "帧不存在。")
    ctype, data = sess["frames"][idx]
    return Response(content=data, media_type=ctype)


@app.get("/api/sessions/{sid}/detections.json")
async def download_detections(sid: str):
    sess = _get_session(sid)
    return JSONResponse({"detections": sess["detections"]})


# ---------------------------------------------------------------- 分析任务

def _run_job(job_id: str, snapshot: dict):
    try:
        report = analyze_burst(snapshot["images"], snapshot["detections"])
        JOBS[job_id].update(status="done", result=report_to_dict(report))
    except ValueError as exc:
        JOBS[job_id].update(
            status="done",
            result={"status": "failed", "message": f"图像分析失败：{exc}"},
        )
    except Exception as exc:  # noqa: BLE001 - 任何分析异常都不能拖垮服务
        JOBS[job_id].update(status="error", error=f"分析内部错误：{exc}")


@app.post("/api/sessions/{sid}/analyze")
async def start_analyze(sid: str):
    """启动异步重复帧分析。任务记录当前输入版本，前端据此丢弃迟到结果。"""
    sess = _get_session(sid)
    job_id = uuid.uuid4().hex[:12]
    snapshot = {
        "images": [b for _, b in sess["frames"]],
        "detections": copy.deepcopy(sess["detections"]),
    }
    JOBS[job_id] = {
        "status": "running",
        "version": sess["version"],
        "session_id": sid,
        "result": None,
        "error": None,
    }
    loop = asyncio.get_running_loop()
    loop.run_in_executor(EXECUTOR, _run_job, job_id, snapshot)
    return {"job_id": job_id, "version": sess["version"]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        _err(404, "分析任务不存在。")
    return job


# ---------------------------------------------------------------- 页面

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
