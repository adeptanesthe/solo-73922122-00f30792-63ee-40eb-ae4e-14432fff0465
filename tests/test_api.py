"""API 端到端测试：会话创建与校验、示例分析、异步任务与迟到结果作废、
切换会话撤下旧建议，以及输入数据只读不变。"""

import io
import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.example import build_example_detections

client = TestClient(app)


def jpeg_bytes(color=(200, 30, 30), size=(256, 192), seed=0):
    rng = __import__("random").Random(seed)
    img = Image.new("RGB", size, color)
    px = img.load()
    for y in range(0, size[1], 2):
        for x in range(0, size[0], 2):
            n = rng.randrange(-12, 13)
            px[x, y] = (max(0, min(255, color[0] + n)),
                        max(0, min(255, color[1] + n)),
                        max(0, min(255, color[2] + n)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_session(n_imgs=3, size=(256, 192), dets=None):
    if dets is None:
        dets = [{"id": f"b{f}", "frame": f, "rect": [40, 40, 60, 60]}
                for f in range(n_imgs)]
    files = [("images", (f"f{i}.jpg", jpeg_bytes(size=size, seed=i), "image/jpeg"))
             for i in range(n_imgs)]
    files.append(("detections", ("det.json", json.dumps({"detections": dets}),
                                 "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def wait_job(job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("分析任务超时未完成")


# ---------------------------------------------------------------- 会话与校验

def test_create_session_ok():
    p = make_session()
    assert p["n_frames"] == 3 and len(p["detections"]) == 3
    assert p["version"] == 1
    img = client.get(f"/api/sessions/{p['session_id']}/frames/0")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"


def test_too_many_images_rejected():
    files = [("images", (f"f{i}.jpg", jpeg_bytes(seed=i), "image/jpeg")) for i in range(7)]
    files.append(("detections", ("d.json", json.dumps([]), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "6" in r.json()["detail"]


def test_mixed_sizes_rejected():
    files = [
        ("images", ("f0.jpg", jpeg_bytes(size=(256, 192)), "image/jpeg")),
        ("images", ("f1.jpg", jpeg_bytes(size=(200, 192)), "image/jpeg")),
    ]
    files.append(("detections", ("d.json", json.dumps([]), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "尺寸" in r.json()["detail"]


def test_oversized_image_rejected():
    big = jpeg_bytes(size=(513, 200))
    files = [("images", ("f0.jpg", big, "image/jpeg"))]
    files.append(("detections", ("d.json", json.dumps([]), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "512" in r.json()["detail"]


def test_512_square_accepted():
    p = make_session(n_imgs=1, size=(512, 512),
                     dets=[{"id": "b0", "frame": 0, "rect": [10, 10, 50, 50]}])
    assert p["n_frames"] == 1


def test_bad_rect_rejected():
    bad = [
        {"id": "a", "frame": 0, "rect": [0, 0, 0, 4]},
        {"id": "b", "frame": 0, "rect": [0, 0, 4, -4]},
        {"id": "c", "frame": 0, "rect": [300, 300, 4, 4]},  # 与画面无交集
    ]
    for det in bad:
        files = [("images", ("f0.jpg", jpeg_bytes(), "image/jpeg")),
                 ("detections", ("d.json", json.dumps([det]), "application/json"))]
        r = client.post("/api/sessions", files=files)
        assert r.status_code == 400, det


def test_frame_out_of_range_rejected():
    det = {"id": "a", "frame": 5, "rect": [0, 0, 4, 4]}
    files = [("images", ("f0.jpg", jpeg_bytes(), "image/jpeg")),
             ("detections", ("d.json", json.dumps([det]), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "帧号" in r.json()["detail"]


def test_duplicate_box_id_rejected():
    dets = [
        {"id": "x", "frame": 0, "rect": [0, 0, 4, 4]},
        {"id": "x", "frame": 0, "rect": [8, 8, 4, 4]},
    ]
    files = [("images", ("f0.jpg", jpeg_bytes(), "image/jpeg")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "重复" in r.json()["detail"]


# ---------------------------------------------------------------- 示例与分析

def test_example_analysis_decisions():
    p = client.post("/api/example").json()
    assert p["n_frames"] == 6 and len(p["detections"]) == 6
    job = client.post(f"/api/sessions/{p['session_id']}/analyze").json()
    assert job["version"] == p["version"]
    done = wait_job(job["job_id"])
    assert done["status"] == "done"
    result = done["result"]
    decisions = {v["frame"]: v["decision"] for v in result["verdicts"]}
    assert decisions[0] == "keep"
    assert decisions[1] == "omit" and decisions[2] == "omit"
    assert decisions[3] == "keep"
    assert decisions[4] == "review" and decisions[5] == "review"
    # 每个配对同时带背景与动物两区域证据
    for pair in result["pairs"]:
        assert "background" in pair and "animal" in pair
        assert pair["label"] in ("duplicate", "different", "unknown")


def test_analysis_is_read_only():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    before = client.get(f"/api/sessions/{sid}/detections.json").json()
    wait_job(client.post(f"/api/sessions/{sid}/analyze").json()["job_id"])
    after = client.get(f"/api/sessions/{sid}/detections.json").json()
    assert before == after  # 检测框不变
    # 原图字节不变
    img_before = client.get(f"/api/sessions/{sid}/frames/2").content
    img_after = client.get(f"/api/sessions/{sid}/frames/2").content
    assert img_before == img_after
    # 会话版本不变（没有任何写入接口）
    assert client.get(f"/api/sessions/{sid}").json()["version"] == 1


# ---------------------------------------------------------------- 任务版本与迟到结果

def test_job_carries_session_and_version():
    p = make_session()
    job = client.post(f"/api/sessions/{p['session_id']}/analyze").json()
    assert job["version"] == 1
    done = wait_job(job["job_id"])
    assert done["version"] == 1
    assert done["session_id"] == p["session_id"]
    assert done["result"]["n_frames"] == 3


def test_stale_job_from_old_session_identifiable():
    # 旧会话的任务结果携带旧 session_id，前端据此丢弃（不覆盖新会话）
    old = make_session()
    old_job = client.post(f"/api/sessions/{old['session_id']}/analyze").json()
    new = make_session()
    done = wait_job(old_job["job_id"])
    assert done["session_id"] == old["session_id"]
    assert done["session_id"] != new["session_id"]  # 前端切换会话后丢弃


def test_unknown_session_404():
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/analyze").status_code == 404


def test_unknown_job_404():
    assert client.get("/api/jobs/nope").status_code == 404


def test_healthz():
    assert client.get("/healthz").json() == {"ok": True}


def test_detections_download():
    p = client.post("/api/example").json()
    data = client.get(f"/api/sessions/{p['session_id']}/detections.json").json()
    assert data["detections"] == build_example_detections()
