"""API 端到端测试：会话创建、校验、判断/参数版本、异步求解与迟到结果作废，
以及相邻帧背景配准接口。"""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.registration import REASONS

client = TestClient(app)


def png_bytes(color=(200, 30, 30)):
    img = Image.new("RGB", (64, 48), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def wait_job(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("求解任务超时未完成")


def wait_reg(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/registration-jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("配准任务超时未完成")


def make_session(n_imgs=3, dets=None):
    if dets is None:
        dets = [
            {"id": "a", "frame": 0, "rect": [1, 1, 8, 8], "vector": [1, 0]},
            {"id": "b", "frame": 1, "rect": [11, 1, 8, 8], "vector": [1, 0]},
        ]
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(n_imgs)]
    files.append(("detections", ("det.json", json.dumps({"detections": dets}), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- 会话与校验

def test_create_session_ok():
    p = make_session()
    assert p["n_frames"] == 3 and len(p["detections"]) == 2
    assert p["version"] == 1
    img = client.get(f"/api/sessions/{p['session_id']}/frames/0")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_too_many_images_rejected():
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(11)]
    files.append(("detections", ("d.json", json.dumps([]), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "10" in r.json()["detail"]


def test_too_many_detections_rejected():
    dets = [{"id": f"b{i}", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1]} for i in range(31)]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "30" in r.json()["detail"]


@pytest.mark.parametrize("det,needle", [
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [0, 0]}, "非零"),
    ({"id": "a", "frame": 5, "rect": [0, 0, 4, 4], "vector": [1]}, "帧号"),
    ({"id": "a", "frame": 0, "rect": [0, 0, -4, 4], "vector": [1]}, "宽"),
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, float("nan")]}, "非法"),
])
def test_bad_detection_rejected(det, needle):
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps([det]), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and needle in r.json()["detail"]


def test_vector_dim_mismatch_rejected():
    dets = [
        {"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, 0]},
        {"id": "b", "frame": 0, "rect": [8, 0, 4, 4], "vector": [1, 0, 0]},
    ]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "维度" in r.json()["detail"]


# ---------------------------------------------------------------- 示例与求解

def test_example_solves_to_two_animals():
    p = client.post("/api/example").json()
    assert p["n_frames"] == 6 and len(p["detections"]) == 11
    job = client.post(f"/api/sessions/{p['session_id']}/solve").json()
    done = wait_job(job["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["feasible"] and res["animal_count"] == 2 and res["n_trajectories"] == 2
    assert res["total_cost"] == pytest.approx(res["assoc_cost"] + res["penalty_cost"])


def test_conflicting_judgments_report_infeasible_without_count():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # A0 与 B0 同处第 0 帧，标记同只 → 无解
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "A0", "b": "B0", "type": "same"}]})
    assert r.status_code == 200
    job = client.post(f"/api/sessions/{sid}/solve").json()
    done = wait_job(job["job_id"])
    res = done["result"]
    assert res["feasible"] is False
    assert "animal_count" not in res
    assert "无解" in res["message"]


def test_same_judgment_changes_result():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # 把 B0 与 A1 判为同只（它们帧差 1），全局解必须让二者同轨
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "B0", "b": "A1", "type": "same"}]})
    done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    res = done["result"]
    assert res["feasible"]
    for t in res["trajectories"]:
        ids = {b["id"] for b in t["boxes"]}
        assert not ({"B0", "A1"} <= ids) or True  # 占位，下面统一检查
    same_traj = [t for t in res["trajectories"]
                 if {"B0", "A1"} <= {b["id"] for b in t["boxes"]}]
    assert len(same_traj) == 1


# ---------------------------------------------------------------- 版本与迟到结果

def test_version_bumps_and_stale_job_detectable():
    p = make_session()
    sid = p["session_id"]
    assert p["version"] == 1

    r = client.put(f"/api/sessions/{sid}/params",
                   json={"w_dist": 2.0, "w_app": 50.0, "penalty": 100.0})
    assert r.status_code == 200 and r.json()["version"] == 2

    job = client.post(f"/api/sessions/{sid}/solve").json()
    assert job["version"] == 2  # 任务记录启动时版本

    # 求解期间继续修改 → 版本再 +1，旧任务结果即为“迟到结果”
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "a", "b": "b", "type": "different"}]})
    assert r.json()["version"] == 3

    done = wait_job(job["job_id"])
    assert done["version"] == 2  # 结果仍携带旧版本号
    current = client.get(f"/api/sessions/{sid}").json()
    assert current["version"] == 3
    assert done["version"] != current["version"]  # 前端据此丢弃，不覆盖新输入


def test_params_validation():
    p = make_session()
    r = client.put(f"/api/sessions/{p['session_id']}/params",
                   json={"w_dist": -1, "w_app": 1, "penalty": 1})
    assert r.status_code == 422


def test_unknown_session_404():
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/solve").status_code == 404


def test_healthz():
    assert client.get("/healthz").json() == {"ok": True}


# ---------------------------------------------------------------- 背景配准

def _shifted_pngs(dx, dy, size=192):
    rng = np.random.RandomState(7)
    base = rng.randint(0, 255, size=(size, size)).astype(np.uint8)
    for (x, y) in [(30, 30), (90, 40), (150, 50), (40, 110), (120, 120),
                   (160, 140), (50, 160), (140, 165)]:
        base[y - 10:y + 10, x - 10:x + 10] = rng.randint(0, 255, size=(20, 20))
    b = np.array(
        Image.fromarray(base).transform(
            (size, size), Image.AFFINE, (1, 0, -dx, 0, 1, -dy)
        )
    )

    def enc(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    return enc(base), enc(b)


def test_registration_endpoint_reliable_shift_and_overlay():
    img0, img1 = _shifted_pngs(6, -4)
    dets = [{"id": "a", "frame": 0, "rect": [1, 1, 8, 8], "vector": [1, 0]}]
    files = [
        ("images", ("f0.png", img0, "image/png")),
        ("images", ("f1.png", img1, "image/png")),
        ("images", ("f2.png", img0, "image/png")),
        ("detections", ("det.json", json.dumps({"detections": dets}), "application/json")),
    ]
    p = client.post("/api/sessions", files=files).json()
    sid = p["session_id"]

    r = client.post(f"/api/sessions/{sid}/registration", json={"frame": 0})
    assert r.status_code == 200
    job = wait_reg(r.json()["job_id"])
    assert job["status"] == "done"
    result = job["result"]
    assert result["reliable"] and result["dx"] == 6 and result["dy"] == -4
    assert result["overlay"] is not None
    assert result["overlay"].startswith("data:image/png;base64,")
    assert result["frame"] == 0 and result["version"] == p["version"]


def test_registration_last_frame_rejected():
    p = make_session(n_imgs=2)
    r = client.post(f"/api/sessions/{p['session_id']}/registration", json={"frame": 1})
    assert r.status_code == 400 and "最后一帧" in r.json()["detail"]


def test_registration_undecodable_image_reported_not_crashed():
    good = png_bytes()
    dets = [{"id": "a", "frame": 0, "rect": [1, 1, 4, 4], "vector": [1]}]
    files = [
        ("images", ("f0.png", good, "image/png")),
        ("images", ("f1.png", b"not-an-image", "image/png")),
        ("detections", ("det.json", json.dumps({"detections": dets}), "application/json")),
    ]
    p = client.post("/api/sessions", files=files).json()
    sid = p["session_id"]
    r = client.post(f"/api/sessions/{sid}/registration", json={"frame": 0})
    job = wait_reg(r.json()["job_id"])
    result = job["result"]
    assert not result["reliable"]
    assert result["reason_code"] == "decode_error"
    assert result["reason"] == REASONS["decode_error"]
    assert result["overlay"] is None and result["dx"] is None and result["dy"] is None


def test_registration_flat_image_unreliable_without_overlay():
    flat = png_bytes((120, 120, 120))
    dets = [{"id": "a", "frame": 0, "rect": [1, 1, 4, 4], "vector": [1]}]
    files = [
        ("images", ("f0.png", flat, "image/png")),
        ("images", ("f1.png", flat, "image/png")),
        ("detections", ("det.json", json.dumps({"detections": dets}), "application/json")),
    ]
    p = client.post("/api/sessions", files=files).json()
    r = client.post(f"/api/sessions/{p['session_id']}/registration", json={"frame": 0})
    job = wait_reg(r.json()["job_id"])
    result = job["result"]
    assert not result["reliable"]
    assert result["reason_code"] == "insufficient_texture"
    assert result["overlay"] is None and result["dx"] is None and result["dy"] is None


def test_registration_unknown_job_and_session_404():
    assert client.get("/api/registration-jobs/nope").status_code == 404
    assert client.post("/api/sessions/nope/registration",
                       json={"frame": 0}).status_code == 404


def test_registration_version_bumps_with_edits():
    img0, img1 = _shifted_pngs(3, 2)
    dets = [{"id": "a", "frame": 0, "rect": [1, 1, 8, 8], "vector": [1, 0]}]
    files = [
        ("images", ("f0.png", img0, "image/png")),
        ("images", ("f1.png", img1, "image/png")),
        ("detections", ("det.json", json.dumps({"detections": dets}), "application/json")),
    ]
    p = client.post("/api/sessions", files=files).json()
    sid = p["session_id"]
    r = client.post(f"/api/sessions/{sid}/registration", json={"frame": 0})
    assert r.json()["version"] == 1
    client.put(f"/api/sessions/{sid}/params",
               json={"w_dist": 2.0, "w_app": 1.0, "penalty": 1.0})
    r2 = client.post(f"/api/sessions/{sid}/registration", json={"frame": 0})
    assert r2.json()["version"] == 2  # 新任务记录新版本，前端据此丢弃迟到旧结果
