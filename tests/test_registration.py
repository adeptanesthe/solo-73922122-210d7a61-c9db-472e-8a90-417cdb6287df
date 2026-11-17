"""背景配准单元测试：可靠平移、检测框排除、纹理不足、歧义/分歧、
解码与尺寸限制，以及叠图渲染。"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.registration import (
    MAX_DIM,
    build_background_mask,
    register_pair,
    render_overlay,
)

W, H = 256, 256


# ---------------------------------------------------------------- 造图工具

def png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def textured_bg(seed=11):
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 255, size=(H, W), dtype=np.uint8)
    # 叠加几块高对比地物，保证各网格块都有结构
    img = base.copy()
    for (x, y) in [(30, 30), (90, 40), (160, 45), (210, 70),
                   (40, 120), (120, 130), (190, 150),
                   (50, 200), (140, 210), (215, 205)]:
        img[y - 8:y + 8, x - 8:x + 8] = rng.randint(0, 255, size=(16, 16))
    return img


def shift_copy(arr, dx, dy):
    """B[Y,X] = A[Y-dy, X-dx]：特征整体平移 (dx, dy)，边缘填黑。"""
    return np.array(
        Image.fromarray(arr).transform(
            arr.shape[::-1], Image.AFFINE, (1, 0, -dx, 0, 1, -dy)
        )
    )


def fill_rect(arr, x, y, w, h, value=255):
    arr[y:y + h, x:x + w] = value


# ---------------------------------------------------------------- 可靠场景

def test_reliable_shift_detected():
    bg = textured_bg()
    a = bg.copy()
    b = shift_copy(bg, 5, -3)
    # 两帧各有一只“动物”（亮块），位置不同，用检测框排除
    fill_rect(a, 20, 20, 24, 24)
    fill_rect(b, 90, 110, 24, 24)
    dets_a = [{"id": "a0", "frame": 0, "rect": [20, 20, 24, 24], "vector": [1, 0]}]
    dets_b = [{"id": "b1", "frame": 1, "rect": [90, 110, 24, 24], "vector": [1, 0]}]

    r = register_pair(png(a), png(b), dets_a, dets_b)
    assert r["reliable"], r
    assert r["dx"] == 5 and r["dy"] == -3
    assert r["n_regions"] >= 3


def test_zero_shift():
    bg = textured_bg()
    r = register_pair(png(bg), png(bg.copy()), [], [])
    assert r["reliable"] and r["dx"] == 0 and r["dy"] == 0


@pytest.mark.parametrize("dx,dy", [(16, 16), (-16, -16), (16, -16)])
def test_shift_at_search_boundary(dx, dy):
    bg = textured_bg()
    r = register_pair(png(bg), png(shift_copy(bg, dx, dy)), [], [])
    assert r["reliable"], r
    assert r["dx"] == dx and r["dy"] == dy


def test_shift_beyond_window_not_reported():
    bg = textured_bg()
    # 真实平移 17 像素，超出 ±16 搜索窗：不得报告 (17,17)
    r = register_pair(png(bg), png(shift_copy(bg, 17, 17)), [], [])
    assert not r["reliable"]
    assert r["dx"] is None and r["dy"] is None


def test_detection_boxes_excluded_from_matching():
    bg = textured_bg()
    a = bg.copy()
    b = bg.copy()  # 背景完全静止
    # 两个大块“动物”在帧间移动；不排除它们会把相关峰拉偏
    fill_rect(a, 30, 30, 60, 60, value=250)
    fill_rect(b, 150, 150, 60, 60, value=250)
    dets_a = [{"id": "a0", "frame": 0, "rect": [30, 30, 60, 60], "vector": [1, 0]}]
    dets_b = [{"id": "b1", "frame": 1, "rect": [150, 150, 60, 60], "vector": [1, 0]}]

    mask = build_background_mask((H, W), dets_a) & build_background_mask((H, W), dets_b)
    assert not mask[40, 40] and not mask[160, 160]  # 框内像素被屏蔽
    assert mask[120, 120]                            # 背景保留

    r = register_pair(png(a), png(b), dets_a, dets_b)
    assert r["reliable"], r
    assert r["dx"] == 0 and r["dy"] == 0


# ---------------------------------------------------------------- 不可靠场景

def test_flat_image_insufficient_texture():
    flat = np.full((H, W), 100, dtype=np.uint8)
    r = register_pair(png(flat), png(flat.copy()), [], [])
    assert not r["reliable"]
    assert r["reason_code"] == "insufficient_texture"
    assert r["dx"] is None and r["dy"] is None


def test_periodic_texture_is_ambiguous_or_unreliable():
    # 周期棋盘格：多个位移具有近乎相同的相关分，不得给出确定平移
    tile = 6
    yy, xx = np.mgrid[0:H, 0:W]
    checker = (((xx // tile) + (yy // tile)) % 2 * 200).astype(np.uint8)
    r = register_pair(png(checker), png(shift_copy(checker, 3, 0)), [], [])
    assert not r["reliable"]
    assert r["reason_code"] in {"ambiguous", "insufficient_texture"}


def test_regions_disagree_rejected():
    bg = textured_bg()
    a = bg.copy()
    b = bg.copy()
    # 左半静止、右半整体平移 8 像素：不同区域指向不同位移
    b[:, W // 2:] = shift_copy(bg, 8, 0)[:, W // 2:]
    r = register_pair(png(a), png(b), [], [])
    assert not r["reliable"]
    assert r["reason_code"] in {"regions_disagree", "ambiguous"}
    assert r["dx"] is None


def test_undecodable_image():
    bg = textured_bg()
    r = register_pair(b"this is definitely not an image", png(bg), [], [])
    assert not r["reliable"] and r["reason_code"] == "decode_error"
    r2 = register_pair(png(bg), b"broken bytes", [], [])
    assert not r2["reliable"] and r2["reason_code"] == "decode_error"


def test_size_mismatch_rejected():
    a = np.full((200, 200), 0, np.uint8)
    b = np.full((201, 200), 0, np.uint8)
    r = register_pair(png(a), png(b), [], [])
    assert not r["reliable"] and r["reason_code"] == "size_mismatch"


def test_too_large_rejected():
    big = np.zeros((MAX_DIM + 1, MAX_DIM), dtype=np.uint8)
    r = register_pair(png(big), png(big.copy()), [], [])
    assert not r["reliable"] and r["reason_code"] == "too_large"

    edge = np.zeros((MAX_DIM, MAX_DIM), dtype=np.uint8)  # 恰好 512 不拒绝
    r2 = register_pair(png(edge), png(edge.copy()), [], [])
    assert r2["reason_code"] != "too_large"


def test_detection_boxes_cover_most_of_image():
    bg = textured_bg()
    dets = [{"id": "x", "frame": 0, "rect": [0, 0, W, H], "vector": [1, 0]}]
    r = register_pair(png(bg), png(bg.copy()), dets, dets)
    assert not r["reliable"]
    assert r["reason_code"] == "insufficient_background"


# ---------------------------------------------------------------- 叠图

def test_overlay_rendering():
    bg = textured_bg()
    a, b = bg.copy(), shift_copy(bg, 4, 2)
    overlay = render_overlay(png(a), png(b), 4, 2)
    img = Image.open(io.BytesIO(overlay))
    assert img.format == "PNG"
    assert img.size == (W, H)
    arr = np.asarray(img.convert("RGB"), dtype=float)
    # 与逐像素参考实现对照（独立于生产代码的向量化写法）
    ref = a.astype(float).copy()
    for y in range(H):
        for x in range(W):
            by, bx = y - 2, x - 4
            if 0 <= by < H and 0 <= bx < W:
                ref[y, x] = 0.5 * b[by, bx] + 0.5 * a[y, x]
    assert np.abs(arr[..., 0] - ref).max() <= 1.0
    # 越界区保留当前帧
    assert np.array_equal(arr[0, :, 0], a[0, :].astype(float))
