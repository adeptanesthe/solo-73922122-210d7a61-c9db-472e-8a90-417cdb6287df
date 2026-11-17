"""相邻帧背景配准：估计 ±16 像素内的整数平移并生成对齐叠图。

适用范围与判定规则（与需求逐条对应）
------------------------------------
- 仅处理**尺寸相同且宽高均不超过 512** 的图片；解码失败、尺寸不一致或
  超出尺寸上限一律判为无法配准。
- 两帧**检测框内的像素不参与匹配**：把两帧的检测矩形（各向外扩 6 像素，
  容忍框线误差）合并为掩膜，块内被掩膜覆盖过多的位置直接弃用。
- 画面均分为 4×4 网格块，每个背景块在 ±16 像素搜索窗内做**掩膜归一化
  互相关（NCC）**，由 ``scipy.signal.fftconvolve`` 加速；块内纹理不足、
  相关峰过低或最优/次优峰差距过小（峰不锐）的块记为不可信。
- 至少 **3 个分散**（覆盖 ≥2 个网格行且 ≥2 个网格列）的可信块指向同一
  位移才接受；若另有可信块指向不同位移，或全局最优/次优位移的平均相关
  分差距过小（可信度相近的不同位移），判为无法可靠配准。
- 配准成功时返回整数位移 ``(dx, dy)``（把下一帧平移该向量即可与当前帧
  背景对齐）以及半透明对齐叠图；失败时不给定位移、不生成叠图。
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve

# ---------------------------------------------------------------- 常量
MAX_DIM = 512                 # 允许配准的最大边长
MAX_SHIFT = 16                # 横、纵搜索范围 ±16 像素
GRID_N = 4                    # 画面划分为 4×4 个候选背景块
MIN_REGIONS = 3               # 至少 3 个可信区域
BOX_DILATION = 6              # 检测框外扩像素（框线/边缘误差余量）
BLOCK_VALID_RATIO = 0.6       # 块内未被检测框掩膜覆盖的最小比例
NCC_VALID_RATIO = 0.5         # 单个候选位移所需有效像素占块面积比例
NCC_MIN_PEAK = 0.5            # 可信块相关峰下限
NCC_PEAK_MARGIN = 0.10        # 块内最优峰与次优峰的最小差距
GLOBAL_SCORE_GAP = 0.06       # 全局最优/次优位移平均相关分最小差距
MIN_TEXTURE_STD = 5.0         # 块内背景灰度标准差下限（0~255 尺度）
MIN_BACKGROUND_RATIO = 0.25   # 全图背景像素最小占比
OVERLAY_ALPHA = 0.5           # 叠图中下一帧的不透明度

# 失败原因（code 供程序断言，message 供界面展示）
REASONS = {
    "decode_error": "图片无法解码，不进行背景配准。",
    "size_mismatch": "两帧图片尺寸不同，无法配准。",
    "too_large": "图片边长超过 512 像素，按要求不做配准。",
    "insufficient_background": "检测框外可用于匹配的背景像素过少，无法配准。",
    "insufficient_texture": "纹理充足且可信的分散背景区域不足 3 处，无法可靠配准。",
    "regions_disagree": "不同背景区域指向了不同位移，无法可靠配准。",
    "ambiguous": "存在可信度相近的不同位移，无法可靠配准。",
}


class RegistrationError(Exception):
    """配准前置条件不满足（解码、尺寸等），携带原因码。"""

    def __init__(self, code: str):
        super().__init__(REASONS[code])
        self.code = code


# ---------------------------------------------------------------- 解码与掩膜

def decode_gray(data: bytes) -> np.ndarray:
    """解码图片为 float64 灰度数组；失败抛 RegistrationError('decode_error')。"""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # 立即触发解码，延迟错误在这里暴露
        gray = img.convert("L")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RegistrationError("decode_error") from exc
    return np.asarray(gray, dtype=np.float64)


def build_background_mask(shape: tuple[int, int], detections: list[dict]) -> np.ndarray:
    """生成背景掩膜：检测框（外扩 BOX_DILATION 像素）内为 False，其余为 True。"""
    h, w = shape
    mask = np.ones((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    for d in detections:
        x, y, bw, bh = (float(v) for v in d["rect"])
        x0 = max(0, int(np.floor(x)) - BOX_DILATION)
        y0 = max(0, int(np.floor(y)) - BOX_DILATION)
        x1 = min(w, int(np.ceil(x + bw)) + BOX_DILATION)
        y1 = min(h, int(np.ceil(y + bh)) + BOX_DILATION)
        if x1 > x0 and y1 > y0:
            blocked[y0:y1, x0:x1] = True
    if blocked.any():
        blocked = binary_dilation(blocked, iterations=BOX_DILATION)
    return ~blocked


# ---------------------------------------------------------------- 掩膜 NCC

def _xcorr_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """二维互相关的 valid 部分：输出 (i,j) = Σ image[i+.,j+.] * kernel。

    用 FFT 卷积实现（卷积核先翻转），比逐位移循环快两个数量级。
    """
    kh, kw = kernel.shape
    full = fftconvolve(image, kernel[::-1, ::-1], mode="full")
    oh = image.shape[0] - kh + 1
    ow = image.shape[1] - kw + 1
    return full[kh - 1:kh - 1 + oh, kw - 1:kw - 1 + ow]


def _block_ncc(gray_a: np.ndarray, gray_b: np.ndarray,
               mask: np.ndarray,
               r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """计算一个块在全部可行位移上的掩膜 NCC，形状 (2S+1, 2S+1)，不可行为 NaN。

    mask 为两帧检测框合并后的联合背景掩膜（True=可参与匹配的背景）。
    """
    s = MAX_SHIFT
    h, w = gray_a.shape
    t = gray_a[r0:r1, c0:c1]
    mt = mask[r0:r1, c0:c1].astype(np.float64)
    bh, bw = t.shape

    r_start = max(0, r0 - s)
    c_start = max(0, c0 - s)
    r_end = min(h, r1 + s)
    c_end = min(w, c1 + s)
    sb = gray_b[r_start:r_end, c_start:c_end]
    mb = mask[r_start:r_end, c_start:c_end].astype(np.float64)

    # 各统计量的滑动求和（掩膜内像素）
    n = _xcorr_valid(mb, mt)
    sum_t = _xcorr_valid(mb, mt * t)
    sum_t2 = _xcorr_valid(mb, mt * t * t)
    sum_s = _xcorr_valid(mb * sb, mt)
    sum_s2 = _xcorr_valid(mb * sb * sb, mt)
    sum_ts = _xcorr_valid(mb * sb, mt * t)

    scores = np.full((2 * s + 1, 2 * s + 1), np.nan, dtype=np.float64)
    min_pixels = NCC_VALID_RATIO * bh * bw
    oh, ow = n.shape
    for u in range(oh):
        for v in range(ow):
            nv = n[u, v]
            if nv < min_pixels:
                continue
            mt_bar = sum_t[u, v] / nv
            ms_bar = sum_s[u, v] / nv
            var_t = sum_t2[u, v] / nv - mt_bar * mt_bar
            var_s = sum_s2[u, v] / nv - ms_bar * ms_bar
            cov = sum_ts[u, v] / nv - mt_bar * ms_bar
            denom = np.sqrt(max(var_t, 0.0) * max(var_s, 0.0))
            if denom <= 1e-12:
                continue
            dy = (r_start + u) - r0
            dx = (c_start + v) - c0
            scores[dy + s, dx + s] = cov / denom
    return scores


def _grid_edges(length: int) -> list[int]:
    return [round(i * length / GRID_N) for i in range(GRID_N + 1)]


def estimate_blocks(gray_a: np.ndarray, gray_b: np.ndarray,
                    mask: np.ndarray) -> list[dict]:
    """逐块估计位移，返回每个网格块的判定明细。"""
    h, w = gray_a.shape
    row_edges = _grid_edges(h)
    col_edges = _grid_edges(w)
    blocks = []
    s = MAX_SHIFT
    for ri in range(GRID_N):
        r0, r1 = row_edges[ri], row_edges[ri + 1]
        for ci in range(GRID_N):
            c0, c1 = col_edges[ci], col_edges[ci + 1]
            bh, bw = r1 - r0, c1 - c0
            mt = mask[r0:r1, c0:c1]
            block = {
                "row": ri, "col": ci,
                "confident": False, "dx": None, "dy": None, "score": None,
                "note": None,
            }
            if mt.sum() < BLOCK_VALID_RATIO * bh * bw:
                block["note"] = "块内背景像素过少（检测框覆盖）"
                blocks.append(block)
                continue
            vals = gray_a[r0:r1, c0:c1][mt]
            if vals.std() < MIN_TEXTURE_STD:
                block["note"] = "块内纹理不足"
                blocks.append(block)
                continue

            scores = _block_ncc(gray_a, gray_b, mask, r0, r1, c0, c1)
            valid = np.isfinite(scores)
            if not valid.any():
                block["note"] = "无满足有效像素要求的候选位移"
                blocks.append(block)
                continue

            # 全局峰（无论是否锐，供全局歧义判定使用）
            best_flat = int(np.nanargmax(scores))
            by, bx = np.unravel_index(best_flat, scores.shape)
            best_score = float(scores[by, bx])
            block["best_dx"], block["best_dy"] = bx - s, by - s
            block["best_score"] = best_score

            if best_score < NCC_MIN_PEAK:
                block["note"] = "相关峰过低"
                blocks.append(block)
                continue
            # 次优峰：排除最优峰周围 3×3 邻域，避免同一峰的旁瓣压低裕度
            masked = scores.copy()
            y0, y1 = max(0, by - 1), min(scores.shape[0], by + 2)
            x0, x1 = max(0, bx - 1), min(scores.shape[1], bx + 2)
            masked[y0:y1, x0:x1] = np.nan
            if np.isfinite(masked).any():
                second = float(np.nanmax(masked))
            else:
                second = -1.0
            if best_score - second < NCC_PEAK_MARGIN:
                block["note"] = "峰不锐（最优与次优位移可信度过近）"
                blocks.append(block)
                continue

            block.update(confident=True, dx=bx - s, dy=by - s,
                         score=best_score)
            blocks.append(block)
    return blocks


# ---------------------------------------------------------------- 主入口

def register_pair(img_a: bytes, img_b: bytes,
                  dets_a: list[dict], dets_b: list[dict]) -> dict:
    """对相邻两帧做背景配准。

    返回字典始终含 ``reliable``；失败时含 ``reason_code``/``reason``，
    且 ``dx``/``dy`` 为 None；成功时含整数 ``dx``/``dy`` 与区域统计。
    """
    try:
        gray_a = decode_gray(img_a)
        gray_b = decode_gray(img_b)
    except RegistrationError as exc:
        return _fail(exc.code, 0, 0)

    if gray_a.shape != gray_b.shape:
        return _fail("size_mismatch", 0, 0)
    h, w = gray_a.shape
    if h > MAX_DIM or w > MAX_DIM:
        return _fail("too_large", GRID_N * GRID_N, GRID_N * GRID_N)

    mask_a = build_background_mask(gray_a.shape, dets_a)
    mask_b = build_background_mask(gray_b.shape, dets_b)
    # 任一帧落在检测框内的像素都不参与匹配
    mask = mask_a & mask_b
    if mask.sum() < MIN_BACKGROUND_RATIO * h * w:
        return _fail("insufficient_background", GRID_N * GRID_N, GRID_N * GRID_N)

    # 两帧使用同一张联合掩膜：分别在 A、B 上屏蔽双方的检测框
    blocks = estimate_blocks(gray_a, gray_b, mask)
    confident = [b for b in blocks if b["confident"]]

    if len(confident) < MIN_REGIONS:
        return _fail("insufficient_texture", len(confident), len(blocks))

    shifts = {(b["dy"], b["dx"]) for b in confident}
    if len(shifts) > 1:
        return _fail("regions_disagree", len(confident), len(blocks))

    winner_dy, winner_dx = next(iter(shifts))
    rows = {b["row"] for b in confident}
    cols = {b["col"] for b in confident}
    if len(rows) < 2 or len(cols) < 2:
        return _fail("insufficient_texture", len(confident), len(blocks))

    # 全局歧义判定：所有能给出相关峰的块按位移汇总平均相关分，
    # 若次优位移与最优位移平均可信度过近，视为可信度相近的不同位移。
    per_shift: dict[tuple[int, int], list[float]] = {}
    for b in blocks:
        if "best_score" in b:
            per_shift.setdefault((b["best_dy"], b["best_dx"]), []).append(
                b["best_score"])
    if len(per_shift) >= 2:
        means = sorted(
            ((float(np.mean(v)), k) for k, v in per_shift.items()),
            reverse=True,
        )
        if means[0][0] - means[1][0] < GLOBAL_SCORE_GAP:
            return _fail("ambiguous", len(confident), len(blocks))

    return {
        "reliable": True,
        "dx": int(winner_dx),
        "dy": int(winner_dy),
        "reason_code": None,
        "reason": None,
        "n_regions": len(confident),
        "n_blocks": len(blocks),
    }


def _fail(code: str, n_regions: int, n_blocks: int) -> dict:
    return {
        "reliable": False,
        "dx": None,
        "dy": None,
        "reason_code": code,
        "reason": REASONS[code],
        "n_regions": n_regions,
        "n_blocks": n_blocks,
    }


# ---------------------------------------------------------------- 叠图

def render_overlay(img_a: bytes, img_b: bytes, dx: int, dy: int) -> bytes:
    """生成对齐后的半透明叠图 PNG（当前帧为底，下一帧平移后 50% 叠加）。

    覆盖区结果为 0.5·A + 0.5·B（B 先整体平移 (dx, dy) 对齐背景）；
    平移后越界、没有下一帧像素覆盖的位置保留当前帧 A 的原色。
    用 NumPy 显式处理坐标，避免 paste 对越界源图的隐式裁切。
    """
    a = np.asarray(Image.open(io.BytesIO(img_a)).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(io.BytesIO(img_b)).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        raise RegistrationError("size_mismatch")
    h, w = a.shape[:2]
    dx, dy = int(dx), int(dy)

    # B 平移后在 A 坐标系中的覆盖范围，与 B 自身像素的对应范围。
    # out[y,x] 覆盖 B[y-dy, x-dx]；交集满足 0≤y<h 且 0≤y-dy<h。
    ay0, ay1 = max(0, dy), min(h, h + dy)
    ax0, ax1 = max(0, dx), min(w, w + dx)
    by0, by1 = ay0 - dy, ay1 - dy
    bx0, bx1 = ax0 - dx, ax1 - dx
    assert (ay1 - ay0, ax1 - ax0) == (by1 - by0, bx1 - bx0)
    out = a.copy()
    if ay1 > ay0 and ax1 > ax0:
        out[ay0:ay1, ax0:ax1] = (
            OVERLAY_ALPHA * b[by0:by1, bx0:bx1]
            + (1.0 - OVERLAY_ALPHA) * a[ay0:ay1, ax0:ax1]
        )
    merged = Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), mode="RGB")

    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    return buf.getvalue()
