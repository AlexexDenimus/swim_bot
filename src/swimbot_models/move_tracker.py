import cv2
import numpy as np
from rtmlib import Body

# ---------------------------------------------------------------------------
# COCO-17 skeleton connections (for future skeleton overlay use)
# ---------------------------------------------------------------------------
COCO_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),  # torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs
]

# ---------------------------------------------------------------------------
# Per-joint confidence thresholds
# ---------------------------------------------------------------------------
_DEFAULT_THR = np.full(17, 0.35)
_DEFAULT_THR[7:11] = 0.15  # elbows + wrists
_DEFAULT_THR[11:13] = 0.18  # hips
_DEFAULT_THR[13:15] = 0.25  # knees
_DEFAULT_THR[15:17] = 0.30  # ankles

_UPDATE_THR = _DEFAULT_THR.copy()
_UPDATE_THR[11:13] = 0.15
_UPDATE_THR[13:15] = 0.22
_UPDATE_THR[15:17] = 0.28

# Threshold used by the DTW / similarity code (RTMPose scores are in [0, 1])
_DEFAULT_SCORE_THR = 0.35

# ---------------------------------------------------------------------------
# Per-joint one-frame motion caps (in body-scale units)
# ---------------------------------------------------------------------------
_MAX_JUMP_BY_JOINT = np.full(17, 1.10, dtype=np.float64)
_MAX_JUMP_BY_JOINT[0:5] = 0.45  # head / face
_MAX_JUMP_BY_JOINT[5:7] = 0.45  # shoulders
_MAX_JUMP_BY_JOINT[7:9] = 1.00  # elbows
_MAX_JUMP_BY_JOINT[9:11] = 1.40  # wrists
_MAX_JUMP_BY_JOINT[11:13] = 0.45  # hips
_MAX_JUMP_BY_JOINT[13:15] = 1.10  # knees
_MAX_JUMP_BY_JOINT[15:17] = 1.30  # ankles

# ---------------------------------------------------------------------------
# Bone definitions used by the temporal smoother
# ---------------------------------------------------------------------------
_TORSO_BONES = ((5, 6), (11, 12), (5, 11), (6, 12))

_LIMB_BONES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# ---------------------------------------------------------------------------
# RTMPose-L model URLs
# ---------------------------------------------------------------------------
_RTMPOSE_L_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip"
)
_YOLOX_M_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
)

_model = Body(
    det=_YOLOX_M_URL,
    det_input_size=(640, 640),
    pose=_RTMPOSE_L_URL,
    pose_input_size=(288, 384),
    backend="onnxruntime",
    device="cpu",
)


# ===========================================================================
# Tracker primitives (ported from move_tracker.ipynb)
# ===========================================================================


def _preprocess(frame: np.ndarray) -> np.ndarray:
    """CLAHE + unsharp mask to lift contrast and counter motion blur."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lc, a, b = cv2.split(lab)
    lc = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lc)
    enhanced = cv2.cvtColor(cv2.merge([lc, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    return cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)


class _OneEuroFilter:
    """One-Euro adaptive low-pass filter for noisy real-time signals."""

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(
        self, x: np.ndarray, update_mask: np.ndarray | None = None
    ) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        if update_mask is not None:
            if x_hat.ndim == 2 and update_mask.ndim == 1:
                update_mask_b = update_mask[:, None]
            else:
                update_mask_b = update_mask
            x_hat = np.where(update_mask_b, x_hat, self.x_prev)
            dx_hat = np.where(update_mask_b, dx_hat, self.dx_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat.copy()


def _estimate_body_center(
    kp_xy: np.ndarray, kp_conf: np.ndarray, thr: np.ndarray
) -> tuple[np.ndarray | None, int]:
    anchors = np.array([5, 6, 11, 12])
    ok = kp_conf[anchors] > thr[anchors]
    if ok.sum() < 2:
        return None, int(ok.sum())
    return kp_xy[anchors][ok].mean(axis=0), int(ok.sum())


def _estimate_body_scale(
    kp_xy: np.ndarray, kp_conf: np.ndarray, thr: np.ndarray
) -> float:
    pairs = [(5, 6), (11, 12), (5, 11), (6, 12)]
    lengths = []
    for a, b in pairs:
        if kp_conf[a] > thr[a] and kp_conf[b] > thr[b]:
            lengths.append(np.linalg.norm(kp_xy[a] - kp_xy[b]))
    if lengths:
        return float(np.median(lengths))
    return 0.0


def _kp_spread(kp_xy: np.ndarray, kp_conf: np.ndarray, thr: np.ndarray) -> float:
    high = kp_conf > thr
    if int(high.sum()) < 4:
        return 0.0
    pts = kp_xy[high]
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _torso_bone_lengths(
    kp_xy: np.ndarray, kp_conf: np.ndarray, thr: np.ndarray
) -> np.ndarray:
    out = np.full(len(_TORSO_BONES), np.nan, dtype=np.float64)
    for i, (a, b) in enumerate(_TORSO_BONES):
        if kp_conf[a] > thr[a] and kp_conf[b] > thr[b]:
            out[i] = float(np.linalg.norm(kp_xy[a] - kp_xy[b]))
    return out


def _limb_bone_lengths(
    kp_xy: np.ndarray, kp_conf: np.ndarray, thr: np.ndarray
) -> np.ndarray:
    out = np.full(len(_LIMB_BONES), np.nan, dtype=np.float64)
    for i, (a, b) in enumerate(_LIMB_BONES):
        if kp_conf[a] > thr[a] and kp_conf[b] > thr[b]:
            out[i] = float(np.linalg.norm(kp_xy[a] - kp_xy[b]))
    return out


def _detect_swim_cap(
    frame: np.ndarray,
    prev_cap: np.ndarray | None = None,
    prev_r: float = 0.0,
    darkness_thr: int = 60,
    min_area: int = 800,
    max_area: int = 8000,
    max_aspect_ratio: float = 2.8,
    continuity_weight: float = 5.0,
    max_jump_px: float = 80.0,
    radius_change_factor: float = 0.5,
) -> tuple[np.ndarray | None, float]:
    """Locate the dark swim-cap blob; return (centroid_xy, radius_px)."""
    if frame is None:
        return None, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, darkness_thr, 255, cv2.THRESH_BINARY_INV)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

    n_labels, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    best_xy: np.ndarray | None = None
    best_r = 0.0
    best_score = -np.inf
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        ar = max(w, h) / max(min(w, h), 1)
        if ar > max_aspect_ratio:
            continue
        cx, cy = float(cents[i, 0]), float(cents[i, 1])
        r = float(np.sqrt(area / np.pi))
        if prev_cap is not None:
            d = float(np.linalg.norm(np.array([cx, cy]) - prev_cap))
            if d > max_jump_px:
                continue
            if prev_r > 1.0 and abs(r - prev_r) > prev_r * radius_change_factor:
                continue
            score = area - continuity_weight * d
        else:
            score = float(area)
        if score > best_score:
            best_score = score
            best_xy = np.array([cx, cy], dtype=np.float64)
            best_r = r
    return best_xy, best_r


def _select_best_person(
    keypoints: np.ndarray,
    scores: np.ndarray,
    prev_center: np.ndarray | None,
    prev_scale: float | None,
) -> int:
    """Pick the detection that best matches the tracked body."""
    n = int(len(keypoints))
    if n == 0:
        return 0
    if n == 1 or prev_center is None or prev_scale is None or prev_scale <= 1e-3:
        return int(np.argmax(scores.mean(axis=1)))

    best_idx = int(np.argmax(scores.mean(axis=1)))
    best_score = -np.inf
    for i in range(n):
        c, _ = _estimate_body_center(keypoints[i], scores[i], _UPDATE_THR)
        mean_s = float(scores[i].mean())
        if c is None:
            score = mean_s - 1.0
        else:
            dist = float(np.linalg.norm(c - prev_center)) / prev_scale
            score = mean_s - 0.3 * dist
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _infer_pose_with_flip(
    frame: np.ndarray,
    prev_center: np.ndarray | None,
    prev_scale: float | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run RTMPose on the frame and its 180°-rotated copy; keep the better one."""
    h, w = frame.shape[:2]

    def _eval(kp_arr, sc_arr):
        if kp_arr is None or len(kp_arr) == 0:
            return None, None, -np.inf
        idx = _select_best_person(kp_arr, sc_arr, prev_center, prev_scale)
        return kp_arr[idx], sc_arr[idx], float(sc_arr[idx].mean())

    kp_n, sc_n = _model(frame)
    best_n_xy, best_n_sc, score_n = _eval(kp_n, sc_n)

    kp_f, sc_f = _model(cv2.flip(frame, -1))
    if kp_f is not None and len(kp_f) > 0:
        kp_f = kp_f.copy()
        kp_f[..., 0] = w - kp_f[..., 0]
        kp_f[..., 1] = h - kp_f[..., 1]
    best_f_xy, best_f_sc, score_f = _eval(kp_f, sc_f)

    if not np.isfinite(score_n) and not np.isfinite(score_f):
        return None, None
    if score_f > score_n:
        return best_f_xy, best_f_sc
    return best_n_xy, best_n_sc


class _KPSmoother:
    """One-Euro filter on each (x, y) coord with last-known position fallback."""

    def __init__(
        self,
        fps: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        update_thr: np.ndarray | float = _UPDATE_THR,
        conf_decay: float = 0.85,
        max_jump_per_joint: np.ndarray = _MAX_JUMP_BY_JOINT,
        max_body_jump_scale: float = 1.10,
        scale_ratio_range: tuple[float, float] = (0.45, 2.20),
        min_spread_ratio: float = 1.35,
        min_bone_ratio: float = 0.45,
        limb_ratio_range: tuple[float, float] = (0.55, 1.80),
        strict_conf_for_long_jump: float = 0.65,
        min_trusted_joints: int = 6,
        velocity_momentum: float = 0.75,
        prediction_decay: float = 0.90,
        hold_conf_frames: int = 10,
        hold_conf_decay: float = 0.97,
        max_missed_frames: int = 18,
        bone_ewma_alpha: float = 0.30,
        limb_ewma_alpha: float = 0.22,
        cap_max_distance_scale: float = 1.0,
        cap_min_scale_from_radius: float = 4.0,
    ):
        self.filter = _OneEuroFilter(freq=fps, min_cutoff=min_cutoff, beta=beta)
        update_thr = np.asarray(update_thr, dtype=np.float64)
        if update_thr.ndim == 0:
            update_thr = np.full(17, float(update_thr))
        self.update_thr = update_thr
        self.conf_decay = conf_decay
        self.max_jump_per_joint = np.asarray(max_jump_per_joint, dtype=np.float64)
        self.max_body_jump_scale = max_body_jump_scale
        self.scale_ratio_range = scale_ratio_range
        self.min_spread_ratio = min_spread_ratio
        self.min_bone_ratio = min_bone_ratio
        self.limb_ratio_range = limb_ratio_range
        self.strict_conf_for_long_jump = strict_conf_for_long_jump
        self.min_trusted_joints = min_trusted_joints
        self.velocity_momentum = velocity_momentum
        self.prediction_decay = prediction_decay
        self.hold_conf_frames = hold_conf_frames
        self.hold_conf_decay = hold_conf_decay
        self.max_missed_frames = max_missed_frames
        self.bone_ewma_alpha = bone_ewma_alpha
        self.limb_ewma_alpha = limb_ewma_alpha
        self.cap_max_distance_scale = cap_max_distance_scale
        self.cap_min_scale_from_radius = cap_min_scale_from_radius
        self._conf: np.ndarray | None = None
        self._xy: np.ndarray | None = None
        self._vel: np.ndarray | None = None
        self._center: np.ndarray | None = None
        self._scale: float | None = None
        self._bone_ewma: np.ndarray | None = None
        self._limb_ewma: np.ndarray | None = None
        self._missed_frames: np.ndarray = np.zeros(17, dtype=np.int32)
        self._cap_xy: np.ndarray | None = None
        self._cap_r: float = 0.0
        self._pose_anchor_cap: np.ndarray | None = None

    def _predict_step(
        self, translate_to_cap: np.ndarray | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._xy is None or self._conf is None:
            return None, None
        if self._vel is None:
            self._vel = np.zeros_like(self._xy)

        if translate_to_cap is not None and self._pose_anchor_cap is not None:
            delta = translate_to_cap - self._pose_anchor_cap
            self._xy = self._xy + delta
            if self._center is not None:
                self._center = self._center + delta
            self._pose_anchor_cap = translate_to_cap.copy()
            self._vel *= self.prediction_decay
        else:
            self._xy = self._xy + self._vel / max(self.filter.freq, 1e-6)
            self._vel *= self.prediction_decay

        next_miss = np.minimum(
            self._missed_frames + 1, np.iinfo(self._missed_frames.dtype).max
        )
        grace = next_miss <= self.hold_conf_frames
        floor = np.minimum(0.98, self.update_thr + 0.03)
        self._conf = np.where(
            grace, np.maximum(self._conf, floor), self._conf * self.hold_conf_decay
        )
        self._missed_frames = next_miss.astype(np.int32)
        return self._xy.copy(), self._conf.copy()

    def predict_only(
        self, current_cap: np.ndarray | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._predict_step(translate_to_cap=current_cap)

    def update(
        self,
        kp_xy: np.ndarray,
        kp_conf: np.ndarray,
        cap_xy: np.ndarray | None = None,
        cap_r: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        kp_xy = kp_xy.astype(np.float64)
        kp_conf = kp_conf.astype(np.float64)
        update = kp_conf > self.update_thr
        center, n_anchors = _estimate_body_center(kp_xy, kp_conf, self.update_thr)
        scale = _estimate_body_scale(kp_xy, kp_conf, self.update_thr)
        spread = _kp_spread(kp_xy, kp_conf, self.update_thr)
        bones = _torso_bone_lengths(kp_xy, kp_conf, self.update_thr)
        limb_bones = _limb_bone_lengths(kp_xy, kp_conf, self.update_thr)

        reject_pose = False
        if self._center is not None and self._scale is not None:
            if center is None or n_anchors < 2 or scale <= 1e-3:
                reject_pose = True
            else:
                center_jump = np.linalg.norm(center - self._center)
                scale_ratio = scale / max(self._scale, 1e-3)
                lo, hi = self.scale_ratio_range
                reject_pose = (
                    center_jump > self.max_body_jump_scale * self._scale
                    or scale_ratio < lo
                    or scale_ratio > hi
                )
            if (
                not reject_pose
                and spread > 0
                and spread < self.min_spread_ratio * self._scale
            ):
                reject_pose = True
            if not reject_pose and self._bone_ewma is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratios = bones / self._bone_ewma
                ok_b = (
                    np.isfinite(ratios)
                    & np.isfinite(self._bone_ewma)
                    & (self._bone_ewma > 1e-3)
                )
                if ok_b.any() and float(ratios[ok_b].max()) < self.min_bone_ratio:
                    reject_pose = True

        trusted_joints = int(update.sum())
        if trusted_joints < self.min_trusted_joints:
            reject_pose = True

        if reject_pose and int(self._missed_frames.min()) >= self.max_missed_frames:
            reject_pose = False

        if cap_xy is None and self._pose_anchor_cap is not None:
            reject_pose = True

        if cap_xy is not None:
            head_idx = np.array([0, 1, 2, 3, 4])
            head_ok = kp_conf[head_idx] > self.update_thr[head_idx]
            if head_ok.any():
                head_xy = kp_xy[head_idx][head_ok].mean(axis=0)
                ref_scale = max(
                    scale,
                    self._scale if self._scale is not None else 0.0,
                    float(cap_r) * self.cap_min_scale_from_radius,
                )
                if (
                    ref_scale > 1e-3
                    and float(np.linalg.norm(head_xy - cap_xy))
                    > self.cap_max_distance_scale * ref_scale
                ):
                    reject_pose = True

                if head_ok.any():
                    hip_idx = np.array([11, 12])
                    hip_ok = kp_conf[hip_idx] > self.update_thr[hip_idx]
                    if hip_ok.any():
                        hip_xy = kp_xy[hip_idx][hip_ok].mean(axis=0)
                        d_head_cap = float(np.linalg.norm(head_xy - cap_xy))
                        d_hip_cap = float(np.linalg.norm(hip_xy - cap_xy))
                        if d_hip_cap < max(d_head_cap, float(cap_r)) * 1.2:
                            reject_pose = True

        if reject_pose:
            if self._xy is not None and self._conf is not None:
                xy_p, conf_p = self._predict_step(translate_to_cap=cap_xy)
                return xy_p, conf_p
            return None, None

        if not reject_pose:
            if self._xy is not None and scale > 1e-3:
                if self._center is not None and center is not None:
                    body_shift = center - self._center
                    rel_jump = np.linalg.norm(kp_xy - (self._xy + body_shift), axis=1)
                else:
                    rel_jump = np.linalg.norm(kp_xy - self._xy, axis=1)
                lost = self._missed_frames > self.max_missed_frames
                strong_new = kp_conf >= self.strict_conf_for_long_jump
                if self._conf is not None:
                    strong_new &= self._conf < (kp_conf * 0.4)
                update &= (
                    lost | strong_new | (rel_jump < self.max_jump_per_joint * scale)
                )

            if self._limb_ewma is not None:
                lo, hi = self.limb_ratio_range
                bad_joint = np.zeros(17, dtype=bool)
                for idx, (a, b) in enumerate(_LIMB_BONES):
                    cur = limb_bones[idx]
                    ref = self._limb_ewma[idx]
                    if not np.isfinite(cur) or not np.isfinite(ref) or ref <= 1e-3:
                        continue
                    ratio = cur / ref
                    if ratio < lo or ratio > hi:
                        suspect = a if kp_conf[a] <= kp_conf[b] else b
                        bad_joint[suspect] = True
                update &= ~(bad_joint & (kp_conf < self.strict_conf_for_long_jump))

        prev_xy = self._xy.copy() if self._xy is not None else None
        smoothed_xy = self.filter(kp_xy, update_mask=update)

        if self._conf is None:
            self._conf = np.where(update, kp_conf, 0.0)
        else:
            new_conf = self._conf.copy()
            new_conf[update] = kp_conf[update]
            new_conf[~update] *= self.conf_decay
            self._conf = new_conf

        if self._vel is None:
            self._vel = np.zeros_like(smoothed_xy)
        if prev_xy is not None:
            dt = 1.0 / max(self.filter.freq, 1e-6)
            raw_vel = (smoothed_xy - prev_xy) / dt
            m = self.velocity_momentum
            up2 = update[:, None]
            self._vel = np.where(
                up2,
                m * self._vel + (1 - m) * raw_vel,
                self._vel * self.prediction_decay,
            )

        self._missed_frames = np.where(update, 0, self._missed_frames + 1).astype(
            np.int32
        )

        self._xy = smoothed_xy.copy()
        if cap_xy is not None:
            self._pose_anchor_cap = cap_xy.copy()
        if not reject_pose and center is not None and scale > 1e-3:
            self._center = center.copy()
            self._scale = scale
            valid_b = np.isfinite(bones)
            if self._bone_ewma is None:
                self._bone_ewma = np.where(valid_b, bones, np.nan)
            else:
                a = self.bone_ewma_alpha
                prev = self._bone_ewma
                new = np.where(
                    valid_b & np.isfinite(prev),
                    a * np.nan_to_num(bones) + (1 - a) * np.nan_to_num(prev),
                    np.where(valid_b, bones, prev),
                )
                self._bone_ewma = new

            valid_l = np.isfinite(limb_bones)
            if self._limb_ewma is None:
                self._limb_ewma = np.where(valid_l, limb_bones, np.nan)
            else:
                a_l = self.limb_ewma_alpha
                prev_l = self._limb_ewma
                new_l = np.where(
                    valid_l & np.isfinite(prev_l),
                    a_l * np.nan_to_num(limb_bones) + (1 - a_l) * np.nan_to_num(prev_l),
                    np.where(valid_l, limb_bones, prev_l),
                )
                self._limb_ewma = new_l
        return smoothed_xy, self._conf


# ===========================================================================
# Pose normalisation and distance metrics
# ===========================================================================


def normalize_kp(kp, scores=None, score_thr=_DEFAULT_SCORE_THR):
    """Center on torso, scale by torso size.

    Translation and scale invariant pose representation. When `scores` is given,
    visibility / confidence is used to pick anchors so rotated or partly
    occluded frames stay stable.
    """
    kp = np.asarray(kp, dtype=np.float64)
    kp_xy = kp[:, :2].copy()

    if scores is None:
        vis = np.ones(17, dtype=bool)
        w = np.ones(17, dtype=np.float64)
    else:
        scores = np.asarray(scores, dtype=np.float64)
        vis = scores > score_thr
        w = np.clip(scores, 0.0, None)

    if vis[11] and vis[12]:
        center = (kp_xy[11] + kp_xy[12]) / 2
    else:
        torso = np.array([5, 6, 11, 12])
        torso_vis = vis[torso]
        if torso_vis.any():
            ww = w[torso][torso_vis][:, None]
            center = (kp_xy[torso][torso_vis] * ww).sum(0) / (ww.sum() + 1e-8)
        elif vis.any():
            ww = w[vis][:, None]
            center = (kp_xy[vis] * ww).sum(0) / (ww.sum() + 1e-8)
        else:
            center = kp_xy.mean(0)

    kp_xy -= center

    candidates = []
    if vis[5] and vis[6]:
        candidates.append(np.linalg.norm(kp_xy[5] - kp_xy[6]))
    if vis[11] and vis[12]:
        candidates.append(np.linalg.norm(kp_xy[11] - kp_xy[12]))
    if vis[5] and vis[11]:
        candidates.append(np.linalg.norm(kp_xy[5] - kp_xy[11]))
    if vis[6] and vis[12]:
        candidates.append(np.linalg.norm(kp_xy[6] - kp_xy[12]))

    if candidates:
        scale = float(np.median(candidates))
    elif vis.any():
        scale = float(np.linalg.norm(kp_xy[vis], axis=1).max())
    else:
        scale = 0.0

    if scale > 1e-3:
        kp_xy /= scale

    return kp_xy


def frame_error_weighted(
    kp_ref, sc_ref, kp_user, sc_user, score_thr=_DEFAULT_SCORE_THR
):
    """Confidence-weighted normalized pose distance between two frames."""
    a = normalize_kp(kp_ref, sc_ref, score_thr=score_thr)
    b = normalize_kp(kp_user, sc_user, score_thr=score_thr)

    sc_ref = np.asarray(sc_ref, dtype=np.float64)
    sc_user = np.asarray(sc_user, dtype=np.float64)
    mask = (sc_ref > score_thr) & (sc_user > score_thr)
    if not mask.any():
        return float("inf")

    d = np.linalg.norm(a[mask] - b[mask], axis=1)
    w = (sc_ref[mask] + sc_user[mask]) / 2.0
    return float((w * d).sum() / (w.sum() + 1e-8))


def _precompute_pose_features(keypoints_seq, scores_seq, score_thr=_DEFAULT_SCORE_THR):
    """Normalize all frames once and return stacked arrays.

    Returns:
        pts:    (T, 17, 2) float64 — normalized keypoint coordinates
        vis:    (T, 17)    bool    — joints with confidence > score_thr
        scores: (T, 17)    float64 — raw confidence scores
    """
    T_frames = len(keypoints_seq)
    pts = np.zeros((T_frames, 17, 2), dtype=np.float64)
    vis = np.zeros((T_frames, 17), dtype=bool)
    scores = np.zeros((T_frames, 17), dtype=np.float64)
    for t, (kp, sc) in enumerate(zip(keypoints_seq, scores_seq)):
        sc_arr = np.asarray(sc, dtype=np.float64)
        pts[t] = normalize_kp(kp, sc, score_thr=score_thr)
        vis[t] = sc_arr > score_thr
        scores[t] = sc_arr
    return pts, vis, scores


def _cost_matrix(ref_feat, user_feat, score_thr=_DEFAULT_SCORE_THR):
    """Build the (M, N) confidence-weighted cost matrix."""
    ref_pts, ref_vis, ref_sc = ref_feat
    usr_pts, usr_vis, usr_sc = user_feat

    M = ref_pts.shape[0]
    N = usr_pts.shape[0]
    C = np.zeros((M, N), dtype=np.float64)

    for k in range(17):
        r_vis = ref_vis[:, k]
        u_vis = usr_vis[:, k]
        shared = np.outer(r_vis, u_vis)
        if not shared.any():
            continue

        diff = ref_pts[:, k, np.newaxis, :] - usr_pts[np.newaxis, :, k, :]
        dist = np.linalg.norm(diff, axis=2)

        r_w = ref_sc[:, k][:, np.newaxis]
        u_w = usr_sc[:, k][np.newaxis, :]
        w = (r_w + u_w) / 2.0

        C += np.where(shared, w * dist, 0.0)

    total_w = np.zeros((M, N), dtype=np.float64)
    for k in range(17):
        r_vis = ref_vis[:, k]
        u_vis = usr_vis[:, k]
        shared = np.outer(r_vis, u_vis)
        r_w = ref_sc[:, k][:, np.newaxis]
        u_w = usr_sc[:, k][np.newaxis, :]
        w = (r_w + u_w) / 2.0
        total_w += np.where(shared, w, 0.0)

    valid = total_w > 1e-8
    C[valid] /= total_w[valid]

    fallback = C[valid].max() * 2.0 if valid.any() else 1.0
    C[~valid] = fallback

    return C


def _subsequence_dtw(C):
    """Open-begin / open-end subsequence DTW.

    The reference axis (rows, length M) must be matched in full.
    The user axis (cols, length N) can start and end freely, so idle
    frames before and after the movement are skipped automatically.

    Returns:
        distance  — total path cost (sum, not mean)
        path      — list of (i, j) index pairs, same format as fastdtw
        j_start   — first user frame used
        j_end     — last user frame used
    """
    M, N = C.shape

    D = np.full((M, N), np.inf, dtype=np.float64)
    D[0, :] = C[0, :]

    for i in range(1, M):
        from_above = D[i - 1, :]
        from_diag = np.full(N, np.inf)
        from_diag[1:] = D[i - 1, :-1]

        row = np.minimum(from_above, from_diag) + C[i, :]

        for j in range(1, N):
            candidate = row[j - 1] + C[i, j]
            if candidate < row[j]:
                row[j] = candidate

        D[i, :] = row

    j_end = int(np.argmin(D[M - 1, :]))
    distance = float(D[M - 1, j_end])

    path = []
    i, j = M - 1, j_end
    while i > 0:
        path.append((i, j))
        options = [
            (D[i - 1, j - 1] if j > 0 else np.inf, i - 1, j - 1),
            (D[i - 1, j], i - 1, j),
            (D[i, j - 1] if j > 0 else np.inf, i, j - 1),
        ]
        _, i, j = min(options, key=lambda x: x[0])
    path.append((i, j))
    path.reverse()

    j_start = path[0][1]
    return distance, path, j_start, j_end


def _stabilize_user_window(path, user_n_frames, window=12, min_hits=10):
    """Trim noisy DTW edges; keep only stable user-frame coverage window."""
    if not path or user_n_frames <= 0:
        return path, 0, 0

    if window <= 0:
        j_start = int(path[0][1])
        j_end = int(path[-1][1])
        return path, j_start, j_end

    window = min(window, user_n_frames)
    min_hits = min(min_hits, window)

    mask = np.zeros(user_n_frames, dtype=np.int32)
    for _, j in path:
        if 0 <= j < user_n_frames:
            mask[j] = 1

    rolling = np.convolve(mask, np.ones(window, dtype=np.int32), mode="valid")
    stable_windows = rolling >= min_hits

    if not stable_windows.any():
        j_start = int(path[0][1])
        j_end = int(path[-1][1])
        return path, j_start, j_end

    first_window = int(np.argmax(stable_windows))
    last_window = int(len(stable_windows) - 1 - np.argmax(stable_windows[::-1]))

    j_start = first_window
    j_end = last_window + window - 1

    filtered_path = [(int(i), int(j)) for i, j in path if j_start <= int(j) <= j_end]
    if not filtered_path:
        j_start = int(path[0][1])
        j_end = int(path[-1][1])
        return path, j_start, j_end

    j_start = int(filtered_path[0][1])
    j_end = int(filtered_path[-1][1])
    return filtered_path, j_start, j_end


def scale_keypoints_to_frame(kp, w_src, h_src, w_dst, h_dst):
    """Map keypoints from source video pixel space to destination frame size."""
    out = kp.astype(np.float64, copy=True)
    out[:, 0] *= w_dst / float(w_src)
    out[:, 1] *= h_dst / float(h_src)
    return out.astype(kp.dtype, copy=False)


def _body_center_and_scale(kp, scores=None, score_thr=_DEFAULT_SCORE_THR):
    """Return (center_xy, scale) of the body in pixel coords."""
    k = np.asarray(kp, dtype=np.float64)
    xy = k[:, :2]

    if scores is not None:
        vis = np.asarray(scores, dtype=np.float64) > score_thr
    elif k.shape[1] >= 3:
        vis = k[:, 2] > 0
    else:
        vis = np.ones(17, dtype=bool)

    if vis[11] and vis[12]:
        center = (xy[11] + xy[12]) / 2
    elif vis[5] or vis[6] or vis[11] or vis[12]:
        torso = np.array([5, 6, 11, 12])
        sel = torso[vis[torso]]
        center = xy[sel].mean(0)
    elif vis.any():
        center = xy[vis].mean(0)
    else:
        center = xy.mean(0)

    candidates = []
    if vis[5] and vis[6]:
        candidates.append(np.linalg.norm(xy[5] - xy[6]))
    if vis[11] and vis[12]:
        candidates.append(np.linalg.norm(xy[11] - xy[12]))
    if vis[5] and vis[11]:
        candidates.append(np.linalg.norm(xy[5] - xy[11]))
    if vis[6] and vis[12]:
        candidates.append(np.linalg.norm(xy[6] - xy[12]))

    scale = float(np.median(candidates)) if candidates else 0.0
    return center, scale


def remap_reference_keypoints_for_overlay(
    kp_ref,
    kp_user,
    sc_ref=None,
    sc_user=None,
    w_src=None,
    h_src=None,
    w_dst=None,
    h_dst=None,
):
    """Place the reference skeleton onto the user frame as a similarity transform."""
    ref_xy = np.asarray(kp_ref, dtype=np.float64)[:, :2]
    ref_c, ref_s = _body_center_and_scale(kp_ref, sc_ref)
    user_c, user_s = _body_center_and_scale(kp_user, sc_user)

    if ref_s < 1e-3 or user_s < 1e-3:
        s = 1.0
    else:
        s = user_s / ref_s

    new_xy = (ref_xy - ref_c) * s + user_c

    out = np.asarray(kp_ref, dtype=np.float64).copy()
    out[:, :2] = new_xy
    return out.astype(np.asarray(kp_ref).dtype, copy=False)


def _draw_skeleton(frame, kp, color, dot_radius=4, line_thickness=2):
    """Draw skeleton lines then joint dots for one person.

    kp is a (17, 3) array with columns [x, y, conf].
    """
    for a, b in COCO_CONNECTIONS:
        if kp[a, 2] > _DEFAULT_SCORE_THR and kp[b, 2] > _DEFAULT_SCORE_THR:
            cv2.line(
                frame,
                (int(kp[a, 0]), int(kp[a, 1])),
                (int(kp[b, 0]), int(kp[b, 1])),
                color,
                line_thickness,
                cv2.LINE_AA,
            )
    for x, y, v in kp:
        if v > _DEFAULT_SCORE_THR:
            cv2.circle(frame, (int(x), int(y)), dot_radius, color, -1)


def _draw_overlay_legend(frame, ref_color, user_color):
    """Draw legend explaining which skeleton belongs to whom."""
    h, w = frame.shape[:2]
    pad = 12
    box_w = 250
    box_h = 70
    x0 = max(w - box_w - pad, 0)
    y0 = pad
    x1 = min(x0 + box_w, w - 1)
    y1 = min(y0 + box_h, h - 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)

    line1_y = y0 + 24
    line2_y = y0 + 50
    swatch_size = 12
    text_x = x0 + 34

    cv2.rectangle(
        frame,
        (x0 + 12, line1_y - swatch_size + 3),
        (x0 + 12 + swatch_size, line1_y + 3),
        ref_color,
        -1,
    )
    cv2.putText(
        frame,
        "Original (reference)",
        (text_x, line1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        frame,
        (x0 + 12, line2_y - swatch_size + 3),
        (x0 + 12 + swatch_size, line2_y + 3),
        user_color,
        -1,
    )
    cv2.putText(
        frame,
        "Your keypoints",
        (text_x, line2_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def draw_overlay(frame, kp, error, threshold=0.5, kp_ref=None):
    frame = frame.copy()
    ref_color = (255, 200, 0)

    if kp_ref is not None:
        _draw_skeleton(frame, kp_ref, color=ref_color, dot_radius=5, line_thickness=2)

    if error < threshold:
        color = (0, 255, 0)
        text = "GOOD"
    else:
        color = (0, 0, 255)
        text = "BAD"

    cv2.putText(
        frame,
        f"{text} | error: {error:.2f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA,
    )

    _draw_skeleton(frame, kp, color=color, dot_radius=4, line_thickness=2)
    _draw_overlay_legend(frame, ref_color=ref_color, user_color=color)

    return frame


def extract_keypoints(
    video_path: str,
    cache_frames: bool = False,
    *,
    min_cutoff: float = 1.3,
    beta: float = 0.012,
    limb_ratio_range: tuple[float, float] = (0.68, 1.45),
    strict_conf_for_long_jump: float = 0.75,
    min_trusted_joints: int = 7,
    prediction_decay: float = 0.90,
    hold_conf_frames: int = 14,
    hold_conf_decay: float = 0.985,
    preprocess: bool = True,
    use_orientation_flip: bool = True,
    use_cap_anchor: bool = False,
    cap_max_distance_scale: float = 1.0,
    frame_stride: int = 1,
) -> dict:
    """Run RTMPose-L + One-Euro smoother on every frame of a video.

    Returns a dict with keys:
        keypoints_seq  — list of (17, 3) arrays [x, y, conf] per frame
        scores_seq     — list of (17,) confidence arrays per frame
        frame_indices  — list of consecutive integers [0 .. N-1]
        frames         — list of BGR frames or None (when cache_frames=False)
        fps            — video frame rate
        wh             — (width, height) of the video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    video_wh = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    smoother = _KPSmoother(
        fps=fps,
        min_cutoff=min_cutoff,
        beta=beta,
        limb_ratio_range=limb_ratio_range,
        strict_conf_for_long_jump=strict_conf_for_long_jump,
        min_trusted_joints=min_trusted_joints,
        prediction_decay=prediction_decay,
        hold_conf_frames=hold_conf_frames,
        hold_conf_decay=hold_conf_decay,
        cap_max_distance_scale=cap_max_distance_scale,
    )

    keypoints_seq = []
    scores_seq = []
    frame_indices = []
    frames_cache = [] if cache_frames else None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cache_frames:
            frames_cache.append(frame)

        if frame_stride > 1 and frame_idx % frame_stride != 0:
            # Skip model inference on this frame; let the smoother predict.
            kp_xy_s, kp_conf_s = smoother.predict_only(current_cap=None)
        else:
            infer_frame = _preprocess(frame) if preprocess else frame

            if use_cap_anchor:
                cap_xy, cap_r = _detect_swim_cap(
                    frame, prev_cap=smoother._cap_xy, prev_r=smoother._cap_r
                )
                if cap_xy is not None:
                    smoother._cap_xy = cap_xy
                    smoother._cap_r = cap_r
            else:
                cap_xy, cap_r = None, 0.0

            if use_orientation_flip:
                best_xy, best_sc = _infer_pose_with_flip(
                    infer_frame, smoother._center, smoother._scale
                )
            else:
                kp_arr, sc_arr = _model(infer_frame)
                if kp_arr is not None and len(kp_arr) > 0:
                    idx = _select_best_person(
                        kp_arr, sc_arr, smoother._center, smoother._scale
                    )
                    best_xy, best_sc = kp_arr[idx], sc_arr[idx]
                else:
                    best_xy = best_sc = None

            if best_xy is not None:
                kp_xy_s, kp_conf_s = smoother.update(
                    best_xy, best_sc, cap_xy=cap_xy, cap_r=cap_r
                )
            else:
                kp_xy_s, kp_conf_s = smoother.predict_only(current_cap=cap_xy)

        if kp_xy_s is not None and kp_conf_s is not None:
            kp_out = np.zeros((17, 3), dtype=np.float64)
            kp_out[:, :2] = kp_xy_s
            kp_out[:, 2] = kp_conf_s
            keypoints_seq.append(kp_out)
            scores_seq.append(kp_conf_s.copy())
            frame_indices.append(frame_idx)

        frame_idx += 1

    cap.release()

    if video_wh[0] == 0:
        video_wh = (0, 0)

    return {
        "keypoints_seq": keypoints_seq,
        "scores_seq": scores_seq,
        "frame_indices": frame_indices,
        "frames": frames_cache,
        "fps": fps,
        "wh": video_wh,
    }


def _open_writer(
    path: str,
    fps: float,
    w: int,
    h: int,
    preferred: str = "avc1",
) -> tuple[cv2.VideoWriter, str]:
    """Open a VideoWriter preferring H.264 (avc1) with mp4v as fallback."""
    for tag in (preferred, "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, (w, h))
        if writer.isOpened():
            return writer, tag
    raise RuntimeError(f"Cannot open VideoWriter for {path}")


def compare_videos(
    user_video: str,
    ref_data: dict,
    output_path: str = "result.mp4",
    threshold: float = 0.5,
    output_max_side: int | None = None,
    user_kwargs: dict | None = None,
) -> dict:
    """Compare a user video against pre-extracted reference keypoints.

    Args:
        user_video: Path to the user's performance video.
        ref_data: Dict returned by extract_keypoints() for the reference video.
        output_path: Where to write the annotated output video.
        threshold: Per-frame error threshold for GOOD/BAD label in the overlay.

    Returns:
        Dict with dtw_distance, normalized_distance, similarity,
        weighted_similarity, per_frame_errors, output_path.
    """
    ref_keypoints_seq = ref_data["keypoints_seq"]
    ref_scores_seq = ref_data["scores_seq"]
    ref_wh = ref_data["wh"]

    if ref_wh[0] <= 0 or ref_wh[1] <= 0:
        raise RuntimeError("Reference data contains no valid frames")
    if not ref_keypoints_seq:
        raise RuntimeError("No keypoints detected in reference video")

    user_data = extract_keypoints(user_video, cache_frames=True, **(user_kwargs or {}))
    user_keypoints_seq = user_data["keypoints_seq"]
    user_scores_seq = user_data["scores_seq"]
    user_frame_indices = user_data["frame_indices"]
    user_frames = user_data["frames"]
    fps = user_data["fps"]
    user_wh = user_data["wh"]

    if not user_frames:
        raise RuntimeError(f"No frames could be read from {user_video}")
    if not user_keypoints_seq:
        raise RuntimeError("No keypoints detected in user video")
    if len(user_keypoints_seq) < len(ref_keypoints_seq):
        raise RuntimeError(
            f"User video too short: {len(user_keypoints_seq)} detected frames, "
            f"reference requires at least {len(ref_keypoints_seq)}."
        )

    ref_feat = _precompute_pose_features(ref_keypoints_seq, ref_scores_seq)
    user_feat = _precompute_pose_features(user_keypoints_seq, user_scores_seq)
    cost = _cost_matrix(ref_feat, user_feat)
    distance, path, j_start, j_end = _subsequence_dtw(cost)
    path, j_start, j_end = _stabilize_user_window(
        path, len(user_keypoints_seq), window=12, min_hits=10
    )
    distance = float(sum(cost[int(i), int(j)] for i, j in path))

    normalized_distance = distance / max(len(path), 1)
    similarity = 1 / (1 + normalized_distance)

    h, w = user_frames[0].shape[:2]
    if (w, h) != user_wh:
        raise RuntimeError("Internal error: user_wh does not match cached frames")

    if output_max_side is not None and max(w, h) > output_max_side:
        scale = output_max_side / max(w, h)
        w_out = int(w * scale) & ~1  # keep even dimensions for H.264
        h_out = int(h * scale) & ~1
    else:
        scale = 1.0
        w_out, h_out = w, h

    out, _used_codec = _open_writer(output_path, fps, w_out, h_out)

    # Build a dense j→i lookup so every user frame in [j_start, j_end] can be
    # annotated without repeating or skipping frames. When the DTW path maps
    # multiple reference frames to the same j we keep the last one (fine for
    # a per-frame error display). Gaps (user frames not explicitly in the path)
    # are filled by forward-filling from the preceding matched reference frame.
    j_to_i: dict[int, int] = {}
    for pi, pj in path:
        j_to_i[int(pj)] = int(pi)

    # Forward-fill: for j values in [j_start, j_end] missing from the path,
    # inherit the last seen reference index so the overlay stays meaningful.
    last_i = j_to_i.get(j_start, 0)
    j_to_i_dense: dict[int, int] = {}
    for j in range(j_start, j_end + 1):
        if j in j_to_i:
            last_i = j_to_i[j]
        j_to_i_dense[j] = last_i

    per_frame_errors = []
    total_sim = 0.0
    sim_count = 0

    # Compute similarity stats from the DTW path (unchanged semantics).
    for pi, pj in path:
        pi, pj = int(pi), int(pj)
        if pi >= len(ref_keypoints_seq) or pj >= len(user_keypoints_seq):
            continue
        kp_ref = ref_keypoints_seq[pi]
        sc_ref = ref_scores_seq[pi]
        kp_user = user_keypoints_seq[pj]
        sc_user = user_scores_seq[pj]

        err = frame_error_weighted(kp_ref, sc_ref, kp_user, sc_user)
        per_frame_errors.append((pi, pj, err))

        kp_ref_n = normalize_kp(kp_ref, sc_ref)
        kp_user_n = normalize_kp(kp_user, sc_user)
        s_num = 0.0
        s_den = 0.0
        for k in range(17):
            if sc_ref[k] > _DEFAULT_SCORE_THR and sc_user[k] > _DEFAULT_SCORE_THR:
                d = np.linalg.norm(kp_ref_n[k] - kp_user_n[k])
                wgt = (sc_ref[k] + sc_user[k]) / 2
                s_num += wgt * np.exp(-d)
                s_den += wgt
        total_sim += s_num / (s_den + 1e-8)
        sim_count += 1

    # Build a per-user-frame error map for the overlay (nearest path error).
    j_to_err: dict[int, float] = {}
    for pi, pj, err in per_frame_errors:
        j_to_err[pj] = err
    # Forward-fill errors for gap frames.
    last_err = 0.0
    j_to_err_dense: dict[int, float] = {}
    for j in range(j_start, j_end + 1):
        if j in j_to_err:
            last_err = j_to_err[j]
        j_to_err_dense[j] = last_err

    # Write every user frame from j_start to j_end in sequence — this keeps
    # the output at native FPS and eliminates DTW-induced freezes/skips.
    for j in range(j_start, j_end + 1):
        if j >= len(user_keypoints_seq):
            break
        video_frame_idx = user_frame_indices[j]
        if video_frame_idx >= len(user_frames):
            continue

        ref_i = j_to_i_dense[j]
        if ref_i >= len(ref_keypoints_seq):
            continue

        kp_ref = ref_keypoints_seq[ref_i]
        sc_ref = ref_scores_seq[ref_i]
        kp_user = user_keypoints_seq[j]
        sc_user = user_scores_seq[j]
        err = j_to_err_dense[j]

        frame = user_frames[video_frame_idx]
        if scale != 1.0:
            kp_ref_draw_src = remap_reference_keypoints_for_overlay(
                kp_ref, kp_user, sc_ref=sc_ref, sc_user=sc_user
            )
            kp_user_scaled = kp_user.copy()
            kp_user_scaled[:, 0] *= scale
            kp_user_scaled[:, 1] *= scale
            kp_ref_draw = kp_ref_draw_src.copy()
            kp_ref_draw[:, 0] *= scale
            kp_ref_draw[:, 1] *= scale
            frame_small = cv2.resize(
                frame, (w_out, h_out), interpolation=cv2.INTER_AREA
            )
            frame_out = draw_overlay(
                frame_small,
                kp_user_scaled,
                err,
                threshold=threshold,
                kp_ref=kp_ref_draw,
            )
        else:
            kp_ref_draw = remap_reference_keypoints_for_overlay(
                kp_ref, kp_user, sc_ref=sc_ref, sc_user=sc_user
            )
            frame_out = draw_overlay(
                frame, kp_user, err, threshold=threshold, kp_ref=kp_ref_draw
            )
        out.write(frame_out)

    out.release()

    weighted_similarity = total_sim / (sim_count + 1e-8)

    return {
        "dtw_distance": float(distance),
        "normalized_distance": float(normalized_distance),
        "similarity": float(similarity),
        "weighted_similarity": float(weighted_similarity),
        "per_frame_errors": per_frame_errors,
        "output_path": output_path,
        "j_start": j_start,
        "j_end": j_end,
        "user_n_frames": len(user_keypoints_seq),
        "user_fps": fps,
    }
