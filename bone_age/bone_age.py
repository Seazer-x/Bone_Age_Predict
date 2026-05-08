"""
bone_age.py

Reconstructed bone-age inference logic for Seazer-x/Bone_Age_Predict.

Pipeline:
1. Use a YOLOv5 detection model to detect 21 hand-bone ROIs.
2. Keep the 13 RUS-CHN scoring ROIs by x-coordinate order.
3. Use 9 YOLOv5 classification models to classify maturation levels.
4. Convert levels to CHN score by sex.
5. Convert CHN score to bone age with the RUS-CHN polynomial.

Expected usage from Bone-pre.py:
    from bone_age.bone_age import Bone_Age

    models = ["Radius", "Ulna", "MCPFirst", "MCP", "PIP",
              "PIPFirst", "MIP", "DIP", "DIPFirst"]
    weights = [Path("./bone_age/" + x + "/best.pt") for x in models]
    Bone = Bone_Age(weights, models)

    msg, success = Bone.run(
        weights_path=Path("./bone_age/bone_age.pt"),
        sex="boy",
        im=image_rgb,
        conf_thres=0.6,
        iou_thres=0.45,
    )
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode


# Detector class order used by the handbone dataset:
# 0 Radius, 1 Ulna, 2 MCPFirst, 3 MCP,
# 4 ProximalPhalanx, 5 MiddlePhalanx, 6 DistalPhalanx.
DEFAULT_DETECT_NAMES = {
    0: "Radius",
    1: "Ulna",
    2: "MCPFirst",
    3: "MCP",
    4: "ProximalPhalanx",
    5: "MiddlePhalanx",
    6: "DistalPhalanx",
}

DETECTION_MAX_COUNTS = {
    "Radius": 1,
    "Ulna": 1,
    "MCPFirst": 1,
    "MCP": 4,
    "ProximalPhalanx": 5,
    "MiddlePhalanx": 4,
    "DistalPhalanx": 5,
}

# The detected boxes are sorted by x1 from left to right.
# For the standard left-hand X-ray dataset, this maps to fifth -> third -> first.
CROP_RULES = [
    ("DistalPhalanx", [0, 2, 4], ["DIPFifth", "DIPThird", "DIPFirst"]),
    ("MiddlePhalanx", [0, 2], ["MIPFifth", "MIPThird"]),
    ("ProximalPhalanx", [0, 2, 4], ["PIPFifth", "PIPThird", "PIPFirst"]),
    ("MCP", [0, 2], ["MCPFifth", "MCPThird"]),
    ("MCPFirst", [0], ["MCPFirst"]),
    ("Ulna", [0], ["Ulna"]),
    ("Radius", [0], ["Radius"]),
]

CATEGORY = [
    "DIPFifth",
    "DIPThird",
    "DIPFirst",
    "MIPFifth",
    "MIPThird",
    "PIPFifth",
    "PIPThird",
    "PIPFirst",
    "MCPFifth",
    "MCPThird",
    "MCPFirst",
    "Ulna",
    "Radius",
]

# 13 scoring parts -> 9 YOLOv5 classification models.
PART_TO_MODEL = {
    "MCPFirst": "MCPFirst",
    "MCPThird": "MCP",
    "MCPFifth": "MCP",
    "PIPFirst": "PIPFirst",
    "PIPThird": "PIP",
    "PIPFifth": "PIP",
    "MIPThird": "MIP",
    "MIPFifth": "MIP",
    "DIPFirst": "DIPFirst",
    "DIPThird": "DIP",
    "DIPFifth": "DIP",
    "Radius": "Radius",
    "Ulna": "Ulna",
}

SCORE = {
    "girl": {
        "Radius": [10, 15, 22, 25, 40, 59, 91, 125, 138, 178, 192, 199, 203, 210],
        "Ulna": [27, 31, 36, 50, 73, 95, 120, 157, 168, 176, 182, 189],
        "MCPFirst": [5, 7, 10, 16, 23, 28, 34, 41, 47, 53, 66],
        "MCPThird": [3, 5, 6, 9, 14, 21, 32, 40, 47, 51],
        "MCPFifth": [4, 5, 7, 10, 15, 22, 33, 43, 47, 51],
        "PIPFirst": [6, 7, 8, 11, 17, 26, 32, 38, 45, 53, 60, 67],
        "PIPThird": [3, 5, 7, 9, 15, 20, 25, 29, 35, 41, 46, 51],
        "PIPFifth": [4, 5, 7, 11, 18, 21, 25, 29, 34, 40, 45, 50],
        "MIPThird": [4, 5, 7, 10, 16, 21, 25, 29, 35, 43, 46, 51],
        "MIPFifth": [3, 5, 7, 12, 19, 23, 27, 32, 35, 39, 43, 49],
        "DIPFirst": [5, 6, 8, 10, 20, 31, 38, 44, 45, 52, 67],
        "DIPThird": [3, 5, 7, 10, 16, 24, 30, 33, 36, 39, 49],
        "DIPFifth": [5, 6, 7, 11, 18, 25, 29, 33, 35, 39, 49],
    },
    "boy": {
        "Radius": [8, 11, 15, 18, 31, 46, 76, 118, 135, 171, 188, 197, 201, 209],
        "Ulna": [25, 30, 35, 43, 61, 80, 116, 157, 168, 180, 187, 194],
        "MCPFirst": [4, 5, 8, 16, 22, 26, 34, 39, 45, 52, 66],
        "MCPThird": [3, 4, 5, 8, 13, 19, 30, 38, 44, 51],
        "MCPFifth": [3, 4, 6, 9, 14, 19, 31, 41, 46, 50],
        "PIPFirst": [4, 5, 7, 11, 17, 23, 29, 36, 44, 52, 59, 66],
        "PIPThird": [3, 4, 5, 8, 14, 19, 23, 28, 34, 40, 45, 50],
        "PIPFifth": [3, 4, 6, 10, 16, 19, 24, 28, 33, 40, 44, 50],
        "MIPThird": [3, 4, 5, 9, 14, 18, 23, 28, 35, 42, 45, 50],
        "MIPFifth": [3, 4, 6, 11, 17, 21, 26, 31, 36, 40, 43, 49],
        "DIPFirst": [4, 5, 6, 9, 19, 28, 36, 43, 46, 51, 67],
        "DIPThird": [3, 4, 5, 9, 15, 23, 29, 33, 37, 40, 49],
        "DIPFifth": [3, 4, 6, 11, 17, 23, 29, 32, 36, 40, 49],
    },
}

REPORT_TEMPLATE = """第一掌骨骺分级{}级，得{}分；第三掌骨骨骺分级{}级，得{}分；第五掌骨骨骺分级{}级，得{}分；
第一近节指骨骨骺分级{}级，得{}分；第三近节指骨骨骺分级{}级，得{}分；第五近节指骨骨骺分级{}级，得{}分；
第三中节指骨骨骺分级{}级，得{}分；第五中节指骨骨骺分级{}级，得{}分；
第一远节指骨骨骺分级{}级，得{}分；第三远节指骨骨骺分级{}级，得{}分；第五远节指骨骨骺分级{}级，得{}分；
尺骨分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。

RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。
"""


def _stride_value(model: DetectMultiBackend) -> int:
    stride = getattr(model, "stride", 32)
    if isinstance(stride, torch.Tensor):
        return int(stride.max().item())
    if isinstance(stride, (list, tuple)):
        return int(max(stride))
    try:
        return int(stride)
    except Exception:
        return 32


def calc_bone_age(score: int | float, sex: str) -> float:
    """Convert total CHN score to bone age in years."""
    if sex == "boy":
        bone_age = (
            2.01790023656577
            + (-0.0931820870747269) * score
            + math.pow(score, 2) * 0.00334709095418796
            + math.pow(score, 3) * (-3.32988302362153e-05)
            + math.pow(score, 4) * (1.75712910819776e-07)
            + math.pow(score, 5) * (-5.59998691223273e-10)
            + math.pow(score, 6) * (1.1296711294933e-12)
            + math.pow(score, 7) * (-1.45218037113138e-15)
            + math.pow(score, 8) * (1.15333377080353e-18)
            + math.pow(score, 9) * (-5.15887481551927e-22)
            + math.pow(score, 10) * (9.94098428102335e-26)
        )
    elif sex == "girl":
        bone_age = (
            5.81191794824917
            + (-0.271546561737745) * score
            + math.pow(score, 2) * 0.00526301486340724
            + math.pow(score, 3) * (-4.37797717401925e-05)
            + math.pow(score, 4) * (2.0858722025667e-07)
            + math.pow(score, 5) * (-6.21879866563429e-10)
            + math.pow(score, 6) * (1.19909931745368e-12)
            + math.pow(score, 7) * (-1.49462900826936e-15)
            + math.pow(score, 8) * (1.162435538672e-18)
            + math.pow(score, 9) * (-5.12713017846218e-22)
            + math.pow(score, 10) * (9.78989966891478e-26)
        )
    else:
        raise ValueError("sex must be 'boy' or 'girl'.")

    return round(float(bone_age), 2)


def build_report(levels: Mapping[str, int], scores: Mapping[str, int], total_score: int, bone_age: float) -> str:
    """Build the same report format used by the original RUS-CHN implementation."""
    return REPORT_TEMPLATE.format(
        levels["MCPFirst"], scores["MCPFirst"],
        levels["MCPThird"], scores["MCPThird"],
        levels["MCPFifth"], scores["MCPFifth"],
        levels["PIPFirst"], scores["PIPFirst"],
        levels["PIPThird"], scores["PIPThird"],
        levels["PIPFifth"], scores["PIPFifth"],
        levels["MIPThird"], scores["MIPThird"],
        levels["MIPFifth"], scores["MIPFifth"],
        levels["DIPFirst"], scores["DIPFirst"],
        levels["DIPThird"], scores["DIPThird"],
        levels["DIPFifth"], scores["DIPFifth"],
        levels["Ulna"], scores["Ulna"],
        levels["Radius"], scores["Radius"],
        total_score, bone_age,
    )


class Bone_Age:
    """
    Bone age predictor compatible with the original Streamlit entry file.

    Parameters
    ----------
    weights:
        Paths to the 9 YOLOv5 classification weights.
    models:
        Names of the 9 classification models, e.g.
        ["Radius", "Ulna", "MCPFirst", "MCP", "PIP", "PIPFirst", "MIP", "DIP", "DIPFirst"].
    device:
        CUDA device string, e.g. "0", "0,1", or "cpu". Empty means auto-select.
    imgsz_det:
        Detector input size.
    imgsz_cls:
        Classifier input size. YOLOv5 classification is usually trained with 224.
    fp16:
        Use FP16 on CUDA. None means use FP16 if CUDA is available.
    crop_padding:
        Relative padding added to each detected ROI. The original logic uses exact boxes,
        so the default is 0.0.
    """

    def __init__(
        self,
        weights: Iterable[str | Path],
        models: Iterable[str],
        device: str = "",
        imgsz_det: int = 640,
        imgsz_cls: int = 224,
        fp16: Optional[bool] = None,
        crop_padding: float = 0.0,
    ) -> None:
        self.device = select_device(device)
        self.fp16 = (self.device.type != "cpu") if fp16 is None else bool(fp16 and self.device.type != "cpu")
        self.imgsz_det = imgsz_det
        self.imgsz_cls = imgsz_cls
        self.crop_padding = float(crop_padding)

        self.classifier_weights: Dict[str, Path] = {
            str(name): Path(weight) for name, weight in zip(models, weights)
        }

        missing = [str(p) for p in self.classifier_weights.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("分类模型权重不存在：\n" + "\n".join(missing))

        self.classifiers: Dict[str, DetectMultiBackend] = {}
        for name, weight in self.classifier_weights.items():
            self.classifiers[name] = DetectMultiBackend(weight, device=self.device, fp16=self.fp16, fuse=True)

        self.detector: Optional[DetectMultiBackend] = None
        self.detector_weight: Optional[Path] = None

        self._warmup_classifiers()

    def _warmup_classifiers(self) -> None:
        if self.device.type == "cpu":
            return
        for model in self.classifiers.values():
            model.warmup(imgsz=(1, 3, self.imgsz_cls, self.imgsz_cls))

    def _load_detector(self, weight: str | Path) -> DetectMultiBackend:
        weight = Path(weight)
        if not weight.exists():
            raise FileNotFoundError(f"检测模型权重不存在：{weight}")

        if self.detector is None or self.detector_weight != weight:
            self.detector = DetectMultiBackend(weight, device=self.device, fp16=self.fp16, fuse=True)
            self.detector_weight = weight
            if self.device.type != "cpu":
                self.detector.warmup(imgsz=(1, 3, self.imgsz_det, self.imgsz_det))

        return self.detector

    @smart_inference_mode()
    def run(
        self,
        weights_path: str | Path,
        sex: str,
        im: np.ndarray,
        conf_thres: float = 0.6,
        iou_thres: float = 0.45,
    ) -> Tuple[str, bool]:
        """
        Run detector + classifiers + RUS-CHN scoring.

        Returns
        -------
        (msg, success)
            msg is the final report if success is True; otherwise it is the failure reason.
        """
        try:
            sex = sex.lower().strip()
            if sex not in SCORE:
                raise ValueError("sex must be 'boy' or 'girl'.")

            image_rgb = self._ensure_rgb(im)
            detector = self._load_detector(weights_path)

            detections = self._detect(detector, image_rgb, conf_thres, iou_thres)
            rois = self._crop_rois(image_rgb, detections)

            missing_rois = [name for name in CATEGORY if name not in rois]
            if missing_rois:
                raise RuntimeError("关键骨骼 ROI 缺失：" + ", ".join(missing_rois))

            levels, probabilities = self._classify_rois(rois)
            missing_levels = [name for name in CATEGORY if name not in levels]
            if missing_levels:
                raise RuntimeError("骨骼分级缺失：" + ", ".join(missing_levels))

            scores: Dict[str, int] = {}
            total_score = 0
            for part in CATEGORY:
                level = levels[part]
                score_list = SCORE[sex][part]
                if level < 1 or level > len(score_list):
                    raise RuntimeError(
                        f"{part} 分类等级越界：level={level}, "
                        f"合法范围=1..{len(score_list)}"
                    )
                score = int(score_list[level - 1])
                scores[part] = score
                total_score += score

            bone_age = calc_bone_age(total_score, sex)
            report = build_report(levels, scores, total_score, bone_age)

            # Keep probabilities visible for debugging but do not alter the original report format.
            low_conf = {
                name: round(prob, 3)
                for name, prob in probabilities.items()
                if prob < conf_thres
            }
            if low_conf:
                report += "\n分类低置信度提示：" + ", ".join(
                    f"{name}={prob}" for name, prob in low_conf.items()
                ) + "\n"

            return report, True

        except Exception as exc:
            return f"推理失败：{exc}", False

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("输入图像为空。")

        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"输入图像形状非法：{image.shape}")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(image)

    def _preprocess(self, image_rgb: np.ndarray, model: DetectMultiBackend, imgsz: int, auto: bool) -> torch.Tensor:
        stride = _stride_value(model)
        imgsz = check_img_size(imgsz, stride)
        img = letterbox(image_rgb, imgsz, stride=stride, auto=auto)[0]
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.fp16 else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _detect(
        self,
        detector: DetectMultiBackend,
        image_rgb: np.ndarray,
        conf_thres: float,
        iou_thres: float,
    ) -> Dict[str, torch.Tensor]:
        inp = self._preprocess(image_rgb, detector, self.imgsz_det, auto=True)

        pred = detector(inp)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)
        det = pred[0] if pred and len(pred) else None
        if det is None or len(det) == 0:
            raise RuntimeError("未检测到手骨关键区域。")

        det[:, :4] = scale_boxes(inp.shape[2:], det[:, :4], image_rgb.shape).round()
        det = det.detach().cpu()

        grouped: Dict[str, List[torch.Tensor]] = {name: [] for name in DETECTION_MAX_COUNTS}

        for row in det:
            cls_id = int(row[5].item())
            cls_name = self._canonical_detect_name(detector, cls_id)
            if cls_name in grouped:
                grouped[cls_name].append(row)

        selected: Dict[str, torch.Tensor] = {}
        error_parts: List[str] = []

        for detect_name, keep_indices, target_names in CROP_RULES:
            boxes = grouped.get(detect_name, [])
            max_count = DETECTION_MAX_COUNTS[detect_name]
            min_needed = max(keep_indices) + 1

            if len(boxes) < min_needed:
                error_parts.append(f"{detect_name}: {len(boxes)}/{min_needed}")
                continue

            # Remove excessive duplicate detections by confidence first,
            # then sort spatially from left to right to recover anatomical order.
            boxes = sorted(boxes, key=lambda x: float(x[4]), reverse=True)[:max_count]
            boxes = sorted(boxes, key=lambda x: float(x[0]))

            for index, target_name in zip(keep_indices, target_names):
                selected[target_name] = boxes[index]

        if error_parts:
            counts = {name: len(items) for name, items in grouped.items()}
            raise RuntimeError(
                "检测结果不足，无法还原 13 个 RUS-CHN ROI；"
                f"缺失/不足：{'; '.join(error_parts)}；"
                f"当前检测计数：{counts}"
            )

        return selected

    def _canonical_detect_name(self, detector: DetectMultiBackend, cls_id: int) -> str:
        names = getattr(detector, "names", None)
        raw_name = None

        if isinstance(names, Mapping):
            raw_name = names.get(cls_id, names.get(str(cls_id)))
        elif isinstance(names, Sequence) and not isinstance(names, str):
            if 0 <= cls_id < len(names):
                raw_name = names[cls_id]

        if raw_name is not None:
            raw_name = str(raw_name)
            if raw_name in DETECTION_MAX_COUNTS:
                return raw_name

        return DEFAULT_DETECT_NAMES.get(cls_id, str(raw_name if raw_name is not None else cls_id))

    def _crop_rois(self, image_rgb: np.ndarray, detections: Mapping[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        h, w = image_rgb.shape[:2]
        crops: Dict[str, np.ndarray] = {}

        for part, row in detections.items():
            x1, y1, x2, y2 = [float(v) for v in row[:4]]

            if self.crop_padding > 0:
                bw = x2 - x1
                bh = y2 - y1
                x1 -= bw * self.crop_padding
                x2 += bw * self.crop_padding
                y1 -= bh * self.crop_padding
                y2 += bh * self.crop_padding

            x1_i = max(0, min(w - 1, int(round(x1))))
            y1_i = max(0, min(h - 1, int(round(y1))))
            x2_i = max(0, min(w, int(round(x2))))
            y2_i = max(0, min(h, int(round(y2))))

            if x2_i <= x1_i or y2_i <= y1_i:
                raise RuntimeError(f"{part} ROI 坐标非法：{(x1_i, y1_i, x2_i, y2_i)}")

            crops[part] = np.ascontiguousarray(image_rgb[y1_i:y2_i, x1_i:x2_i].copy())

        return crops

    def _classify_rois(self, rois: Mapping[str, np.ndarray]) -> Tuple[Dict[str, int], Dict[str, float]]:
        levels: Dict[str, int] = {}
        probabilities: Dict[str, float] = {}

        for part in CATEGORY:
            crop = rois[part]
            model_name = PART_TO_MODEL[part]
            if model_name not in self.classifiers:
                raise RuntimeError(f"缺少分类模型：{model_name}")

            model = self.classifiers[model_name]
            inp = self._preprocess(crop, model, self.imgsz_cls, auto=False)

            logits = model(inp)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if logits.ndim > 2:
                logits = torch.flatten(logits, 1)

            probs = torch.softmax(logits.float(), dim=1)[0]
            idx = int(torch.argmax(probs).item())
            prob = float(probs[idx].item())

            raw_name = self._class_name(model, idx)
            level = self._parse_level(raw_name, idx, part)

            levels[part] = level
            probabilities[part] = prob

        return levels, probabilities

    def _class_name(self, model: DetectMultiBackend, idx: int) -> str:
        names = getattr(model, "names", None)

        if isinstance(names, Mapping):
            value = names.get(idx, names.get(str(idx)))
            return str(value) if value is not None else str(idx)

        if isinstance(names, Sequence) and not isinstance(names, str):
            if 0 <= idx < len(names):
                return str(names[idx])

        return str(idx)

    def _parse_level(self, raw_name: str, idx: int, part: str) -> int:
        """
        Parse YOLOv5 classification label into a 1-based maturation level.

        Supported labels:
        - "MCP_1", "Radius_14"
        - "1", "2", ...
        - "0", "1", ... as zero-based labels
        - fallback: argmax index + 1
        """
        max_level = len(SCORE["boy"][part])
        raw_name = str(raw_name).strip()

        numbers = re.findall(r"\d+", raw_name)
        if numbers:
            value = int(numbers[-1])
            if 1 <= value <= max_level:
                return value
            if value == idx and 0 <= idx < max_level:
                return idx + 1

        if 0 <= idx < max_level:
            return idx + 1

        raise RuntimeError(
            f"{part} 分类标签无法解析为等级：raw_name={raw_name!r}, "
            f"idx={idx}, max_level={max_level}"
        )
