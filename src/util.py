import io
import random
import statistics
import string
from dataclasses import dataclass, astuple
from functools import cmp_to_key
from typing import List, Union, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from fastapi import UploadFile

HANDWRITTEN = 0
TYPESET = 1


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class Measure:
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox


def generate_random_id() -> str:
    rid = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    rid = f"{rid[:3]}-{rid[3:]}"

    return rid


def cmp_measure_bboxes(a: Measure, b: Measure) -> int:
    """Compares bounding boxes of two measures and returns which one should come first"""
    a, b = a.bbox, b.bbox

    if a.x1 >= b.x1 and a.y1 >= b.y1:
        return +1  # a after b
    elif a.x1 < b.x1 and a.y1 < b.y1:
        return -1  # a before b
    else:
        overlap_y = min(a.y2 - b.y1, b.y2 - a.y1) / min(a.y2 - a.y1, b.y2 - b.y1)
        if overlap_y >= 0.5:
            if a.x1 < b.x1:
                return -1
            else:
                return +1
        else:
            if a.x1 < b.x1:
                return +1
            else:
                return -1


def get_geometry(a: BBox, b: BBox) -> Tuple[float, float, float]:
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)

    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)

    if right < left or bottom < top:
        intersection_area = 0
    else:
        intersection_area = (right - left) * (bottom - top)

    return intersection_area, area_a, area_b


def remove_overlapping_measures(measures: List[Measure], thresh=0.7) -> List[Measure]:
    valid = [True] * len(measures)

    for a, measure_a in enumerate(measures):
        for b, measure_b in enumerate(measures):
            if a == b:
                continue

            conf_a = measure_a.confidence
            conf_b = measure_b.confidence

            intersection_area, area_a, area_b = get_geometry(
                measure_a.bbox, measure_b.bbox
            )
            if intersection_area == 0:
                continue

            ioa_a = intersection_area / area_a
            ioa_b = intersection_area / area_b
            iou = intersection_area / float(area_a + area_b - intersection_area)

            if ioa_a > thresh or ioa_b > thresh or iou > thresh:
                if conf_a > conf_b:
                    valid[b] = False
                else:
                    valid[a] = False

    results = []
    for i, is_valid in enumerate(valid):
        if is_valid:
            results.append(measures[i])

    return results


def unify_measures(
    measures: List[Measure],
    page_type: str,
    expand: bool = False,
    trim: bool = False,
    auto: bool = False,
) -> List[Measure]:
    if not any([expand, trim, auto]):
        return measures

    if auto:
        if page_type == "typeset" and auto:
            expand = True
            trim = True
        else:
            expand = False
            trim = False

    system_tops = []
    system_bottoms = []

    cur_bbox = BBox(0, 0, 1, 1)
    cur_system_top = 1
    cur_system_bottom = 0
    measure_num_to_system_idx = []
    for i, measure in enumerate(measures):
        bbox = measure.bbox
        overlap_y = min(bbox.y2 - cur_bbox.y1, cur_bbox.y2 - bbox.y1) / min(
            bbox.y2 - bbox.y1, cur_bbox.y2 - cur_bbox.y1
        )
        if bbox.x1 > cur_bbox.x1 and overlap_y > 0.5:
            cur_system_top = min(cur_system_top, bbox.y1)
            cur_system_bottom = max(cur_system_bottom, bbox.y2)
        else:
            system_tops.append(cur_system_top)
            system_bottoms.append(cur_system_bottom)
            cur_system_top = 1
            cur_system_bottom = 0
        cur_bbox = bbox
        measure_num_to_system_idx.append(len(system_tops))
    system_tops.append(cur_system_top)
    system_bottoms.append(cur_system_bottom)

    # expand measures to unified measure height for each system
    if expand:
        results = []
        for i, measure in enumerate(measures):
            x1, y1, x2, y2 = measure.bbox
            y1 = system_tops[measure_num_to_system_idx[i]]
            y2 = system_bottoms[measure_num_to_system_idx[i]]
            measure.bbox = BBox(x1, y1, x2, y2)

            results.append(measure)
        measures = results

    # remove horizontal overlap from adjacent measures
    if trim:
        c_vals = []
        for i in range(len(measures) - 1):
            if measure_num_to_system_idx[i] == measure_num_to_system_idx[i + 1]:
                ax2 = measures[i].bbox.x2
                bx1 = measures[i + 1].bbox.x1

                if ax2 > bx1:
                    c = (ax2 - bx1) / 2
                    measures[i].bbox.x2 -= c
                    measures[i + 1].bbox.x1 += c
                    c_vals.append(c)
        c_mean = statistics.mean(c_vals)

        # trim also outer edges of first and last measure of each system
        for i in range(1, len(measures) - 1):
            if measure_num_to_system_idx[i] != measure_num_to_system_idx[i + 1]:
                measures[i].bbox.x2 -= c_mean
                measures[i + 1].bbox.x1 += c_mean

        measures[0].bbox.x1 += c_mean
        measures[-1].bbox.x2 -= c_mean

    return measures


def draw_debug_image(img: np.ndarray, measures: List[Measure]) -> bytes:
    pil = Image.fromarray(img).convert("RGBA")
    h, w = img.shape[:2]
    colors = ["#aa75ff", "#1ec7c7"]
    font = ImageFont.load_default()

    for i, measure in enumerate(measures):
        overlay = Image.new("RGBA", pil.size)
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = measure.bbox
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h
        color = colors[measure.class_id]
        draw.rectangle((x1, y1, x2, y2), fill=f"{color}40", outline=color, width=2)

        text = f"{i + 1} ({int(measure.confidence * 100)}%)"
        # textbbox is available in recent Pillow; fallback to approximate box if needed
        try:
            tbx1, tby1, tbx2, tby2 = draw.textbbox((x1, y1), text, font=font)
            tw, th = tbx2 - tbx1, tby2 - tby1
        except Exception:
            tw, th = (len(text) * 6, 10)
        draw.rectangle(
            (x1, y1, x1 + tw + 1, y1 + th + 1), outline=color, fill=color, width=2
        )
        draw.text((x1 + 2, y1 + 1), text, fill="#FFFFFF", font=font)

        pil = Image.alpha_composite(pil, overlay)
    pil = pil.convert("RGB")
    temp = io.BytesIO()
    pil.save(temp, format="jpeg")

    return temp.getvalue()


def detect_measures(
    model,
    file: UploadFile,
    expand: bool = False,
    trim: bool = False,
    auto: bool = False,
    debug: bool = False,
) -> Tuple[Union[List[Measure], bytes], str, float]:
    # load and convert image
    data = file.file.read()
    img: np.ndarray = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if img is None:
        raise ValueError("Invalid image data uploaded")
    img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # predict measures
    with torch.no_grad():
        results = model(img, size=1280).xyxyn[0]

    # restructure results
    measures: List[Measure] = []
    for obj in results:
        measures.append(
            Measure(
                class_id=int(obj[5]),
                class_name=model.model.names[int(obj[5])],
                confidence=round(float(obj[4]), 3),
                bbox=BBox(*[round(i, 5) for i in obj[:4].tolist()]),
            )
        )

    # sort boxes in musically sequential order
    measures.sort(key=cmp_to_key(cmp_measure_bboxes))

    # detect page type
    page_type, type_conf = detect_page_type(measures)

    # post-processing
    measures = remove_overlapping_measures(measures)
    measures = unify_measures(measures, page_type, expand, trim, auto)

    if debug:
        return draw_debug_image(img, measures), page_type, type_conf

    return measures, page_type, type_conf


def detect_page_type(measures: List[Measure]) -> Tuple[str, float]:
    if not measures:
        return "unknown", 0.0

    scores = [0, 0]
    for measure in measures:
        x1, y1, x2, y2 = measure.bbox
        area = (x2 - x1) * (y2 - y1)
        box_class = measure.class_id
        box_score = area * measure.confidence
        scores[box_class] += box_score

    sum_scores = sum(scores)
    if sum_scores == 0:
        return "unknown", 0.0
    scores = [s / sum_scores for s in scores]

    if scores[HANDWRITTEN] > scores[TYPESET]:
        return "handwritten", scores[HANDWRITTEN]
    else:
        return "typeset", scores[TYPESET]


def is_true(val: str) -> bool:
    return val in ["1", "t", "T", "true", "True", "TRUE", "y", "yes", "Yes", "YES"]
