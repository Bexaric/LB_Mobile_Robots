import math
import cv2
import numpy as np

from typing import Tuple


def is_convex_quad(pts: np.ndarray) -> bool:
    """
    Проверяет, образуют ли 4 точки выпуклый четырёхугольник.

    Входы:
        - pts: np.ndarray — координаты четырёх точек.
    Выходы:
        - bool — True, если четырёхугольник выпуклый, иначе False.
    """

    pts = pts.reshape(4, 2)
    signs = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        signs.append(np.sign(cross))
    return all(s == signs[0] for s in signs if s != 0)


def order_points_clockwise(pts: np.ndarray):
    """
    Упорядочивает 4 точки по часовой стрелке начиная с верхнего левого угла.

    Входы:
        - pts: np.ndarray — исходные точки.
    Выходы:
        - rect: np.ndarray — упорядоченные точки: [верх-лево, верх-право, низ-право, низ-лево].
    """

    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def compute_projected_center(
        perspective_matrix: np.ndarray,
        frame_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Возвращает проекцию оптического центра исходного кадра
    на плоскость рабочей области после перспективного преобразования.

    Входы:
        - perspective_matrix: np.ndarray — матрица перспективного преобразования.
        - frame_shape: Tuple[int, int] — (высота, ширина) исходного кадра в пикселях.
    Выходы:
        - warped_center: np.ndarray — координаты [x, y] проекции центра на рабочей области.
    """

    height: int = frame_shape[0]
    width: int = frame_shape[1]

    original_center: np.ndarray = np.array(
        [[[width / 2.0, height / 2.0]]],
        dtype=np.float32
    )
    warped_center: np.ndarray = cv2.perspectiveTransform(
        original_center,
        perspective_matrix
    )

    return warped_center[0][0]


def project_mask(
        mask: np.ndarray,
        H: float,
        h: float,
        center: Tuple[float, float]
) -> np.ndarray:
    """
    Корректирует бинарную маску препятствий с учётом высоты объекта (параллакс).
    Масштабирует маску относительно точки center с коэффициентом k = (H - h)/H.

    Входы:
        - mask: np.ndarray — бинарная маска препятствий (0/255).
        - H: float — высота камеры над полом (мм).
        - h: float — высота препятствия (мм).
        - center: Tuple[float, float] — координаты проекции оптического центра камеры на рабочую область.
    Выходы:
        - np.ndarray — скорректированная маска.
    """

    k: float = (H - h) / H
    cx, cy = center

    M: np.ndarray = np.array([
        [k, 0, cx * (1 - k)],
        [0, k, cy * (1 - k)]
    ], dtype=np.float32)

    return cv2.warpAffine(mask, M, mask.shape[::-1], flags=cv2.INTER_NEAREST)
