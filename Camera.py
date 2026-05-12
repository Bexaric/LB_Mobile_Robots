import cv2
import numpy as np

from typing import Tuple


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
