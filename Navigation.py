import math
import time
import numpy as np
import heapq
from typing import List, Dict, Optional, Tuple


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


def compute_velocity(current_position,
                     target_position,
                     v_max, tolerance):
    """
    Пропорциональный регулятор скорости к целевой точке.

    Входы:
        - current_position: np.ndarray — текущие координаты [x, y] (пиксели).
        - target_position: np.ndarray — координаты цели [x, y] (пиксели).
        - v_max: float — максимальная скорость (пикселей/с).
        - tolerance: float — расстояние остановки (пиксели).
    Выходы:
        - velocity: np.ndarray — вектор скорости [vx, vy] (пикселей/с).
        - distance: float — расстояние до цели (пиксели).
    """

    error = target_position.astype(np.float32) - current_position.astype(np.float32)
    distance = float(np.linalg.norm(error))

    if distance <= tolerance:
        return np.zeros(2, dtype=np.float32), distance

    velocity = (error / distance) * v_max

    return velocity.astype(np.float32), distance


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Евклидово расстояние между двумя ячейками сетки.

    Входы:
        - a: Tuple[int, int] — первая ячейка.
        - b: Tuple[int, int] — вторая ячейка.
    Выходы:
        - float — Евклидово расстояние в единицах размера ячейки.
    """

    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Восстанавливает путь от старта к цели по словарю предков.

    Входы:
        - came_from: Dict[Tuple[int, int], Tuple[int, int]] словарь {узел: предшественник}.
        - start: Tuple[int, int] — стартовый узел.
        - goal: Tuple[int, int] — конечный узел.
    Выходы:
        - path: List[Tuple[int, int]] — Список узлов от start до goal включительно.
    """

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def astar(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        allow_diagonal: bool = True
) -> List[Tuple[int, int]]:
    """
    Реализация алгоритма A* для двумерной сетки.

    Входы:
        - grid: np.ndarray — массив, где 0 — свободная ячейка, 1 — препятствие.
        - start: Tuple[int, int] — стартовая ячейка.
        - goal: Tuple[int, int] — целевая ячейка.
        - allow_diagonal: bool — если True, используются 8 соседей (диагонали разрешены).
    Выходы:
        - List[Tuple[int, int]] — Список ячеек оптимального пути.
    """

    if grid[start[0], start[1]] != 0 or grid[goal[0], goal[1]] != 0:
        return []

    H, W = grid.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr, nc] != 0:
                continue

            move_cost = math.hypot(dr, dc) if allow_diagonal else 1.0
            tentative_g = g_score[current] + move_cost
            neighbor = (nr, nc)

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[neighbor] = current

    return []


def dijkstra(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Алгоритм Дейкстры (частный случай A* без эвристики).

    Входы:
        - grid: np.ndarray — массив, где 0 — свободная ячейка, 1 — препятствие.
        - start: Tuple[int, int] — стартовая ячейка.
        - goal: Tuple[int, int] — целевая ячейка.
    Выходы:
        - List[Tuple[int, int]] — Список ячеек кратчайшего пути.
    """

    if grid[start[0], start[1]] != 0 or grid[goal[0], goal[1]] != 0:
        return []

    H, W = grid.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        g, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr, nc] != 0:
                continue
            tentative_g = g + 1.0
            neighbor = (nr, nc)
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))
                came_from[neighbor] = current
    return []


def greedy_best_first(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Жадный поиск по наилучшему соответствию (использует только эвристику).
    Не гарантирует оптимальность пути.

    Входы:
        - grid: np.ndarray — массив, где 0 — свободная ячейка, 1 — препятствие.
        - start: Tuple[int, int] — стартовая ячейка.
        - goal: Tuple[int, int] — целевая ячейка.

    Выходы:
        - List[Tuple[int, int]] — Список ячеек пути.
    """
    if grid[start[0], start[1]] != 0 or grid[goal[0], goal[1]] != 0:
        return []

    H, W = grid.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr, nc] != 0:
                continue
            neighbor = (nr, nc)
            if neighbor not in visited:
                h = heuristic(neighbor, goal)
                heapq.heappush(open_set, (h, neighbor))
                came_from[neighbor] = current
    return []


def _merge_paths(
        meeting: Tuple[int, int],
        came_from_fwd: Dict,
        came_from_bwd: Dict,
        start: Tuple[int, int],
        goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Объединяет части путей от точки встречи.

    Входы:
        - meeting: Tuple[int, int] — точка встречи прямого и обратного поиска.
        - came_from_fwd: Tuple[int, int] — словарь предков прямого поиска.
        - came_from_bwd: Tuple[int, int] — словарь предков обратного поиска.
        - start: Tuple[int, int] — стартовая ячейка.
        - goal: Tuple[int, int] — целевая ячейка.
    Выходы:
        - path: List[Tuple[int, int]] — Список ячеек пути от start до goal.
    """
    path = []
    # От встречи к старту
    cur = meeting
    while cur != start:
        path.append(cur)
        cur = came_from_fwd[cur]
    path.append(start)
    path.reverse()
    # От встречи к цели
    cur = meeting
    while cur != goal:
        cur = came_from_bwd[cur]
        path.append(cur)
    return path


def bidirectional_astar(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Двунаправленный A* (одновременный поиск от старта и цели).

    Входы:
        - grid: np.ndarray — массив, где 0 — свободная ячейка, 1 — препятствие.
        - start: Tuple[int, int] — стартовая ячейка.
        - goal: Tuple[int, int] — целевая ячейка.
    Выходы:
        - List[Tuple[int, int]] — Список ячеек оптимального пути.
    """

    if grid[start[0], start[1]] != 0 or grid[goal[0], goal[1]] != 0:
        return []

    H, W = grid.shape

    open_fwd = [(0.0, start)]
    open_bwd = [(0.0, goal)]
    came_from_fwd = {}
    came_from_bwd = {}
    g_fwd = {start: 0.0}
    g_bwd = {goal: 0.0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_fwd and open_bwd:
        # Расширяем прямой поиск
        if open_fwd:
            _, current_fwd = heapq.heappop(open_fwd)
            if current_fwd in g_bwd:
                return _merge_paths(current_fwd, came_from_fwd, came_from_bwd, start, goal)

            for dr, dc in directions:
                nr, nc = current_fwd[0] + dr, current_fwd[1] + dc
                if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] != 0:
                    continue
                tentative = g_fwd[current_fwd] + 1.0
                neighbor = (nr, nc)
                if tentative < g_fwd.get(neighbor, float('inf')):
                    g_fwd[neighbor] = tentative
                    f = tentative + heuristic(neighbor, goal)
                    heapq.heappush(open_fwd, (f, neighbor))
                    came_from_fwd[neighbor] = current_fwd

        # Расширяем обратный поиск
        if open_bwd:
            _, current_bwd = heapq.heappop(open_bwd)
            if current_bwd in g_fwd:
                return _merge_paths(current_bwd, came_from_fwd, came_from_bwd, start, goal)

            for dr, dc in directions:
                nr, nc = current_bwd[0] + dr, current_bwd[1] + dc
                if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] != 0:
                    continue
                tentative = g_bwd[current_bwd] + 1.0
                neighbor = (nr, nc)
                if tentative < g_bwd.get(neighbor, float('inf')):
                    g_bwd[neighbor] = tentative
                    f = tentative + heuristic(neighbor, start)
                    heapq.heappush(open_bwd, (f, neighbor))
                    came_from_bwd[neighbor] = current_bwd

    return []


def find_nearest_free(
        grid: np.ndarray,
        start: Tuple[int, int],
        max_radius: int = 50
) -> Optional[Tuple[int, int]]:
    """
    Поиск ближайшей свободной ячейки вокруг занятой стартовой точки.

    Входы:
        - grid: np.ndarray — массив, где 1 — свободно, 0 — занято.
        - start: Tuple[int, int] — центральная точка поиска.
        - max_radius: int — максимальный радиус поиска.
    Выходы:
        - (ny, nx): Optional[Tuple[int, int]] — ближайшая свободная ячейка.
    """
    h, w = grid.shape
    y0, x0 = start
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dy) == r or abs(dx) == r:
                    ny, nx = y0 + dy, x0 + dx
                    if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 1:
                        return ny, nx
    return None


def get_lookahead_point(
        robot_pos: np.ndarray,
        path: list,
        lookahead_dist: float
) -> np.ndarray:
    """
    Возвращает точку на глобальном пути на расстоянии lookahead_dist.
    Увеличивает ошибку следования, но сглаживает движение.

    Входы:
        - robot_pos: np.ndarray — координаты робота [x, y] (пкс).
        - path: list — список точек запланированного пути [(x, y), ...] (пкс).
        - lookahead_dist: list — желаемое расстояние вдоль пути (пкс).
    Выходы:
        - np.ndarray — координаты [x, y] искомой точки (пск).
    """

    if not path:
        return robot_pos

    # Найти ближайшую точку
    path_arr = np.array(path)
    dists = np.linalg.norm(path_arr - robot_pos, axis=1)
    nearest_idx = int(np.argmin(dists))

    accumulated = 0.0
    for i in range(nearest_idx, len(path) - 1):
        seg_start = path_arr[i]
        seg_end = path_arr[i + 1]
        seg_len = np.linalg.norm(seg_end - seg_start)
        if accumulated + seg_len >= lookahead_dist:
            remaining = lookahead_dist - accumulated
            direction = (seg_end - seg_start) / seg_len
            lookahead_point = seg_start + direction * remaining
            return lookahead_point.astype(np.float32)
        accumulated += seg_len
    return path_arr[-1].astype(np.float32)


def compute_metrics(
        start_time: float,
        trajectory: List[np.ndarray],
        planned_path: List[Tuple[float, float]],
        pixel_per_mm: float
) -> dict:
    """
    Вычисляет метрики движения после завершения поездки.

    Входы:
        - start_time: float — время старта движения.
        - trajectory: List[np.ndarray] — список позиций робота [x, y] (пкс).
        - planned_path: List[Tuple[float, float]] — список точек запланированного пути (x, y) (пкс).
        - pixel_per_mm: float — коэффициент перевода пикселей в миллиметры.
    Выходы:
        metrics: dict — словарь с метриками:
            - time_s: float — общее время (сек).
            - planned_length_mm: float — длина запланированного пути (мм).
            - actual_length_mm: float — фактически пройденное расстояние (мм).
            - mse_mm2: float — среднеквадратичное отклонение от плана (мм²).
            - r_squared: float — коэффициент детерминации (безразмерный).
            - num_points: int — количество точек в траектории.
    """

    elapsed = time.time() - start_time

    planned_length_px = 0.0
    if planned_path and len(planned_path) > 1:
        for i in range(len(planned_path) - 1):
            p1 = np.array(planned_path[i])
            p2 = np.array(planned_path[i + 1])
            planned_length_px += np.linalg.norm(p2 - p1)
    planned_length_mm = planned_length_px / pixel_per_mm

    # Фактическая длина траектории
    actual_length_px = 0.0
    if trajectory and len(trajectory) > 1:
        for i in range(len(trajectory) - 1):
            actual_length_px += np.linalg.norm(trajectory[i + 1] - trajectory[i])
    actual_length_mm = actual_length_px / pixel_per_mm

    # MSE
    mse_px2 = 0.0
    n = len(trajectory)
    if n > 0 and planned_path:
        planned_arr = np.array(planned_path)
        for pt in trajectory:
            dist = np.min(np.linalg.norm(planned_arr - pt, axis=1))
            mse_px2 += dist * dist
        mse_px2 /= n
    mse_mm2 = mse_px2 / (pixel_per_mm ** 2)

    # R²
    r_squared = None
    if n > 1 and planned_path:
        ss_res = 0.0
        planned_arr = np.array(planned_path)
        for pt in trajectory:
            dist = np.min(np.linalg.norm(planned_arr - pt, axis=1))
            ss_res += dist * dist
        mean_traj = np.mean(trajectory, axis=0)
        ss_tot = np.sum(np.linalg.norm(trajectory - mean_traj, axis=1) ** 2)
        if ss_tot > 1e-12:
            r_squared = 1.0 - ss_res / ss_tot
        else:
            r_squared = 1.0

    metrics = {
        'time_s': round(elapsed, 3),
        'planned_length_mm': round(planned_length_mm, 2),
        'actual_length_mm': round(actual_length_mm, 2),
        'mse_mm2': round(mse_mm2, 3),
        'r_squared': round(r_squared, 4) if r_squared is not None else None,
        'num_points': n
    }
    return metrics
