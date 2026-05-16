import math
import time
from typing import Tuple, Optional

import cv2
import cv2.aruco as aruco
import numpy as np
import yaml

from Camera import (is_convex_quad, order_points_clockwise,
                    compute_projected_center, project_mask)
from Navigation import (compute_velocity, astar,
                        get_lookahead_point, find_nearest_free,
                        dijkstra, greedy_best_first,
                        bidirectional_astar, compute_metrics)
from Robotino import connect_to_robotino, send_velocity

with open('parameters.yaml') as config_file:
    config = yaml.safe_load(config_file)
print(config)

output_size: Optional[Tuple[int, int]] = None         # размер рабочей области [ширина, высота] (пкс)
start_time_task: float | None = None                  # время старта текущего маршрута (сек)

# Состояние интерфейса
mode: str = 'calibrate'                               # текущий режим интерфейса: 'calibrate' и 'working'
calib_points: list = []                               # координаты четырёх углов рабочей области
perspective_matrix: Optional[np.ndarray] = None       # матрица перспективного преобразования
trans_center: Optional[Tuple[float, float]] = None    # проекция оптического центра камеры на рабочую область
frame_shape: Optional[Tuple[int, int]] = None         # размеры исходного кадра [ширина, высота]
sharpening_kernel: Optional[np.float32] = None        # ядро для повышения резкости

motion_started: bool = False                          # флаг активности движения
global_path: list = []                                # запланированный глобальный путь (список точек) (пкс)
trajectory: list[np.ndarray] = []                     # фактическая траектория движения (список точек) (пкс)
last_completed_path: Optional[dict] = None            # последний завершённый маршрут (список точек) (пкс)
start_point: Optional[np.ndarray] = None              # координаты точки начала движения (пкс)
click_point: np.ndarray | None = None                 # координаты целевой точки (пкс)
last_click_point: np.ndarray | None = None            # предыдущая целевая точка (пкс)

center_x: int = 0                                     # X-координата центра робота (пкс)
center_y: int = 0                                     # Y-координата центра робота (пкс)
dx: float = 0.0                                       # разность X между двумя углами маркера (пкс)
dy: float = 0.0                                       # разность Y между двумя углами маркера (пкс)
angle: float = 0.0                                    # угол ориентации робота

# Автоматический режим
auto_mode_enable: bool = bool(config['auto_mode']['enable'])
auto_start: np.ndarray = np.array([config['auto_mode']['start_x'], config['auto_mode']['start_y']], dtype=np.float32)
auto_end: np.ndarray = np.array([config['auto_mode']['end_x'], config['auto_mode']['end_y']], dtype=np.float32)
auto_algorithms = ["astar", "dijkstra", "greedy", "bdastar"]
auto_phase: str = 'to_start'
auto_algo_idx: int = 0
auto_last_path: list = []
auto_trajectories: list = []

# Параметры с файла "parameters.yaml"
robot_status: bool = config['socket_params']['enable']

H: float = config['camera']['H']
h: float = config['camera']['h']
offset_x: float = config['camera']['offset_x']
offset_y: float = config['camera']['offset_y']
wall_width: int = config['camera']['wall_width']
extra_offset_red: int = config['camera']['add_zone_red']
extra_offset_green: int = config['camera']['add_zone_green']
extra_offset_blue: int = config['camera']['add_zone_blue']
scale: float = config['camera']['scale_factor']
scale_algorithm: float = config['camera']['scale_factor_mask']
calibration_display_scale: float = config['camera']['calibration_display_scale']
working_display_scale: int = config['camera']['working_display_scale']
sharpening: bool = config['camera']['sharpening']
camera_status: bool = config['camera']['online']
parallax: bool = config['camera']['parallax']
video_stream: int = config['camera']['video_stream']

resolution: int = config['map_params']['resolution']
algorithm: str = config['map_params']['algorithm']
dynamic_update: bool = config['map_params']['dynamic_update']
smoothing: float = config['map_params']['smoothing']
acceptable_error: float = config['map_params']['acceptable_error']

robot_radius: float = config['robot']['radius']
v_max: float = config['robot']['max_speed']


def mouse_callback(
        event: int,
        x: int,
        y: int,
        flags: int,
        param
) -> None:
    """
    Обработчик событий мыши для калибровки области и задания целевой точки.

    Входы:
        - event: int — тип события OpenCV (cv2.EVENT_LBUTTONDOWN и др.)
        - x, y: int — координаты курсора в окне (пкс)
        - flags: int — дополнительные флаги (не используются)
        - param: пользовательские данные (не используются)

    Режим 'calibrate':
        - Собирает 4 клика (углы рабочей области).
        - Проверяет, что точки образуют выпуклый четырёхугольник.
        - Упорядочивает точки по часовой стрелке.
        - Вычисляет матрицу перспективы и проекцию центра кадра.
        - Переключает интерфейс в рабочий режим ('working').

    Режим 'working':
        - При левом клике задаёт целевую точку для движения робота.
        - Координаты клика масштабируются и ограничиваются границами
          рабочей зоны.
        - Сохраняет цель в глобальную переменную click_point,
          сбрасывает траекторию и запускает движение.
    """

    # Создание глобальных переменных
    global mode, calib_points, perspective_matrix, trans_center
    global click_point, motion_started, start_time_task, trajectory
    global output_size, start_point, last_completed_path
    global offset_x, offset_y

    if auto_mode_enable and mode == 'working':
        return

    if mode == 'calibrate':
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x = x / calibration_display_scale
            real_y = y / calibration_display_scale

            if len(calib_points) < 4:
                calib_points.append((real_x, real_y))
                idx = len(calib_points)
                print(f"Угол {idx} отмечен")

                if idx == 4:
                    pts = np.array(calib_points, dtype=np.float32)
                    if not is_convex_quad(pts):
                        print("Точки не образуют выпуклый четырёхугольник")
                        calib_points.clear()
                        return

                    # Упорядочиваем точки
                    ordered_pts = order_points_clockwise(pts)
                    global output_size
                    dst_ordered = np.float32([
                        [0, 0],
                        [output_size[0], 0],
                        [output_size[0], output_size[1]],
                        [0, output_size[1]]
                    ])
                    perspective_matrix = cv2.getPerspectiveTransform(ordered_pts, dst_ordered)

                    # Проекция центра кадра
                    if frame_shape is not None:
                        orig_center_in_area = compute_projected_center(
                            perspective_matrix=perspective_matrix,
                            frame_shape=frame_shape
                        )
                        trans_center = (orig_center_in_area[1] + offset_x, orig_center_in_area[0] + offset_y)
                        print(orig_center_in_area[1], orig_center_in_area[0])
                    else:
                        trans_center = (output_size[0] // 2, output_size[1] // 2)

                    mode = 'working'
                    print("Калибровка завершена.")

    elif mode == 'working':
        if event == cv2.EVENT_LBUTTONDOWN:
            raw_x = x * working_display_scale
            raw_y = y * working_display_scale

            # Ограничение по границам рабочей зоны
            output_size = (resolution, resolution)
            max_x = output_size[0] - wall_width - 1
            max_y = output_size[1] - wall_width - 1
            clamped_x = max(wall_width, min(raw_x, max_x))
            clamped_y = max(wall_width, min(raw_y, max_y))

            click_point = np.array([clamped_y, clamped_x], dtype=np.int32)
            start_point = None
            last_completed_path = None
            motion_started = True
            start_time_task = time.time()
            trajectory.clear()
            print(f"Целевая точка: {click_point}")


def main():
    # Создание глобальных переменных
    global mode, calib_points, trans_center, center_y, center_x, dx, dy, angle, sharpening_kernel
    global robot_position, auto_pause_until, auto_trajectories, auto_phase, auto_algo_idx, auto_last_path
    global output_size, click_point, motion_started, start_time_task, trajectory, start_point
    global auto_mode_enable, auto_phase, auto_algo_idx, auto_last_path, algorithm, click_point
    global motion_started, start_time_task, trajectory, start_point
    global frame_shape, global_path, last_click_point, last_completed_path, video_stream

    # Подключение к Robotino
    sock = None
    if robot_status:
        sock = connect_to_robotino()
        if sock is None:
            print("Не удалось подключиться к роботу.")
            return
        print("Успешное подключение.")
    else:
        print("Режим работы без робота")

    output_size = (resolution, resolution)

    # Создание фильтра для резкости
    if sharpening:
        sharpening_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    # Подключение к камере или к видео
    if camera_status:
        cap = cv2.VideoCapture(video_stream)
    else:
        # "video_photo_space\vid.mp4"
        # "video_photo_space\WIN_20260508_21_24_57_Pro.mp4"
        cap = cv2.VideoCapture(
            r"video_photo_space\vid.mp4")
    if not cap.isOpened():
        raise RuntimeError("Ошибка воспроизведения")

    # Aruco-маркер на Robotino
    aruco_6x6 = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_6X6_100),
        aruco.DetectorParameters()
    )

    kernel_denoise: np.ndarray = np.ones((10, 10), np.uint8)

    cv2.namedWindow("map_planner")
    cv2.setMouseCallback('map_planner', mouse_callback)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Повышаем резкость изображения
            if sharpening_kernel is not None:
                frame = cv2.filter2D(frame, -1, sharpening_kernel)

            # Сохраняем форму кадра
            if frame_shape is None:
                frame_shape = frame.shape[:2]

            # Выделение границ карты
            if mode == 'calibrate':
                for i, pt in enumerate(calib_points):
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (int(pt[0]) + 5, int(pt[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                disp = cv2.resize(frame, None, fx=calibration_display_scale, fy=calibration_display_scale)
                cv2.imshow("map_planner", disp)
                cv2.resizeWindow("map_planner", disp.shape[1], disp.shape[0])
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                continue

            # Выпрямление изображения (аффинное преобразование)
            working_area = cv2.warpPerspective(frame, perspective_matrix, output_size)

            # Определение позиции робота
            corners, ids, _ = aruco_6x6.detectMarkers(working_area)

            if ids is not None:
                c = corners[0][0]
                center_x = int(np.mean(c[:, 1]))
                center_y = int(np.mean(c[:, 0]))
                robot_position = np.array([center_x, center_y], dtype=np.float32)
                dx = c[1][0] - c[0][0]
                dy = c[1][1] - c[0][1]
                angle = math.atan2(dx, dy)

            # Работа с масками (в HSV)
            hsv = cv2.cvtColor(working_area, cv2.COLOR_BGR2HSV)

            # Создание
            mask_red_1 = cv2.inRange(hsv,
                                   np.array([0, 100, 135], dtype=np.uint8),
                                   np.array([5, 255, 255], dtype=np.uint8))

            mask_red_2 = cv2.inRange(hsv,
                                   np.array([160, 100, 135], dtype=np.uint8),
                                   np.array([200, 255, 255], dtype=np.uint8))

            mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

            mask_green = cv2.inRange(hsv,
                                     np.array([40, 100, 135], dtype=np.uint8),
                                     np.array([100, 255, 255], dtype=np.uint8))

            mask_blue = cv2.inRange(hsv,
                                    np.array([110, 100, 135], dtype=np.uint8),
                                    np.array([160, 255, 255], dtype=np.uint8))

            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_denoise)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_denoise)
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel_denoise)

            # Проекция маски (для учёта высоты камеры)
            if parallax:
                mask_red = project_mask(mask_red,
                                        H=H,
                                        h=h,
                                        center=trans_center)

                mask_blue = project_mask(mask_blue,
                                         H=H,
                                         h=h,
                                         center=trans_center)

                mask_green = project_mask(mask_green,
                                          H=H,
                                          h=h,
                                          center=trans_center)

            mask_2 = cv2.bitwise_or(mask_red, mask_blue)

            mask = cv2.bitwise_or(mask_2, mask_green)

            raw_obstacle_mask = mask.copy()

            mask_copy_red = mask_red.copy()
            mask_copy_green = mask_green.copy()
            mask_copy_blue = mask_blue.copy()

            # Границы зоны езды
            mask[:wall_width, :] = 255
            mask[-wall_width:, :] = 255
            mask[:, :wall_width] = 255
            mask[:, -wall_width:] = 255

            # Масштабирование под размер поля
            pixel_per_mm = output_size[0] / resolution
            robot_radius_px = int(robot_radius * pixel_per_mm)
            smoothing_px = int(smoothing * pixel_per_mm)
            acceptable_error_px = int(acceptable_error * pixel_per_mm)

            # Очистка масок внутри зоны робота
            if ids is not None and robot_radius_px > 0:
                cv2.circle(mask, (center_y, center_x), robot_radius_px, 0, -1)

            if robot_radius_px > 0:

                small_w_r = int(mask_copy_red.shape[1] * scale)
                small_h_r = int(mask_copy_red.shape[0] * scale)

                small_w_g = int(mask_copy_green.shape[1] * scale)
                small_h_g = int(mask_copy_green.shape[0] * scale)

                small_w_b = int(mask_copy_blue.shape[1] * scale)
                small_h_b = int(mask_copy_blue.shape[0] * scale)

                small_raw_r = cv2.resize(mask_copy_red, (small_w_r, small_h_r), interpolation=cv2.INTER_NEAREST)
                small_raw_g = cv2.resize(mask_copy_green, (small_w_g, small_h_g), interpolation=cv2.INTER_NEAREST)
                small_raw_b = cv2.resize(mask_copy_blue, (small_w_b, small_h_b), interpolation=cv2.INTER_NEAREST)

                # Расширение маски препятствия на r_small
                r_small_red = max(1, int(robot_radius_px * scale + extra_offset_red))
                r_small_green = max(1, int(robot_radius_px * scale + extra_offset_green))
                r_small_blue = max(1, int(robot_radius_px * scale + extra_offset_blue))

                kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (2 * r_small_red + 1, 2 * r_small_red + 1))

                kernel_green = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (2 * r_small_green + 1, 2 * r_small_green + 1))

                kernel_blue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (2 * r_small_blue + 1, 2 * r_small_blue + 1))

                small_dilated_r = cv2.dilate(small_raw_r, kernel_red)
                small_dilated_g = cv2.dilate(small_raw_g, kernel_green)
                small_dilated_b = cv2.dilate(small_raw_b, kernel_blue)

                dilated_only_obstacles_r = cv2.resize(small_dilated_r,
                                                      (mask_copy_red.shape[1], mask_copy_red.shape[0]),
                                                      interpolation=cv2.INTER_NEAREST)

                dilated_only_obstacles_g = cv2.resize(small_dilated_g,
                                                      (mask_copy_green.shape[1], mask_copy_green.shape[0]),
                                                      interpolation=cv2.INTER_NEAREST)

                dilated_only_obstacles_b = cv2.resize(small_dilated_b,
                                                      (mask_copy_blue.shape[1], mask_copy_blue.shape[0]),
                                                      interpolation=cv2.INTER_NEAREST)

                dilated_only_obstacles_2 = cv2.bitwise_or(dilated_only_obstacles_r, dilated_only_obstacles_g)
                dilated_only_obstacles = cv2.bitwise_or(dilated_only_obstacles_2, dilated_only_obstacles_b)

            else:
                dilated_only_obstacles = raw_obstacle_mask.copy()

            # Нахождение контуров масок препятствий
            contours_raw, _ = cv2.findContours(raw_obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(working_area, contours_raw, -1, (0, 255, 0), 2)

            contours_red, _ = cv2.findContours(dilated_only_obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(working_area, contours_red, -1, (0, 0, 255), 3)

            # Создание маски для алгоритмов нахождения пути
            mask_for_algorithm = cv2.bitwise_or(mask, dilated_only_obstacles)

            # Граница стен
            cv2.rectangle(working_area,
                          (wall_width, wall_width),
                          (output_size[0] - wall_width, output_size[1] - wall_width),
                          (0, 0, 255), 2)

            # Отображение робота
            if ids is not None:
                cv2.circle(working_area, (center_y, center_x), 25, (0, 255, 255), -1)
                cv2.circle(working_area, (center_y, center_x), robot_radius_px, (0, 0, 0), 8)

                # Оси координат на роботе (относительно позиции Aruco)
                axis_len = 250
                norm_fwd = np.linalg.norm([dx, dy])
                if norm_fwd > 0:
                    fwd_x = dx / norm_fwd
                    fwd_y = dy / norm_fwd

                    # Ось Y
                    end_y = (int(center_y + axis_len * fwd_x),
                             int(center_x + axis_len * fwd_y))
                    cv2.arrowedLine(working_area, (center_y, center_x), end_y,
                                    (0, 0, 255), 8, tipLength=0.15)
                    cv2.putText(working_area, "Y", (end_y[0] + 10, end_y[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

                    # Ось X
                    side_x = -fwd_y
                    side_y = fwd_x
                    end_x = (int(center_y + axis_len * side_x),
                             int(center_x + axis_len * side_y))
                    cv2.arrowedLine(working_area, (center_y, center_x), end_x,
                                    (255, 0, 0), 8, tipLength=0.15)
                    cv2.putText(working_area, "X", (end_x[0] + 10, end_x[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 8)

                # Глобальная система координат
                origin_global = (100, 100)
                axis_len_global = 200

                # Ось X
                cv2.arrowedLine(working_area, origin_global,
                                (origin_global[0] + axis_len_global, origin_global[1]),
                                (0, 0, 0), 8, tipLength=0.15)
                cv2.putText(working_area, "X",
                            (origin_global[0] + axis_len_global + 5, origin_global[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
                # Ось Y
                cv2.arrowedLine(working_area, origin_global,
                                (origin_global[0], origin_global[1] + axis_len_global),
                                (0, 0, 0), 8, tipLength=0.15)
                cv2.putText(working_area, "Y",
                            (origin_global[0] - 16, origin_global[1] + axis_len_global + 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)

                # Автоматический режим (ЛБ2)
                if auto_mode_enable:
                    if global_path:
                        auto_last_path = global_path.copy()

                    if auto_phase == 'to_start':
                        click_point = np.array([auto_start[1], auto_start[0]])
                        algorithm = "astar"

                        if not motion_started:
                            motion_started = True
                            start_time_task = time.time()
                            start_point = robot_position.copy()

                        if np.linalg.norm(robot_position - auto_start[::-1]) < acceptable_error_px:
                            auto_phase = 'to_target'
                            auto_algo_idx = 0
                            motion_started = False
                            auto_pause_until = time.time() + 5.0
                            click_point = np.array([auto_end[1], auto_end[0]])
                            algorithm = auto_algorithms[auto_algo_idx]
                            trajectory.clear()
                            print(f"Начало тестирования: {algorithm}")

                    elif auto_phase == 'to_target':
                        if time.time() < auto_pause_until:
                            motion_started = False
                        else:
                            if not motion_started:
                                motion_started = True
                                start_time_task = time.time()
                                start_point = robot_position.copy()
                        if np.linalg.norm(robot_position - auto_end[::-1]) < acceptable_error_px:
                            if trajectory:
                                path_for_metrics = auto_last_path if auto_last_path else global_path
                                metrics = compute_metrics(start_time_task, trajectory.copy(), path_for_metrics.copy(),
                                                          pixel_per_mm)
                                print(f"Метрики движения ({algorithm}):")
                                print(f"Время прохождения маршрута: {metrics['time_s']} с")
                                print(f"Планируемый путь: {metrics['planned_length_mm']} мм")
                                print(f"Пройденный путь: {metrics['actual_length_mm']} мм")
                                print(f"MSE (мм²): {metrics['mse_mm2']}")
                                if metrics['r_squared'] is not None:
                                    print(f"R² (коэффициент детерминации): {metrics['r_squared']}")
                                print(f"Количество точек пути: {metrics['num_points']}")

                            auto_trajectories.append((algorithm, trajectory.copy()))
                            auto_phase = 'to_return'
                            motion_started = False
                            auto_pause_until = time.time() + 2.0
                            click_point = np.array([auto_start[1], auto_start[0]])
                            algorithm = "astar"
                            trajectory.clear()
                            print("\nВозврат на старт (astar)\n")

                    elif auto_phase == 'to_return':
                        if time.time() < auto_pause_until:
                            motion_started = False
                        else:
                            if not motion_started:
                                motion_started = True
                                start_time_task = time.time()
                                start_point = robot_position.copy()

                        if np.linalg.norm(robot_position - auto_start[::-1]) < acceptable_error_px:
                            auto_algo_idx += 1
                            if auto_algo_idx < len(auto_algorithms):
                                auto_phase = 'to_target'
                                motion_started = False
                                auto_pause_until = time.time() + 2.0
                                algorithm = auto_algorithms[auto_algo_idx]
                                click_point = np.array([auto_end[1], auto_end[0]])
                                trajectory.clear()
                                print(f"Тестирование: {algorithm}")
                            else:
                                auto_phase = 'finished'
                                motion_started = False
                                click_point = None
                                start_point = None
                                global_path = []
                                trajectory.clear()
                                last_completed_path = None

                    elif auto_phase == 'finished':
                        motion_started = False
                        click_point = None

            # Планирование маршрута с учётом препятствий
            if ids is not None and click_point is not None:
                if dynamic_update or (last_click_point is None or
                                      not np.array_equal(click_point, last_click_point) or not global_path):
                    last_click_point = click_point.copy()

                    small_mask = cv2.resize(np.transpose(mask_for_algorithm),
                                            (int(mask_for_algorithm.shape[1] * scale_algorithm),
                                             int(mask_for_algorithm.shape[0] * scale_algorithm)),
                                            interpolation=cv2.INTER_NEAREST)

                    # Создание двух масок для работы алгоритмов
                    grid_mask = (small_mask != 255).astype(np.uint8)
                    grid_algorithm = (small_mask == 255).astype(np.uint8)

                    start_small = (int(center_y * scale_algorithm), int(center_x * scale_algorithm))
                    goal_small = (int(click_point[1] * scale_algorithm), int(click_point[0] * scale_algorithm))

                    # Построение маршрута для выхода из зоны препятствия
                    if (0 <= start_small[0] < grid_mask.shape[0] and
                            0 <= start_small[1] < grid_mask.shape[1]):
                        if grid_mask[start_small] == 0:
                            nearest = find_nearest_free(grid_mask, start_small, max_radius=50)
                            if nearest is not None:
                                start_small = nearest
                            else:
                                global_path = []
                                continue

                    # Построение маршрута по одному из алгоритмов
                    if (0 <= start_small[0] < grid_algorithm.shape[0] and 0 <= start_small[1] < grid_algorithm.shape[
                        1] and
                            0 <= goal_small[0] < grid_algorithm.shape[0] and 0 <= goal_small[1] < grid_algorithm.shape[
                                1]):
                        new_path = []
                        if algorithm == "astar":
                            new_path = astar(grid_algorithm, start_small, goal_small)
                        elif algorithm == "dijkstra":
                            new_path = dijkstra(grid_algorithm, start_small, goal_small)
                        elif algorithm == "greedy":
                            new_path = greedy_best_first(grid_algorithm, start_small, goal_small)
                        elif algorithm == "bdastar":
                            new_path = bidirectional_astar(grid_algorithm, start_small, goal_small)

                        if new_path:
                            global_path = [(pt[1] / scale_algorithm, pt[0] / scale_algorithm) for pt in new_path]
                        else:
                            global_path = []
                    else:
                        global_path = []

            # Разрешаем движение, если есть путь
            if ids is not None and click_point is not None:
                if global_path:
                    motion_started = True
                else:
                    motion_started = False

            # Управление движением робота
            if ids is not None:
                robot_position = np.array((center_x, center_y), dtype=np.float32)

                # Запоминаем начальную точку в момент клика
                if click_point is not None and start_point is None:
                    start_point = robot_position.copy()

                # Если цель ещё не задана — остаёмся на месте
                if click_point is None:
                    click_point = robot_position.copy()

                # Присвоение target_waypoint точек из global_path
                if global_path:
                    target_waypoint = get_lookahead_point(robot_position,
                                                          global_path,
                                                          smoothing_px)
                else:
                    target_waypoint = click_point

                # Вычисление максимальной скорости
                v_max_px = v_max * pixel_per_mm

                # Расчёт скорости и расстояния до точки
                v_att, dist_to_target = compute_velocity(robot_position,
                                                         target_waypoint,
                                                         v_max=v_max_px,
                                                         tolerance=1.0)

                # Расчёт результирующей скорости
                velocity = v_att
                speed = float(np.linalg.norm(velocity))

                # Остановка при достижении цели
                dist_to_goal = np.linalg.norm(robot_position - click_point)
                if dist_to_goal < acceptable_error_px and auto_mode_enable == 0:
                    motion_started = False

                    # Сохраняем завершённый маршрут для отображения
                    if global_path and trajectory:
                        last_completed_path = {
                            'planned': global_path.copy(),
                            'actual': trajectory.copy(),
                            'start': start_point.copy() if start_point is not None else None,
                            'end': click_point.copy() if click_point is not None else None
                        }

                    # Сбор метрик
                    if not dynamic_update:
                        if start_time_task is not None and trajectory and global_path:
                            traj_copy = trajectory.copy()
                            path_copy = global_path.copy()
                            metrics = compute_metrics(
                                start_time_task,
                                traj_copy,
                                path_copy,
                                pixel_per_mm
                            )
                            # Вывод в консоль
                            print(f"\nМетрики движения ({algorithm})")
                            print(f"Время прохождения маршрута: {metrics['time_s']} с")
                            print(f"Планируемый путь: {metrics['planned_length_mm']} мм")
                            print(f"Пройденный путь: {metrics['actual_length_mm']} мм")
                            print(f"MSE (мм²): {metrics['mse_mm2']}")
                            if metrics['r_squared'] is not None:
                                print(f"R² (коэффициент детерминации): {metrics['r_squared']}")
                            print(f"Количество точек пути: {metrics['num_points']}")

                    # Сброс состояния
                    global_path = []
                    trajectory.clear()
                    start_time_task = None

                # Ограничение максимальной скорости (в относительном масштабе)
                if speed > v_max and speed > 1e-6:
                    velocity = velocity / speed * v_max

                # Расчёт скорости в связанной системе координат
                comp_matrix = np.array([[0, 1],
                                        [-1, 0]])
                rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                            [math.sin(angle), math.cos(angle)]])
                Vx, Vy = velocity @ comp_matrix @ rotation_matrix

                # Запоминаем начальную точку при первом кадре движения
                if motion_started and start_point is None:
                    start_point = robot_position.copy()

                # Отправка данных о скоростях (позже по обратной
                # кинематике преобразуются в угловые скорости колёс)
                if motion_started:
                    trajectory.append(robot_position.copy())
                    if robot_status:
                        send_velocity(Vx, Vy, 0)
                        pass
                else:
                    if robot_status:
                        send_velocity(0, 0, 0)
                        pass
                    pass

                # Целевая точка
                if click_point is not None:
                    # красный круг цели (всегда)
                    cv2.circle(working_area, (int(click_point[1]), int(click_point[0])), 10, (0, 0, 255), -1)
                    # подписи только если маршрут существует
                    cv2.putText(working_area, "End",
                                (int(click_point[1]) + 15, int(click_point[0]) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
                    cv2.putText(working_area,
                                f"[{int(click_point[1])}, {int(click_point[0])}]",
                                (int(click_point[1]) + 15, int(click_point[0]) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                # Начальная точка маршрута
                if start_point is not None and auto_mode_enable == 0:
                    # чёрный круг старта (всегда, если был запомнен)
                    cv2.circle(working_area, (int(start_point[1]), int(start_point[0])), 10, (0, 0, 0), -1)
                    # подписи только если маршрут существует
                    cv2.putText(working_area, "Start",
                                (int(start_point[1]) + 15, int(start_point[0]) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
                    cv2.putText(working_area,
                                f"[{int(start_point[1])}, {int(start_point[0])}]",
                                (int(start_point[1]) + 15, int(start_point[0]) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

                # Координаты робота
                coord_text = f"Robot: [{center_y}, {center_x}]"
                (text_w, text_h), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = output_size[0] - text_w - 375
                text_y = 80 + text_h
                cv2.putText(working_area, coord_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)

                # Ошибка позиционирования
                if click_point is not None:
                    goal_x = click_point[0]
                    goal_y = click_point[1]
                    dx = goal_x - center_x
                    dy = goal_y - center_y
                    error_text = f"Error: [{int(dx)}, {int(dy)}]"
                    (ew, eh), _ = cv2.getTextSize(error_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(working_area, error_text,
                                (text_x, text_y + eh + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)

                # Глобальный путь чёрными стрелками
                if global_path and len(global_path) > 1:
                    for i in range(len(global_path) - 1):
                        pt1 = (int(global_path[i][1]), int(global_path[i][0]))
                        pt2 = (int(global_path[i + 1][1]), int(global_path[i + 1][0]))
                        cv2.arrowedLine(working_area, pt1, pt2, (0, 0, 0), 3, tipLength=0.15)

                # Вектор притяжения (красная стрелка)
                cv2.arrowedLine(working_area,
                                tuple(robot_position[::-1].astype(int)),
                                (int(robot_position[1] + v_att[1] * 1000),
                                 int(robot_position[0] + v_att[0] * 1000)),
                                (0, 0, 0), 16)

                # Траектория
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        pt1 = tuple(trajectory[i - 1][::-1].astype(int))
                        pt2 = tuple(trajectory[i][::-1].astype(int))
                        cv2.line(working_area, pt1, pt2, (255, 38, 0), 5)

                # Отрисовка маршрута после проезда
                if last_completed_path is not None:
                    # Плановый путь
                    planned = last_completed_path['planned']
                    if planned and len(planned) > 1:
                        for i in range(len(planned) - 1):
                            pt1 = (int(planned[i][1]), int(planned[i][0]))
                            pt2 = (int(planned[i + 1][1]), int(planned[i + 1][0]))
                            cv2.arrowedLine(working_area, pt1, pt2,
                                            (128, 128, 128), 2, tipLength=0.1)

                    # Фактическая траектория
                    actual = last_completed_path['actual']
                    if actual and len(actual) > 1:
                        for i in range(1, len(actual)):
                            pt1 = (int(actual[i - 1][1]), int(actual[i - 1][0]))
                            pt2 = (int(actual[i][1]), int(actual[i][0]))
                            cv2.line(working_area, pt1, pt2, (255, 38, 0), 3)

                # Отрисовка траекторий автоматического тестирования
                if auto_mode_enable and auto_phase == 'finished' and auto_trajectories:
                    color_map = {
                        "astar": (0, 0, 0),
                        "dijkstra": (255, 0, 0),
                        "greedy": (0, 140, 0),
                        "bdastar": (255, 0, 255)
                    }

                    for algo_name, traj in auto_trajectories:
                        if len(traj) > 1:
                            pts = np.array([(int(p[1]), int(p[0])) for p in traj], dtype=np.int32)
                            cv2.polylines(working_area, [pts], False,
                                          color_map.get(algo_name, (128, 128, 128)), 6)

            # Отображение
            display = cv2.resize(working_area,
                                 (output_size[0] // working_display_scale,
                                  output_size[1] // working_display_scale))
            cv2.imshow("map_planner", display)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    except (ConnectionResetError, BrokenPipeError):
        print("Client disconnected.")
    finally:
        if robot_status:
            sock.close()
        print("Final")


if __name__ == '__main__':
    main()
