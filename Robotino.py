import requests
import socket

import yaml

with open('parameters.yaml') as config_file:
    config = yaml.safe_load(config_file)


# Параметры для Robotino
robot_id: str = config['socket_params']['ip_address']
robot_port: int = config['socket_params']['port']


def connect_to_robotino():
    """
    Устанавливает TCP-сокет соединение с роботом.

    Выходы:
        - sock: socket.socket — при успехе подключения, None при ошибке.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((robot_id, robot_port))
        return sock
    except Exception as e:
        print(f"Ошибка TCP подключения: {e}")
        return None


def send_velocity(vx: float,
                  vy: float,
                  omega: float):
    """
    Отправляет вектор скорости роботу через HTTP API (Robotino).

    Входы:
        - vx: float — линейная скорость вперёд (м/с).
        - vy: float — боковая скорость влево (м/с).
        - omega: float — угловая скорость (рад/с).

    Выходы:
        - bool — True, если сервер ответил. False при сетевой ошибке или ошибке HTTP.
    """
    url = f"http://{robot_id}/data/omnidrive"
    data = [vx, vy, omega]
    try:
        response = requests.post(url, json=data, timeout=1)
        if response.status_code == 200:
            return True
        else:
            print(f"Ошибка HTTP {response.status_code}: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Сетевая ошибка: {e}")
        return False
