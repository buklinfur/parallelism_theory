import os
import cv2
import logging
import argparse
import queue
import threading
import time
from logging.handlers import RotatingFileHandler

# Создать папку для логов
if not os.path.exists('log'):
    os.makedirs('log')

# -------------------- Настройка логирования --------------------
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        RotatingFileHandler(
            'log/app.log',
            maxBytes=1024*1024,
            backupCount=3
        )
    ],
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------- Базовый класс Sensor --------------------
class Sensor:
    def __init__(self, name):
        self.name = name
        self.queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run)

    def _run(self):
        raise NotImplementedError

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self.thread.join()

# -------------------- Класс для камеры --------------------
class SensorCam(Sensor):
    def __init__(self, cam_name, resolution):
        super().__init__("Camera")
        self.cam_name = cam_name
        self.resolution = tuple(map(int, resolution.split('x')))
        self.cap = None

        try:
            # Используйте cv2.CAP_DSHOW для Windows
            self.cap = cv2.VideoCapture(int(self.cam_name), cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError(f"Camera {self.cam_name} not found")
        except Exception as e:
            logging.error(f"Camera error: {str(e)}")
            raise

    def _run(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Camera frame read error")
                break
            if self.queue.empty():
                self.queue.put(frame)
        self.cap.release()

    def get(self):
        return self.queue.get() if not self.queue.empty() else None

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

# -------------------- Виртуальные датчики --------------------
class SensorX(Sensor):
    def __init__(self, interval):
        super().__init__(f"Sensor_{interval}")
        self.interval = interval
        self.value = 0.0

    def _run(self):
        while not self._stop_event.is_set():
            self.value = time.time() % 100
            if self.queue.empty():
                self.queue.put(self.value)
            time.sleep(self.interval)

# -------------------- Класс для отображения --------------------
class WindowImage:
    def __init__(self, fps):
        self.fps = fps
        self.window_name = "Sensor Display"
        self.last_values = {}

    def show(self, img, sensors):
        h, w = img.shape[:2]
        for i, (name, value) in enumerate(sensors.items()):
            cv2.putText(
                img,
                f"{name}: {value:.2f}",
                (10, h - 30 - i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        cv2.imshow(self.window_name, img)
        if cv2.waitKey(int(1000/self.fps)) & 0xFF == ord('q'):
            return False
        return True

    def __del__(self):
        cv2.destroyAllWindows()


def list_available_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        cameras.append(index)
        cap.release()
        index += 1
    return cameras

# -------------------- Основная программа --------------------
def main():
    print(f"Available devices: {list_available_cameras()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--resolution", default="640x480")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    camera = None  # Инициализация переменных
    sensors = []

    try:
        camera = SensorCam(args.camera, args.resolution)
        sensors = [
            SensorX(0.01),
            SensorX(0.1),
            SensorX(1)
        ]

        camera.start()
        for s in sensors:
            s.start()

        window = WindowImage(args.fps)
        last_values = {s.name: 0.0 for s in sensors}

        while True:
            frame = camera.get()
            if frame is None:
                continue

            for s in sensors:
                if not s.queue.empty():
                    last_values[s.name] = s.queue.get()

            if not window.show(frame.copy(), last_values):
                break

    except Exception as e:
        logging.critical(f"Critical error: {str(e)}", exc_info=True)
    finally:
        if camera is not None:
            camera.stop()
        for s in sensors:
            s.stop()
        if 'window' in locals():
            del window


if __name__ == "__main__":
    main()