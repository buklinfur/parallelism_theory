import argparse
import logging
import time
import cv2
import threading
from queue import Queue, LifoQueue
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Sensor:
    def get(self):
        raise NotImplementedError('Subclasses must implement method get()')


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self,
                 cam_idx: int = 0,
                 width: int = 640,
                 height: int = 480):

        self.cam_idx = cam_idx
        self.width = width
        self.height = height
        self._cap = None

        try:
            self._cap = cv2.VideoCapture(cam_idx)
            if not self._cap.isOpened():
                logger.error(f"Critical error: Couldn't find a camera linked "
                             f"to input index {cam_idx}.")
                raise RuntimeError(f'Camera with index {cam_idx} not found')

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info(f'Camera initialized with resolution {width}x{height}.')
        except Exception as e:
            logger.error(f'Critical error during camera initialization: {str(e)}')
            self.stop()
            raise

    def get(self):
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError('Camera is not available.')

        ret, frame = self._cap.read()
        frame = cv2.flip(frame, 180)
        if not ret:
            logger.error('Critical error: Couldn\'t capture the frame.')
            raise RuntimeError('Frame capture failed.')
        return frame

    def stop(self):
        if hasattr(self, '_cap') and self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info('Camera is released.')

    def __del__(self):
        self.stop()


def sensor_worker(sensor: SensorX, queue: LifoQueue, flag: threading.Event):
    while flag.is_set():
        try:
            data = sensor.get()
            try:
                queue.get_nowait()
            except:
                pass
            queue.put_nowait(data)
        except Exception as e:
            logger.error(f'Sensor worker error: {str(e)}')
            break


def camera_worker(queue: Queue, flag: threading.Event,
                  cam_idx: int, width: int, height: int):
    try:
        cam = SensorCam(cam_idx, width, height)
        while flag.is_set():
            try:
                frame = cam.get()
                try:
                    queue.get_nowait()
                except:
                    pass
                queue.put_nowait(frame)
            except Exception as e:
                logger.error(f'Camera worker error: {str(e)}')
                break
    finally:
        cam.stop()


class ImageWindow:
    def __init__(self, fps: int = 15, height: int = 480):
        self._sensor_data = [0, 0, 0]
        self.frame = None
        self.fps = fps
        self._height = height
        self._lock = threading.Lock()

    def show(self, cam_queue: Queue, queues: List[LifoQueue]):
        try:
            with self._lock:
                for i in range(3):
                    try:
                        self._sensor_data[i] = queues[i].get_nowait()
                    except:
                        pass

                try:
                    self.frame = cam_queue.get_nowait()
                except:
                    pass

                if self.frame is not None:
                    cv2.putText(
                        self.frame,
                        f'Sensor1: {self._sensor_data[0]}  '
                        f'Sensor2: {self._sensor_data[1]}  '
                        f'Sensor3: {self._sensor_data[2]}',
                        (10, self._height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        1
                    )
                    cv2.imshow('Camera and Sensors', self.frame)
        except Exception as e:
            logger.error(f'Error in show(): {str(e)}')

    def stop(self):
        cv2.destroyAllWindows()

    def __del__(self):
        self.stop()


def main():
    parser = argparse.ArgumentParser(description='Camera and sensors display')
    parser.add_argument('--camera', type=int, default=0, help='Camera index in system')
    parser.add_argument('--height', type=int, default=480, help='Camera height resolution')
    parser.add_argument('--width', type=int, default=720, help='Camera width resolution')
    parser.add_argument('--fps', type=float, default=15, help='Display refresh rate')
    args = parser.parse_args()

    flag = threading.Event()
    flag.set()

    try:
        sensors = [
            SensorX(0.01),
            SensorX(0.1),
            SensorX(1)
        ]

        sensor_queues = [LifoQueue(maxsize=1) for _ in range(3)]
        cam_queue = Queue(maxsize=1)

        workers = []
        for i in range(3):
            worker = threading.Thread(
                target=sensor_worker,
                args=(sensors[i], sensor_queues[i], flag),
                daemon=True
            )
            worker.start()
            workers.append(worker)

        cam_worker = threading.Thread(
            target=camera_worker,
            args=(cam_queue, flag, args.camera, args.width, args.height),
            daemon=True
        )
        cam_worker.start()
        workers.append(cam_worker)

        window = ImageWindow(fps=args.fps, height=args.height)

        while True:
            window.show(cam_queue, sensor_queues)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info('Exit requested by user.')
                break

            time.sleep(1 / args.fps)

    except KeyboardInterrupt:
        logger.info('Program interrupted by user.')
    except Exception as e:
        logger.error(f'Main thread error: {str(e)}')
    finally:
        flag.clear()
        for worker in workers:
            worker.join(timeout=1)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
