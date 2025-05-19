from argparse import ArgumentParser
from queue import Queue, Empty
from threading import Event, Thread
from time import time
import cv2
from ultralytics import YOLO


class VideoProcessor:
    def __init__(self, video_path, output_file, num_threads):
        self.video_path = video_path
        self.output_file = output_file
        self.num_threads = num_threads
        self.video = cv2.VideoCapture(video_path)
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_file, fourcc, self.fps, (self.frame_width, self.frame_height))

        self.stop_event = Event()
        self.task_queue = Queue()
        self.result_by_index = {}
        self.threads = [
            Thread(target=self.worker, args=())
            for _ in range(num_threads)
        ]

    def __enter__(self):
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        self.writer.release()
        self.video.release()

    def run(self):
        processing_begin_time = time()

        counter = 0
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            self.task_queue.put((frame, counter))
            counter += 1

        self.task_queue.join()

        self.stop_event.set()

        counter = 0
        while counter in self.result_by_index:
            self.writer.write(self.result_by_index[counter])
            counter += 1

        process_time = time() - processing_begin_time
        print(f'Processing took {process_time:.2f} seconds')

    def worker(self):
        model = YOLO('yolov8s-pose.pt')

        while not self.stop_event.is_set():
            try:
                frame, index = self.task_queue.get(timeout=0.1)
            except Empty:
                continue
            annotated_frame = model.predict(frame, verbose=False, device='cpu')[0].plot()
            self.result_by_index[index] = annotated_frame
            self.task_queue.task_done()


def parse_args():
    parser = ArgumentParser(description='Yolo')
    parser.add_argument('vidPath', type=str, help='Path to video')
    parser.add_argument('threadsCount', type=int, help='Count of worker threads')
    parser.add_argument('outputName', type=str, help='Name of output file')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.threadsCount < 1:
        print('Expected at least one worker thread')
    else:
        with VideoProcessor(args.vidPath, args.outputName, args.threadsCount) as processor:
            processor.run()