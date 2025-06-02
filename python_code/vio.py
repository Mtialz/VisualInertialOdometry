from queue import Queue
from threading import Thread

from config import ConfigEuRoC
from image import ImageProcessor
from optimization_vio import OptimizationBasedVIO


class VisualIntertialOdometry(object):
    def __init__(self, config, img_queue, imu_queue, max_steps, viewer=None):
        self.config = config
        self.viewer = viewer

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.optimization_vio = OptimizationBasedVIO(config)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()
        self.max_steps = max_steps
        self.counter = 0

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            feature_msg = self.image_processor.stereo_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return

            self.image_processor.imu_callback(imu_msg)
            self.optimization_vio.imu_callback(imu_msg)

    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return

            result = self.optimization_vio.feature_callback(feature_msg)

            if result is not None and self.viewer is not None:
                self.viewer.update_pose(result.cam0_pose)


if __name__ == "__main__":
    import time
    import argparse

    from dataset import EuRoCDataset, DataPublisher
    from viewer import Viewer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="data/MH_01_easy",
        help="Path of EuRoC MAV dataset.",
    )
    parser.add_argument(
        "--view", action="store_true", default=True, help="Show trajectory."
    )
    args = parser.parse_args()

    # visualizer of the estimated trajectory
    viewer = Viewer()

    # loading the dataset
    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=40.0)  # start from static state

    # queues for storing image and IMU data
    img_queue = Queue()
    imu_queue = Queue()
    # gt_queue = Queue()

    # configuration parameters to operate the EuRoC dataset
    # a class collection of the parameters
    config = ConfigEuRoC()

    # visual-inertial odometry class
    vio = VisualIntertialOdometry(
        config, img_queue, imu_queue, max_steps=20, viewer=viewer
    )

    duration = float("inf")

    # smaller if image processing is slow
    ratio = 0.4

    # data publisher for the dataset
    imu_publisher = DataPublisher(dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(dataset.stereo, img_queue, duration, ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
