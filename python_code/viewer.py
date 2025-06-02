import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("agg")


class Viewer(object):
    def __init__(self, output_folder="output", save_frequency=10):
        self.trajectory = []
        self.camera = None
        self.image = None
        self.frame_count = 0
        self.output_folder = output_folder
        self.save_frequency = save_frequency

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.fig = plt.figure(figsize=(12, 6))
        self.ax1 = self.fig.add_subplot(121, projection="3d")
        self.ax2 = self.fig.add_subplot(122)

    def update_pose(self, pose):
        if pose is not None:
            self.trajectory.append(pose.t)
            self.camera = pose
        self.update()

    def update_image(self, image):
        if image is not None:
            if image.ndim == 2:
                image = np.repeat(image[..., np.newaxis], 3, axis=2)
            self.image = cv2.resize(image, (376, 240))
        self.update()

    def update(self):
        self.ax1.clear()
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("Y")
        self.ax1.set_zlabel("Z")
        self.ax1.set_xlim(-20, 2)
        self.ax1.set_ylim(-1, 10)
        self.ax1.set_zlim(-1, 4)
        self.ax1.set_aspect("equal")
        self.ax1.set_title("Camera Trajectory")

        # Plot full trajectory
        if self.trajectory:
            trajectory_array = np.array(self.trajectory)
            self.ax1.plot(
                trajectory_array[:, 0],
                trajectory_array[:, 1],
                trajectory_array[:, 2],
            )
        axis_length = 1
        origin = np.zeros(3)
        # X-axis (Red)
        if self.camera is not None:
            self.ax1.quiver(
                self.camera.t[0],
                self.camera.t[1],
                self.camera.t[2],
                self.camera.R[0, 0] * axis_length,
                self.camera.R[1, 0] * axis_length,
                self.camera.R[2, 0] * axis_length,
                color="r",
            )
            # Y-axis (Green)
            self.ax1.quiver(
                self.camera.t[0],
                self.camera.t[1],
                self.camera.t[2],
                self.camera.R[0, 1] * axis_length,
                self.camera.R[1, 1] * axis_length,
                self.camera.R[2, 1] * axis_length,
                color="g",
            )
            # Z-axis (Blue)
            self.ax1.quiver(
                self.camera.t[0],
                self.camera.t[1],
                self.camera.t[2],
                self.camera.R[0, 2] * axis_length,
                self.camera.R[1, 2] * axis_length,
                self.camera.R[2, 2] * axis_length,
                color="b",
            )

        # Show image
        # self.ax2.clear()
        if self.image is not None:
            self.ax2.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.ax2.axis("off")
        self.ax2.set_title("Camera View")

        # Queue the figure for saving if it's time
        if self.frame_count % self.save_frequency == 0:
            self.fig.savefig(f"output/frame_{self.frame_count:04d}.png")
        self.frame_count += 1

    def run(self):
        pass  # No need to keep a window open when saving to files
