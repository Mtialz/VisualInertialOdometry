import numpy as np
from scipy.stats import chi2
from scipy.optimize import least_squares
from utils import quaternion_normalize

def quaternion_from_axis_angle(axis_angle):
    """Convert axis-angle representation to quaternion."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis_angle / angle
    sin_angle = np.sin(angle / 2)
    cos_angle = np.cos(angle / 2)
    return np.array(
        [
            axis[0] * sin_angle,
            axis[1] * sin_angle,
            axis[2] * sin_angle,
            cos_angle,
        ]
    )


def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


from utils import *
from feature import Feature

import time
from collections import namedtuple


class StereoReprojectionError:
    def __init__(self, left_obs, right_obs, baseline, weight):
        self.left_obs = left_obs
        self.right_obs = right_obs
        self.baseline = baseline
        self.weight = weight

    def __call__(self, quaternion, position, point3d):
        """
        Compute the stereo reprojection error.
        """
        R_wc = to_rotation(quaternion)
        t_wc = position

        # Transform point from world to camera frame
        p_c = R_wc.T @ (point3d - t_wc)
        x, y, z = p_c

        if z <= 0:
            return np.zeros(4)  # avoid projection behind camera

        fx, fy, cx, cy = self.weight['fx'], self.weight['fy'], self.weight['cx'], self.weight['cy']
        b = self.baseline

        u_l = fx * x / z + cx
        v_l = fy * y / z + cy
        u_r = fx * (x - b) / z + cx
        v_r = fy * y / z + cy

        residual = np.array([
            u_l - self.left_obs[0],
            v_l - self.left_obs[1],
            u_r - self.right_obs[0],
            v_r - self.right_obs[1]
        ])

        return residual



class IMUState(object):
    # id for next IMU state
    next_id = 0

    # Gravity vector in the world frame
    gravity = np.array([0.0, 0.0, -9.81])

    # Transformation offset from the IMU frame to the body frame.
    # The transformation takes a vector from the IMU frame to the
    # body frame. The z axis of the body frame should point upwards.
    # Normally, this transform should be identity.
    T_imu_body = Isometry3d(np.identity(3), np.zeros(3))

    def __init__(self, new_id=None):
        # An unique identifier for the IMU state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the IMU (body) frame.
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])

        # Position of the IMU (body) frame in the world frame.
        self.position = np.zeros(3)
        # Velocity of the IMU (body) frame in the world frame.
        self.velocity = np.zeros(3)

        # Bias for measured angular velocity and acceleration.
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)

        # These three variables should have the same physical
        # interpretation with `orientation`, `position`, and
        # `velocity`. There three variables are used to modify
        # the transition matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0.0, 0.0, 0.0, 1.0])
        self.position_null = np.zeros(3)
        self.velocity_null = np.zeros(3)

        # Transformation between the IMU and the left camera (cam0)
        self.R_imu_cam0 = np.identity(3)
        self.t_cam0_imu = np.zeros(3)


class CAMState(object):
    # Takes a vector from the cam0 frame to the cam1 frame.
    R_cam0_cam1 = None
    t_cam0_cam1 = None

    def __init__(self, new_id=None):
        # An unique identifier for the CAM state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the camera frame.
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])

        # Position of the camera frame in the world frame.
        self.position = np.zeros(3)

        # These two variables should have the same physical
        # interpretation with `orientation` and `position`.
        # There two variables are used to modify the measurement
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0.0, 0.0, 0.0, 1.0])
        self.position_null = np.zeros(3)


class StateServer(object):
    """
    Store one IMU states and several camera states for constructing
    measurement model.
    """

    def __init__(self):
        self.imu_state = IMUState()
        self.cam_states = dict()  # <CAMStateID, CAMState>, ordered dict


class OptimizationBasedVIO(object):
    def __init__(self, config):
        self.config = config
        self.optimization_config = config.optimization_config

        # IMU data buffer
        self.imu_msg_buffer = []

        # State vector
        self.state_server = StateServer()
        # Features used
        self.map_server = dict()  # <FeatureID, Feature>

        # Set the initial IMU state.
        self.state_server.imu_state.velocity = config.velocity

        # Gravity vector in the world frame
        IMUState.gravity = config.gravity

        # Transformation between the IMU and the left camera (cam0)
        T_cam0_imu = np.linalg.inv(config.T_imu_cam0)
        self.state_server.imu_state.R_imu_cam0 = T_cam0_imu[:3, :3].T
        self.state_server.imu_state.t_cam0_imu = T_cam0_imu[:3, 3]

        # Extrinsic parameters of camera and IMU.
        T_cam0_cam1 = config.T_cn_cnm1
        CAMState.R_cam0_cam1 = T_cam0_cam1[:3, :3]
        CAMState.t_cam0_cam1 = T_cam0_cam1[:3, 3]
        Feature.R_cam0_cam1 = CAMState.R_cam0_cam1
        Feature.t_cam0_cam1 = CAMState.t_cam0_cam1
        IMUState.T_imu_body = Isometry3d(
            config.T_imu_body[:3, :3], config.T_imu_body[:3, 3]
        )

        # Tracking rate.
        self.tracking_rate = None

        # Indicate if the gravity vector is set.
        self.is_gravity_set = False
        # Indicate if the received image is the first one. The system will
        # start after receiving the first image.
        self.is_first_img = True

    def imu_callback(self, imu_msg):
        """
        Callback function for the imu message.
        """
        self.imu_msg_buffer.append(imu_msg)

        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 200:
                self.initialize_gravity_and_bias()
                self.is_gravity_set = True

    def feature_callback(self, feature_msg):
        """
        Callback function for feature measurements.
        """
        if not self.is_gravity_set:
            return
        start = time.time()

        # Start the system if the first image is received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.state_server.imu_state.timestamp = feature_msg.timestamp

        # Add new observations for existing features or new features
        # in the map server.
        self.add_feature_observations(feature_msg)

        # Optimize pose using features
        self.optimize_pose()

        print("---optimization elapsed:    ", time.time() - start)

        try:
            # Publish the odometry.
            return self.publish(feature_msg.timestamp)
        finally:
            # Reset the system if necessary.
            self.online_reset()

    def online_reset(self):
        """Reset the system when necessary."""
        # Check if we need to reset based on number of tracked features
        min_required_features = (
            3  # Minimum features needed for reliable tracking
        )
        num_tracked_features = sum(
            1
            for feature in self.map_server.values()
            if self.state_server.imu_state.id in feature.observations
        )

        # Check for numerical instability in state estimates
        position_valid = np.all(
            np.isfinite(self.state_server.imu_state.position)
        )
        orientation_valid = np.all(
            np.isfinite(self.state_server.imu_state.orientation)
        )

        should_reset = (
            num_tracked_features < min_required_features
            or not position_valid
            or not orientation_valid
        )

        if should_reset:
            # Clear feature tracking history
            self.map_server.clear()

            # Reset IMU biases and velocity
            self.state_server.imu_state.gyro_bias = np.zeros(3)
            self.state_server.imu_state.acc_bias = np.zeros(3)
            self.state_server.imu_state.velocity = np.zeros(3)

            # Keep position and orientation as is, unless they're invalid
            if not position_valid:
                self.state_server.imu_state.position = np.zeros(3)
            if not orientation_valid:
                self.state_server.imu_state.orientation = np.array(
                    [0.0, 0.0, 0.0, 1.0]
                )

            # Reset null space variables
            self.state_server.imu_state.orientation_null = (
                self.state_server.imu_state.orientation.copy()
            )
            self.state_server.imu_state.position_null = (
                self.state_server.imu_state.position.copy()
            )
            self.state_server.imu_state.velocity_null = (
                self.state_server.imu_state.velocity.copy()
            )

    def initialize_gravity_and_bias(self):
        """
        Initialize the IMU bias and initial orientation based on the
        first few IMU readings.
        """
        sum_angular_vel = np.zeros(3)
        sum_linear_acc = np.zeros(3)
        for msg in self.imu_msg_buffer:
            sum_angular_vel += msg.angular_velocity
            sum_linear_acc += msg.linear_acceleration

        gyro_bias = sum_angular_vel / len(self.imu_msg_buffer)
        self.state_server.imu_state.gyro_bias = gyro_bias

        # This is the gravity in the IMU frame.
        gravity_imu = sum_linear_acc / len(self.imu_msg_buffer)

        # Initialize the initial orientation, so that the estimation
        # is consistent with the inertial frame.
        gravity_norm = np.linalg.norm(gravity_imu)
        IMUState.gravity = np.array([0.0, 0.0, -gravity_norm])

        q0_i_w = from_two_vectors(gravity_imu, -IMUState.gravity)
        self.state_server.imu_state.orientation = to_quaternion(
            np.transpose(to_rotation(q0_i_w))
        )

    def add_feature_observations(self, feature_msg):
        """Add new feature observations to the map"""

        # Add all features in the feature_msg to self.map_server
        for feature in feature_msg.features:
            if feature.id not in self.map_server:

                # This is a new feature.
                map_feature = Feature(feature.id, self.optimization_config)
                map_feature.observations[self.state_server.imu_state.id] = (
                    np.array([feature.u0, feature.v0, feature.u1, feature.v1])
                )
                # Initialize the feature position using triangulation
                current_pose = Isometry3d(
                    to_rotation(self.state_server.imu_state.orientation).T,
                    self.state_server.imu_state.position,
                )

                map_feature.is_initialized = (
                    True  # Mark as initialized if triangulation succeeds
                )
                self.map_server[feature.id] = map_feature
            else:
                # This is an old feature.
                self.map_server[feature.id].observations[
                    self.state_server.imu_state.id
                ] = np.array([feature.u0, feature.v0, feature.u1, feature.v1])

    def optimize_pose(self):
        imu_state = self.state_server.imu_state
        q_init = imu_state.orientation
        p_init = imu_state.position

        def residual_fn(x):
            q = quaternion_normalize(x[:4])
            p = x[4:7]
            residuals = []

            for feature in self.map_server.values():
                if not feature.is_initialized:
                    continue
                obs = feature.observations.get(imu_state.id, None)
                if obs is None:
                    continue
                stereo_error = StereoReprojectionError(
                    obs[:2], obs[2:], self.config.baseline, self.config.cam_intrinsics
                )
                res = stereo_error(q, p, feature.position)
                residuals.append(res)

            return np.concatenate(residuals) if residuals else np.zeros(1)

        x0 = np.hstack([q_init, p_init])
        result = least_squares(residual_fn, x0, method='trf')

        q_opt = quaternion_normalize(result.x[:4])
        p_opt = result.x[4:7]

        imu_state.orientation = q_opt
        imu_state.position = p_opt


    def publish(self, time):
        """Publish current pose estimate"""
        imu_state = self.state_server.imu_state

        # Save to file
        filename = "publish_estimated.txt"
        with open(filename, "a") as f:
            f.write(f"{imu_state.timestamp} "
                    f"{' '.join(map(str, imu_state.position))} "
                    f"{' '.join(map(str, imu_state.orientation))}\n")

        # Transform poses
        T_i_w = Isometry3d(
            to_rotation(imu_state.orientation).T, imu_state.position
        )
        T_b_w = IMUState.T_imu_body * T_i_w * IMUState.T_imu_body.inverse()
        body_velocity = IMUState.T_imu_body.R @ imu_state.velocity

        R_w_c = imu_state.R_imu_cam0 @ T_i_w.R.T
        t_c_w = imu_state.position + T_i_w.R @ imu_state.t_cam0_imu
        T_c_w = Isometry3d(R_w_c.T, t_c_w)

        return namedtuple(
            "vio_result", ["timestamp", "pose", "velocity", "cam0_pose"]
        )(time, T_b_w, body_velocity, T_c_w)
