import csv
from airsim import Vector3r
import setup_path
import airsim
import sys
import time
import os
import tempfile
import numpy as np
import cv2
import pprint
import math
import logging
import logging.handlers
import json
from airsim import utils


class CvPathPlanner:
    def __init__(
            self,
            planner,
            planning_log_path,
            flight_box,
            forward_overlap,
            lateral_overlap,
            cam_foc,
            sensor_w,
            sensor_h,
            playstart_x_offset,
            playstart_y_offset,
            playstart_z_offset,
            img_save_path,
            scene_buildings_num=None,
            instance_required=False,
            img_position_log="uav_position.txt",
            log_path="uav.log"
    ):
        self.planner = planner
        self.planning_log_path = planning_log_path
        self.client = None
        self.start_x = flight_box[0]
        self.start_y = flight_box[1]
        self.start_z = flight_box[2]
        self.end_x = flight_box[3]
        self.end_y = flight_box[4]
        self.end_z = flight_box[5]

        self.flight_xrange = abs(self.end_x - self.start_x)
        self.flight_yrange = abs(self.end_y - self.start_y)

        self.uav_name = []

        self.img_save_path = img_save_path
        if not os.path.exists(self.img_save_path):
            os.mkdir(self.img_save_path)
        self.log_path = os.path.join(img_save_path, log_path)
        self.img_position_log = os.path.join(img_save_path, img_position_log)

        self.logger = self.init_logger()
        self.uav_list = []

        # Camera parameters
        self.cam_foc = cam_foc
        self.forward_overlap = forward_overlap
        self.lateral_overlap = lateral_overlap
        self.sensor_w = sensor_w
        self.sensor_h = sensor_h

        # Collect relevant parameters
        self.flight_dir = None
        self.x_step = None
        self.y_step = None
        self.shot_step = None
        self.path_num = None
        self.stripe_width = None
        self.path = []
        self.distance = None

        self.uav_path_num = None
        self.uav_extra_path_num = None

        self.init_logger()

        # Pose
        self.playstart_x_offset = playstart_x_offset
        self.playstart_y_offset = playstart_y_offset
        self.playstart_z_offset = playstart_z_offset

        self.airsim_setting = r"C:\Users\Talon\Documents\AirSim\settings.json"      # Airsim settings file, used to set the initial number and position of drones

        self.instance_required = instance_required
        self.scene_buildings_num = scene_buildings_num

        self.init_logger()
        # self.init_weather()
        self.init_uav()

        if not self.planner:
            self.init_cam_pose()
            self.generate_path()
            self.fly_shot()
            self.convert2_ue_path_flight_log()
        else:
            self.playstart_x_offset = 0
            self.playstart_y_offset = 0
            self.playstart_z_offset = 0
            self.plan_shot()
            self.convert2_ue_path_flight_log()

    def init_logger(self):
        logger = logging.getLogger("UavFlight")
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(filename=self.log_path)

        logger.setLevel(logging.DEBUG)
        handler1.setLevel(logging.WARNING)
        handler2.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)

        logger.addHandler(handler1)
        logger.addHandler(handler2)
        return logger

    def init_uav(self):
        self.logger.info('Request connection to drone')
        client = airsim.MultirotorClient()
        client.confirmConnection()
        self.logger.info('Connection successful')
        self.client = client

        self.client.simGetCameraInfo("0")

        if self.instance_required:
            for i in range(self.scene_buildings_num):
                self.client.simSetSegmentationObjectID("building_" + str(i+1), i+1, True)

            self.client.simSetSegmentationObjectID("ground[\w]*", -1, True)
            self.client.simSetSegmentationObjectID("SkySphere", -1, True)

    # def init_weather(self):
    #     wind = airsim.Vector3r(0, 0, 0)
    #     self.client.simSetWind(wind)
    #     time.sleep(1)

    def init_cam_pose(self):
        camera_pose_0 = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.radians(-45), 0, 0))  # PRY in radians      pitch row yaw
        camera_pose_1 = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.radians(-45), 0, np.radians(90)))
        camera_pose_2 = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.radians(-45), 0, np.radians(270)))
        camera_pose_4 = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.radians(-45), 0, np.radians(180)))
        camera_pose_3 = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.radians(-90), 0, 0))
        self.client.simSetCameraPose(0, camera_pose_0)
        self.client.simSetCameraPose(1, camera_pose_1)
        self.client.simSetCameraPose(2, camera_pose_2)
        self.client.simSetCameraPose(3, camera_pose_3)
        self.client.simSetCameraPose(4, camera_pose_4)

    def generate_path(self):
        # Calculate the distance for each frame captured based on camera parameters, ground height, and acquisition overlap
        scene_w = abs(self.sensor_w * self.end_z / self.cam_foc)
        scene_h = abs(self.sensor_h * self.end_z / self.cam_foc)

        self.flight_dir = "x"
        self.stripe_width = self.y_step = scene_w * (1 - self.lateral_overlap)
        self.shot_step = self.x_step = scene_h * (1 - self.forward_overlap)
        self.path_num = self.flight_yrange // self.stripe_width + 1
        self.logger.info("Collect along the %s direction first" % self.flight_dir)
        self.logger.info("Shooting step size is % s" % self.shot_step)
        self.logger.info("The route width is %s" % self.stripe_width)
        self.logger.info("Expected to fly %s lanes" % self.path_num)

        path_num = int((self.end_y - self.start_y) // self.stripe_width + 1)
        path_shoot_num = int((self.end_x - self.start_x) // self.shot_step + 1)
        self.logger.info("Number of viewpoints on a path %s" % path_shoot_num)

        for i in range(path_num):
            for j in range(path_shoot_num):
                self.path.append([j * self.shot_step, i * self.stripe_width, self.end_z])

    def shot_save(self, path_point_idx):
        if not self.instance_required:
            responses = self.client.simGetImages([
                airsim.ImageRequest('bottom_center', airsim.ImageType.Scene, False, False),
                airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False),
                airsim.ImageRequest('front_right', airsim.ImageType.Scene, False, False),
                airsim.ImageRequest('back_center', airsim.ImageType.Scene, False, False),
                airsim.ImageRequest('front_left', airsim.ImageType.Scene, False, False),
            ])

            for idx, response in enumerate(responses):
                camera_position = response.camera_position
                cam_quaternion = response.camera_orientation
                pitch_roll_yaw = np.degrees(utils.to_eularian_angles(cam_quaternion))
                print('pitchRollYaw', pitch_roll_yaw)

                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                # filename = os.path.join(self.img_save_path, str(idx) + "-" + time.strftime("%Y%m%d-%H%M%S"))
                filename = os.path.join(self.img_save_path, str(idx), str(idx) + "-" + str(path_point_idx))
                airsim.write_png(os.path.normpath(filename + '.jpg'), img_rgb)

                with open(self.img_position_log, "a+") as f:
                    f.write(filename + ".jpg")
                    f.write(',')
                    f_csv = csv.writer(f, delimiter=',')
                    info = [camera_position.x_val + self.playstart_x_offset * 0.01,
                            camera_position.y_val + self.playstart_y_offset * 0.01,
                            -camera_position.z_val + self.playstart_z_offset * 0.01]
                    info.extend(pitch_roll_yaw)
                    f_csv.writerow(info)
                    # f_csv.writerow(camera_orientation_euler)
        else:
            responses = self.client.simGetImages([
                airsim.ImageRequest('bottom_center', airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)
            ])  # scene vision image in uncompressed RGBA array

            for idx, response in enumerate(responses):
                camera_position = response.camera_position
                camQuaternion = response.camera_orientation
                pitchRollYaw = utils.to_eularian_angles(camQuaternion)
                print('pitchRollYaw', pitchRollYaw)

                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                # filename = os.path.join(self.img_save_path, str(idx), str(idx) + "-" + time.strftime("%Y%m%d-%H%M%S"))
                filename = os.path.join(self.img_save_path, str(idx), str(idx) + "-" + time.strftime("%Y%m%d-%H%M%S"))
                airsim.write_png(os.path.normpath(filename + '.jpg'), img_rgb)

                with open(self.img_position_log, "a+") as f:
                    f.write(filename + ".jpg")
                    f.write(',')
                    f_csv = csv.writer(f, delimiter=',')
                    info = [camera_position.x_val + self.playstart_x_offset * 0.01,
                            camera_position.y_val + self.playstart_y_offset * 0.01,
                            -camera_position.z_val + self.playstart_z_offset * 0.01]
                    info.extend(pitchRollYaw)
                    f_csv.writerow(info)
                    # f_csv.writerow(camera_orientation_euler)

    def fly_shot(self):
        for i in range(5):
            if not os.path.exists(os.path.join(self.img_save_path, str(i))):
                os.mkdir(os.path.join(self.img_save_path, str(i)))

        # Drone cluster execution
        for i in range(len(self.path)):
            # self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(self.path[0], self.path[1], self.path[2]), airsim.to_quaternion(0, 0, 0)), True)
            self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(self.path[i][0], self.path[i][1], self.path[i][2]), airsim.to_quaternion(0, 0, 0)), True)
            time.sleep(0.1)
            self.shot_save(i)

    def plan_shot(self):
        log_file = open(self.planning_log_path, 'r')
        planning_path = log_file.readlines()
        planning_path_list = []
        for i in range(len(planning_path)):
            parts = planning_path[i].split(',')
            one_location = []
            one_location.append(parts[0])                           # phone name
            one_location.append(-float(parts[1])*0.01)              # -x
            one_location.append(-float(parts[2])*0.01)              # -y
            one_location.append(-float(parts[3])*0.01)              # -z
            one_location.append(-float(parts[4]))                   # pitch
            one_location.append(float(parts[5]))                    # roll
            one_location.append(float(parts[6].strip('\n'))+270)    # roll+270
            planning_path_list.append(one_location)

        for i in range(1):
            if not os.path.exists(os.path.join(self.img_save_path, str(i))):
                os.mkdir(os.path.join(self.img_save_path, str(i)))

        for i in range(len(planning_path_list)):
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(planning_path_list[i][1], planning_path_list[i][2], planning_path_list[i][3]),
                            airsim.to_quaternion(0, 0, 0)), True)
            camera_pose_0 = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(
                np.radians(planning_path_list[i][4]),
                np.radians(planning_path_list[i][5]),
                np.radians(planning_path_list[i][6])))
            self.client.simSetCameraPose(0, camera_pose_0)

            responses = self.client.simGetImages([
                airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False)
            ])  # scene vision image in uncompressed RGBA array

            for idx, response in enumerate(responses):
                camera_position = response.camera_position
                cam_quaternion = response.camera_orientation
                pitch_roll_yaw = np.degrees(utils.to_eularian_angles(cam_quaternion))
                print('pitchRollYaw', pitch_roll_yaw)

                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                # filename = os.path.join(self.img_save_path, str(idx) + "-" + time.strftime("%Y%m%d-%H%M%S"))
                filename = os.path.join(self.img_save_path, str(idx), planning_path_list[i][0])
                airsim.write_png(os.path.normpath(filename + '.jpg'), img_rgb)

                with open(self.img_position_log, "a+") as f:
                    f.write(filename + ".jpg")
                    f.write(',')
                    f_csv = csv.writer(f, delimiter=',')
                    info = [camera_position.x_val + self.playstart_x_offset * 0.01,
                            camera_position.y_val + self.playstart_y_offset * 0.01,
                            -camera_position.z_val + self.playstart_z_offset * 0.01]
                    info.extend(pitch_roll_yaw)
                    f_csv.writerow(info)
                    # f_csv.writerow(camera_orientation_euler)

    def convert2_ue_path_flight_log(self):
        input_path = self.img_position_log

        with open(input_path, 'r') as log_file, open(input_path[:-4] + '_cc.txt', 'w') as cc_file:
            lines = log_file.readlines()

            for line in lines:
                if line.strip():  # 检查非空行
                    parts = line.split(',')  # UE
                    formatted_line = (
                        f"{parts[0]},"
                        f"{float(parts[1])},"
                        f"{-float(parts[2])},"
                        f"{float(parts[3])},"
                        f"{float(parts[4])},"
                        f"{float(parts[5])},"
                        f"{float(parts[6].strip()) + 90}\n"
                    )
                    cc_file.write(formatted_line)


# japan scene Oblique photography collection
# test_flight = CvPathPlanner(
#     planner=False,
#     planning_log_path="",
#     flight_box=[0, 0, 0, 220, 210, -100],
#     forward_overlap=0.9,
#     lateral_overlap=0.9,
#     cam_foc=8.8,
#     sensor_w=13.2,
#     sensor_h=8.8,
#     playstart_x_offset=-12000.0,
#     playstart_y_offset=-14000.0,
#     playstart_z_offset=0,
#     instance_required=False,
#     scene_buildings_num=188,
#     img_save_path="japan-0.9-0.9-100",
#     img_position_log="uav_position.txt",
#     log_path="test_flight.log"
# )

# japan scene Planning collection
test_flight = CvPathPlanner(
    planner=True,
    planning_log_path=r"E:\data\contextcapture\japan-0.8-0.8-100\box\path\path0.log",
    flight_box=[0, 0, 0, 220, 210, -100],
    forward_overlap=0.9,
    lateral_overlap=0.9,
    cam_foc=8.8,
    sensor_w=13.2,
    sensor_h=8.8,
    playstart_x_offset=0,
    playstart_y_offset=0,
    playstart_z_offset=0,
    instance_required=False,
    scene_buildings_num=188,
    img_save_path="japan-0.8-0.8-100-0-box_safe_10_sample_0.85_view_10",
    img_position_log="uav_position.txt",
    log_path="test_flight.log"
)

# playstart_x_offset = -55310.0,
# playstart_y_offset = -69820.0,
# playstart_z_offset = 100.0,


# sign = test_flight.cal_shot_dis()
# if sign:
#
#     # test_flight.cal_path()
#     test_flight.generate_path()
#     test_flight.muti_uav_assign()
#     test_flight.fly_shot()
#
# else:
#     print("Modify the number of drones")


