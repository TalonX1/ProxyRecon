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


def EulerAndQuaternionTransform(intput_data):
    data_len = len(intput_data)
    angle_is_not_rad = True

    if data_len == 3:
        r = 0
        p = 0
        y = 0
        if angle_is_not_rad:  # 180 ->pi
            r = math.radians(intput_data[0])
            p = math.radians(intput_data[1])
            y = math.radians(intput_data[2])
        else:
            r = intput_data[0]
            p = intput_data[1]
            y = intput_data[2]

        sinp = math.sin(p / 2)
        siny = math.sin(y / 2)
        sinr = math.sin(r / 2)

        cosp = math.cos(p / 2)
        cosy = math.cos(y / 2)
        cosr = math.cos(r / 2)

        w = cosr * cosp * cosy + sinr * sinp * siny
        x = sinr * cosp * cosy - cosr * sinp * siny
        y = cosr * sinp * cosy + sinr * cosp * siny
        z = cosr * cosp * siny - sinr * sinp * cosy
        return [w, x, y, z]

    elif data_len == 4:

        w = intput_data[0]
        x = intput_data[1]
        y = intput_data[2]
        z = intput_data[3]

        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        if 2 * (w * y - z * x) < -1:
            p = math.asin(-1)
        elif 2 * (w * y - z * x) > 1:
            p = math.asin(1)
        else:
            p = math.asin(2 * (w * y - z * x) + 1e-10)
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        if angle_is_not_rad:  # pi -> 180
            r = math.degrees(r)
            p = math.degrees(p)
            y = math.degrees(y)
        return [r, p, y]


class SimPathPlanner:
    def __init__(
            self,
            flight_box,
            forward_overlap,
            lateral_overlap,
            cam_pix,
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
        self.client = None
        self.start_x = flight_box[0]
        self.start_y = flight_box[1]
        self.start_z = flight_box[2]

        self.end_x = flight_box[3]
        self.end_y = flight_box[4]
        self.end_z = flight_box[5]

        self.flight_xrange = abs(self.end_x - self.start_x)
        self.flight_yrange = abs(self.end_y - self.start_y)

        self.log_path = log_path

        self.uav_name = []

        self.logger = self.init_logger()
        self.uav_list = []

        # Camera parameters
        self.cam_pix = cam_pix
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
        self.path = None
        self.paths = None
        self.forward_paths = None
        self.uav_flight_idx = None
        self.distance = None
        self.velocity = 8                             # The speed cannot exceed the minimum movement step, responsible for preventing movement
        self.img_position_log = img_position_log
        self.img_save_path = img_save_path

        self.uav_path_num = None
        self.uav_extra_path_num = None

        self.init_logger()

        self.playstart_x_offset = playstart_x_offset
        self.playstart_y_offset = playstart_y_offset
        self.playstart_z_offset = playstart_z_offset

        self.airsim_setting = r"C:\Users\Talon\Documents\AirSim\settings.json"   # Airsim settings file, used to set the initial number and position of drones

        self.instance_required = instance_required
        self.scene_buildings_num = scene_buildings_num

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
        with open(self.airsim_setting, 'r') as f:
            airsim_json = json.load(f)
            # print(airsim_json)
        vehicals_num = len(airsim_json['Vehicles'])
        for i in range(vehicals_num):
            self.uav_name.append("Drone"+str(i))

        self.logger.info('Request connection to drone')
        client = airsim.MultirotorClient()
        client.confirmConnection()
        self.logger.info('Connection successful')
        for i in range(len(self.uav_name)):
            client.enableApiControl(True, self.uav_name[i])
            client.armDisarm(True, self.uav_name[i])
        self.client = client

        self.client.simGetCameraInfo("0")

        if self.instance_required:
            for i in range(self.scene_buildings_num):
                self.client.simSetSegmentationObjectID("building_" + str(i+1), i+1, True)

            self.client.simSetSegmentationObjectID("ground[\w]*", -1, True)
            self.client.simSetSegmentationObjectID("SkySphere", -1, True)

    def init_weather(self):
        wind = airsim.Vector3r(0, 0, 0)
        self.client.simSetWind(wind)
        time.sleep(1)

    def take_off(self):
        for i in self.uav_name:
            locals()[i] = self.client.takeoffAsync(vehicle_name=i)
        for i in self.uav_name:
            locals()[i] = self.client.moveToPositionAsync(0, 0, self.end_z, 20, vehicle_name=i)
        for i in self.uav_name:
            locals()[i].join()

        # d1 = self.client.takeoffAsync(vehicle_name=self.uav_name[0])
        # d1.join()

    def init_camera(self):
        # cam1_pitch_angle = -5
        # cam2_pitch_angle = -13
        # cam3_pitch_angle = -21
        # cam4_pitch_angle = -27f
        # cam5_pitch_angle = -33
        cam_angles = [[-45, 0, 0], [-45, 0, 90], [-45, 0, -90], [0, 0, -90], [-45, 0, 180]]
        for cam_idx, cam_angle in enumerate(cam_angles):
            camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0),
                                      airsim.to_quaternion(math.radians(cam_angle[0]),math.radians(cam_angle[1]), math.radians(cam_angle[2])))
            self.client.simSetCameraPose(str(cam_idx), camera_pose, vehicle_name="Drone0")

        print("Camera angle adjusted for Drone0")
        time.sleep(1)

    def cal_shot_dis(self):
        photo_w = self.sensor_w
        photo_h = self.sensor_h
        scene_w = abs(photo_w * self.end_z / self.cam_foc)
        scene_h = abs(photo_h * self.end_z / self.cam_foc)

        if self.flight_xrange >= self.flight_yrange:
            self.flight_dir = "x"
            self.stripe_width = self.y_step = scene_w * (1 - self.lateral_overlap)
            self.shot_step = self.x_step = scene_h * (1 - self.forward_overlap)
            self.path_num = self.flight_yrange // self.stripe_width + 1
            self.logger.info("Collect along the %s direction first" % self.flight_dir)
            self.logger.info("Shooting step size is % s" % self.shot_step)
            self.logger.info("The route width is %s" % self.stripe_width)

        else:
            self.flight_dir = "y"
            self.shot_step = self.y_step = scene_h * (1 - self.forward_overlap)
            self.stripe_width = self.x_step = scene_w * (1 - self.lateral_overlap)
            self.path_num = self.flight_xrange // self.stripe_width + 1
            self.logger.info("Collect along the %s direction first" % self.flight_dir)
            self.logger.info("Shooting step size is % s" % self.shot_step)
            self.logger.info("The route width is %s" % self.stripe_width)

        self.velocity = 7

        self.logger.info("velocity %s" % self.velocity)
        self.logger.info("Expected to fly %s lanes" % self.path_num)

        with open(self.airsim_setting, 'r') as f:
            airsim_json = json.load(f)
            # print(airsim_json)

        vehicals_num = len(airsim_json['Vehicles'])
        if vehicals_num == self.path_num:
            self.init_uav()
            self.init_weather()
            # self.init_camera()
            self.take_off()
            return True

        elif vehicals_num < self.path_num:
            for i in range(vehicals_num, int(self.path_num)):
                airsim_json['Vehicles']['Drone'+str(i)] = airsim_json['Vehicles']['Drone0']

            for j in range(int(self.path_num)):
                airsim_json['Vehicles']['Drone'+str(j)]["Y"] = 2*j

            with open('settings.json', 'w') as f:
                json.dump(airsim_json, f)

            with open('settings.json', 'r') as f:
                airsim_json_modify = json.load(f)
            for j in range(int(self.path_num)):
                airsim_json_modify['Vehicles']['Drone'+str(j)]["Y"] = 2*j

            with open('settings.json', 'w') as f:
                json.dump(airsim_json_modify, f)

            return False
        else:
            for i in range(int(self.path_num), vehicals_num):
                del airsim_json['Vehicles']['Drone' + str(i)]

            with open('settings.json', 'w') as f:
                json.dump(airsim_json, f)
            return False

    def generate_path(self):
        if self.flight_dir == 'y':
            path_num = int((self.end_x - self.start_x)//self.stripe_width + 1)
            path_shoot_num = int((self.end_y - self.start_y)//self.shot_step + 1)

            self.forward_paths = []
            for i in range(path_num):
                path = []
                for j in range(path_shoot_num):
                    path.append([i * self.stripe_width, j * self.shot_step, self.end_z])
                self.forward_paths.append(path)
        else:
            path_num = int((self.end_y - self.start_y) // self.stripe_width + 1)
            path_shoot_num = int((self.end_x - self.start_x)//self.shot_step + 1)

            self.forward_paths = []
            for i in range(path_num):
                path = []
                for j in range(path_shoot_num):
                    path.append([j * self.shot_step, i * self.stripe_width, self.end_z])
                self.forward_paths.append(path)

        self.logger.info("Flight speed is %s" % self.velocity)
        self.logger.info("Expected to fly %s lanes" % self.path_num)
        self.logger.info("Number of viewpoints on a path %s" % path_shoot_num)

    def cal_path(self):
        self.path = []
        self.paths = []
        self.distance = 0
        x = self.start_x
        y = self.start_y
        # self.path.append(Vector3r(self.start_x, self.start_y, self.end_z))
        self.path.append([self.start_x, self.start_y, self.end_z])
        self.distance += (self.end_z - self.start_z)
        if self.flight_dir == 'y':
            while x < self.end_x:
                if y < self.end_y:
                    while y < self.end_y:
                        y += self.shot_step
                        self.distance += self.shot_step
                        # self.path.append(Vector3r(x, y, self.end_z))
                        self.path.append([x, y, self.end_z])
                        print([x, y, self.end_z])
                    self.paths.append(self.path)
                    self.path = []
                else:
                    while y > self.start_y:
                        y -= self.shot_step
                        self.distance += self.shot_step
                        # self.path.append(Vector3r(x, y, self.end_z))
                        self.path.append([x, y, self.end_z])
                        print([x, y, self.end_z])
                    self.paths.append(self.path)
                    self.path = []
                x += self.stripe_width
                self.distance += self.stripe_width
                # self.path.append(Vector3r(x, y, self.end_z))
                self.path.append([x, y, self.end_z])
        elif self.flight_dir == 'x':
            while y < self.end_y:
                if x < self.end_x:
                    while x < self.end_x:
                        x += self.shot_step
                        self.distance += self.shot_step
                        # self.path.append(Vector3r(x, y, self.end_z))
                        self.path.append([x, y, self.end_z])
                    self.paths.append(self.path)
                    self.path = []
                else:
                    while x > self.start_x:
                        x -= self.shot_step
                        self.distance += self.shot_step
                        # self.path.append(Vector3r(x, y, self.end_z))
                        self.path.append([x, y, self.end_z])
                    self.paths.append(self.path)
                    self.path = []
                y += self.stripe_width
                self.distance += self.stripe_width
                # self.path.append(Vector3r(x, y, self.end_z))
                self.path.append([x, y, self.end_z])

    def muti_uav_assign(self):
        self.uav_path_num = len(self.forward_paths) // len(self.uav_name)
        self.uav_extra_path_num = len(self.forward_paths) % len(self.uav_name)

        self.uav_flight_idx = []
        path_idx = 0
        for i in range(len(self.uav_name)):
            if self.uav_extra_path_num != 0:
                if i + 1 <= self.uav_extra_path_num:
                    self.uav_flight_idx.append([path_idx + j for j in range(self.uav_path_num)])
                    self.uav_flight_idx[i].append(path_idx + self.uav_path_num)
                    path_idx += self.uav_path_num + 1
                else:
                    self.uav_flight_idx.append([path_idx + j for j in range(self.uav_path_num)])
                    path_idx += self.uav_path_num
            else:
                self.uav_flight_idx.append([path_idx + j for j in range(self.uav_path_num)])
                path_idx += self.uav_path_num

        for i in range(len(self.uav_flight_idx)):
            for j in range(len(self.uav_flight_idx[i])):
                if j % 2 != 0:
                    self.forward_paths[self.uav_flight_idx[i][j]].reverse()
        self.paths = self.forward_paths

    def shot_save(self):
        for uav_name in self.uav_name:
            if not self.instance_required:
                responses = self.client.simGetImages([
                    airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest('front_right', airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest('front_left', airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest('bottom_center', airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest('back_center', airsim.ImageType.Scene, False, False),
                ], vehicle_name=uav_name)

                for idx, response in enumerate(responses):
                    # x_val, y_val, z_val, w_val = response.camera_orientation
                    camera_position = response.camera_position
                    camera_orientation_quaternion = response.camera_orientation
                    # print('camera_orientation_quaternion',camera_orientation_quaternion)
                    # print(type(camera_orientation_quaternion))
                    w_val = camera_orientation_quaternion.w_val
                    x_val = camera_orientation_quaternion.x_val
                    y_val = camera_orientation_quaternion.y_val
                    z_val = camera_orientation_quaternion.z_val
                    camera_orientation_euler = EulerAndQuaternionTransform([w_val, x_val, y_val, z_val])
                    print('camera_orientation_euler', camera_orientation_euler)

                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(response.height, response.width, 3)
                    filename = os.path.join(self.img_save_path, str(idx), uav_name + "-" + str(idx) + "-" + time.strftime("%Y%m%d-%H%M%S"))
                    airsim.write_png(os.path.normpath(filename + '.png'), img_rgb)

                    with open(self.img_position_log, "a+") as f:
                        f.write(filename + ".png")
                        f.write(',')
                        f_csv = csv.writer(f, delimiter=',')
                        info = [camera_position.x_val + self.playstart_x_offset * 0.01,
                                camera_position.y_val + self.playstart_y_offset * 0.01,
                                -camera_position.z_val + self.playstart_z_offset * 0.01]
                        info.extend(camera_orientation_euler)
                        f_csv.writerow(info)
                        # f_csv.writerow(camera_orientation_euler)
            else:
                responses = self.client.simGetImages([
                    airsim.ImageRequest('bottom_center', airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)
                ], vehicle_name=uav_name)  # scene vision image in uncompressed RGBA array

                for idx, response in enumerate(responses):
                    # x_val, y_val, z_val, w_val = response.camera_orientation
                    # print("相机角度：", EulerAndQuaternionTransform([x_val, y_val, z_val, w_val]))
                    camera_position = response.camera_position
                    camera_orientation_quaternion = response.camera_orientation
                    # print('camera_orientation_quaternion',camera_orientation_quaternion)
                    # print(type(camera_orientation_quaternion))
                    w_val = camera_orientation_quaternion.w_val
                    x_val = camera_orientation_quaternion.x_val
                    y_val = camera_orientation_quaternion.y_val
                    z_val = camera_orientation_quaternion.z_val
                    camera_orientation_euler = EulerAndQuaternionTransform([w_val, x_val, y_val, z_val])
                    print('camera_orientation_euler', camera_orientation_euler)

                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(response.height, response.width, 3)
                    filename = os.path.join(self.img_save_path, str(idx),
                                            uav_name + "-" + str(idx) + "-" + time.strftime("%Y%m%d-%H%M%S"))
                    airsim.write_png(os.path.normpath(filename + '.png'), img_rgb)

                    with open(self.img_position_log, "a+") as f:
                        f.write(filename + ".png")
                        f.write(',')
                        f_csv = csv.writer(f, delimiter=',')
                        info = [camera_position.x_val + self.playstart_x_offset * 0.01,
                                camera_position.y_val + self.playstart_y_offset * 0.01,
                                -camera_position.z_val + self.playstart_z_offset * 0.01]
                        info.extend(camera_orientation_euler)
                        f_csv.writerow(info)
                        # f_csv.writerow(camera_orientation_euler)

    def fly_shot(self):
        # result = self.client.moveOnPath(self.path, self.velocity, trip_time, DrivetrainType.ForwardOnly,
        #                                 YawMode(False, 0), lookahead, 1)

        if not os.path.exists(self.img_save_path):
            os.mkdir(self.img_save_path)

        for i in range(5):
            if not os.path.exists(os.path.join(self.img_save_path, str(i))):
                os.mkdir(os.path.join(self.img_save_path, str(i)))

        for flight_time in range(self.uav_path_num):
            # if flight_time//2 !=0:
            #     for uav_name in self.uav_name:
            #         locals()[uav_name] = self.client.rotateToYawAsync(-180, vehicle_name=uav_name)
            #     for uav_name in self.uav_name:
            #         locals()[uav_name].join()
            # else:
            #     for uav_name in self.uav_name:
            #         locals()[uav_name] = self.client.rotateToYawAsync(0, vehicle_name=uav_name)
            #     for uav_name in self.uav_name:
            #         locals()[uav_name].join()

            for path_point in range(len(self.paths[0])):
                for uav_idx in range(len(self.uav_name)):
                    path_idx = self.uav_flight_idx[uav_idx][flight_time]
                    path_point_idx = self.paths[path_idx][path_point]
                    locals()[self.uav_name[uav_idx]] = self.client.moveToPositionAsync(
                        path_point_idx[0],
                        path_point_idx[1],
                        path_point_idx[2],
                        self.velocity,
                        vehicle_name=self.uav_name[
                           uav_idx])
                for uav_idx in range(len(self.uav_name)):
                    locals()[self.uav_name[uav_idx]].join()

                time.sleep(15)
                self.shot_save()

        if self.uav_extra_path_num != 0:
            for path_point in range(len(self.paths[0])):
                for uav_idx in range(self.uav_extra_path_num):
                    path_point_idx = self.paths[self.uav_flight_idx[uav_idx][self.uav_path_num]][path_point]
                    locals()[self.uav_name[uav_idx]] = self.client.moveToPositionAsync(
                        path_point_idx[0],
                        path_point_idx[1],
                        path_point_idx[2],
                        self.velocity,
                        vehicle_name=self.uav_name[
                           uav_idx])
                for uav_idx in range(self.uav_extra_path_num):
                    locals()[self.uav_name[uav_idx]].join()
                time.sleep(15)
                self.shot_save()

        for uav_idx, uav_name in enumerate(self.uav_name):
            locals()[self.uav_name[uav_idx]] = self.client.moveToPositionAsync(
                0,
                uav_idx+2*uav_idx,
                self.end_z,
                self.velocity,
                vehicle_name=self.uav_name[
                   uav_idx])
        for uav_name in self.uav_name:
            locals()[uav_name].join()

        for uav_idx, uav_name in enumerate(self.uav_name):
            locals()[self.uav_name[uav_idx]] = self.client.moveToPositionAsync(0,
                                                                           uav_idx+2*uav_idx,
                                                                           -2,
                                                                           self.velocity,
                                                                           vehicle_name=self.uav_name[
                                                                               uav_idx])
        for uav_name in self.uav_name:
            locals()[uav_name].join()

        for uav_name in self.uav_name:
            locals()[uav_name] = self.client.goHomeAsync(vehicle_name=uav_name)

        time.sleep(3)
        for uav_name in self.uav_name:
            locals()[uav_name].join()
            self.client.armDisarm(False, uav_name)
            self.client.reset()
            self.client.enableApiControl(False, uav_name)


# test_flight = SimPathPlanner(
#     flight_box=[0, 0, 0, 705, 1260, -250],
#     forward_overlap=0.8,
#     lateral_overlap=0.7,
#     uav_num=6,
#     cam_pix=[5472, 3648],
#     pix_meta=2.41,
#     cam_foc=8.8,
#     cam_dpi=[72, 72],
#     sensor_w=13.2,
#     sensor_h=8.8,
#     playstart_x_offset=-41260.0,
#     playstart_y_offset=-68060.0,
#     playstart_z_offset=100.0,
#     instance_required=True,
#     scene_buildings_num=188,
#     img_save_path="inst-monaco-0.8-0.7-250",
#     img_position_log="uav_position.txt",
#     log_path="test_flight.log"
# )

test_flight = SimPathPlanner(
    flight_box=[0, 0, 0, 320, 320, -150],
    forward_overlap=0.8,
    lateral_overlap=0.8,
    uav_num=6,
    cam_pix=[5472, 3648],
    pix_meta=2.41,
    cam_foc=8.8,
    cam_dpi=[72, 72],
    sensor_w=13.2,
    sensor_h=8.8,
    playstart_x_offset=-41260.0,
    playstart_y_offset=-68060.0,
    playstart_z_offset=100.0,
    instance_required=True,
    scene_buildings_num=188,
    img_save_path="city_1_op_0.8",
    img_position_log="uav_position.txt",
    log_path="test_flight.log"
)
sign = test_flight.cal_shot_dis()
if sign:

    # test_flight.cal_path()
    test_flight.generate_path()
    test_flight.muti_uav_assign()
    test_flight.fly_shot()

else:
    print("modify uav num")


