import os
import yaml


class Option:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, "r", encoding='UTF-8'))

        # input info
        self.input_pcd_path = self.config["input_pcd_path"]
        self.inst_scalar = self.config["inst_scalar"]

        # save path
        self.output_inst_color = self.config["output_inst_color"]
        self.save_path = os.path.join('experiment', self.config["save_path"])
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.color_file_path = os.path.join(self.save_path, self.config["color_file_path"])
        self.obj_save_path = os.path.join(self.save_path, self.config["obj_save_path"])
        self.footprint_save_path = os.path.join(self.save_path, self.config["footprint_save_path"])
        self.ground_segment_pcd_save_path = os.path.join(self.save_path, self.config['ground_segment_pcd_save_path'])
        self.building_segment_pcd_save_path = os.path.join(self.save_path, self.config['building_segment_pcd_save_path'])

        self.boundary_mode = self.config["boundary_mode"]
        self.output_boundary_path = os.path.join(self.save_path, self.config['output_boundary_path'])

        # dbscan cluster
        self.need_cluster = self.config["need_cluster"]
        self.segment_z = self.config["segment_z"]
        self.dbscan_eps = self.config["dbscan_eps"]
        self.dbscan_min_points = self.config["dbscan_min_points"]

        # layer
        self.ground_z = self.config["ground_z"]
        self.layer_mode = self.config["layer_mode"]
        self.layer_num = self.config["layer_num"]
        self.area_threshold = self.config["area_threshold"]

    def _prepare(self):
        self.save_path = os.path.join(self.save_path, "eps{}_min-num{}_{}_layer-num{}_area{}".format(
            self.dbscan_eps, self.dbscan_min_points, self.layer_mode, self.layer_num, self.area_threshold
        ))

    def check_path(self):
        if os.path.exists(self.save_path):
            print("file exist: {}".format((self.save_path)))
            self.obj_save_path = os.path.join(self.save_path, self.obj_save_path)
            self.footprint_save_path = os.path.join(self.save_path, self.footprint_save_path)