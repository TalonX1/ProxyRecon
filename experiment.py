import numpy as np
import open3d as o3d
from layer.mesh import Mesh
from layer.basic_proess import get_alpha_shape_xyz0
from layer.basic_proess import get_convexhull_xyz0
import matplotlib.pyplot as plt
from option import Option
from scipy.spatial import KDTree
from layer.basic_proess import point_in_poly
from layer.basic_proess import poly_bounding_box
from tqdm import tqdm
import copy
import sys


class Experiment:
    def __init__(self, settings: Option):
        self.pcd_inst_dict_key = None
        self.pcd_inst_dict = None
        self.settings = settings
        self.inst_poly_2d = None
        self.building_inst_color = None

        if self.settings.inst_scalar:
            self.read_segment_pcd(self.settings.input_pcd_path)
            self.inst_scalar2_inst_poly()
            self.build_segment_proxy()
        else:
            self.input_pcd = o3d.io.read_point_cloud(self.settings.input_pcd_path)
            if self.settings.layer_mode == 'convex-hull':
                self.input_pcd_array = np.around(np.array(self.input_pcd.points), 2)
            elif self.settings.layer_mode == 'alpha-shape':
                self.input_pcd_array = np.around(np.array(self.input_pcd.points), 1)

            if self.settings.need_cluster:
                self.building_pcd_array = self.input_pcd_array[self.input_pcd_array[:, 2] >= self.settings.segment_z]
                self.ground_pcd_array = self.input_pcd_array[self.input_pcd_array[:, 2] < self.settings.segment_z]
                self.cluster()
                self.build_segment_proxy()
            else:
                self.build_whole_proxy()

    def read_segment_pcd(self, file_path):
        pcd_inst_dict = {}
        pcd_inst_array = np.loadtxt(file_path, delimiter=',')

        xyz = pcd_inst_array[:, :3]
        labels = pcd_inst_array[:, 3].astype(int)

        for point, label in zip(xyz, labels):
            if label not in pcd_inst_dict:
                pcd_inst_dict[label] = []
            pcd_inst_dict[label].append(point)

        for label in pcd_inst_dict:
            pcd_inst_dict[label] = np.array(pcd_inst_dict[label])

        self.building_inst_num = max(pcd_inst_dict.keys())
        self.pcd_inst_dict = pcd_inst_dict
        self.building_inst_color = np.random.randint(0, 256, (len(self.pcd_inst_dict), 3))

    def cluster(self):
        building_pcd_array_copy = copy.deepcopy(self.building_pcd_array)
        building_pcd_array_copy[:, 2] = 0
        building_pcd = o3d.geometry.PointCloud()
        building_pcd.points = o3d.utility.Vector3dVector(building_pcd_array_copy)

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                building_pcd.cluster_dbscan(eps=self.settings.dbscan_eps, min_points=self.settings.dbscan_min_points,
                                            print_progress=True))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        building_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        o3d.visualization.draw_geometries([building_pcd])

        all_cluster_points_index = []
        for i in range(max_label + 1):
            all_cluster_points_index.append(np.where(labels == i)[0])

        all_cluster_xyz = {}
        for i in range(len(all_cluster_points_index)):
            one_cluster_xyz = self.building_pcd_array[all_cluster_points_index[i]]
            all_cluster_xyz[i] = np.array(one_cluster_xyz)

        self.pcd_inst_dict = all_cluster_xyz
        self.building_inst_num = max(all_cluster_xyz.keys())
        outlier_points_index = np.where(labels == -1)
        outlier_points_xyz = np.array(self.building_pcd_array[outlier_points_index])
        self.new_ground_pcd_array = np.vstack((self.ground_pcd_array, outlier_points_xyz))
        self.building_inst_color = np.random.randint(0, 256, (len(all_cluster_xyz), 3))

    def inst_scalar2_inst_poly(self):
        assert self.settings.inst_scalar is True
        inst_points_2d = np.loadtxt(self.settings.input_pcd_path, delimiter=',')
        inst_num = int(np.max(inst_points_2d[:, 3]))
        inst_poly_2d = []
        print("cal 2d boundary")
        for i in tqdm(range(inst_num + 1), desc="Processing", unit="item"):
            one_inst_point2d = inst_points_2d[inst_points_2d[:, 3] == i]
            if self.settings.boundary_mode == "convex-hull":
                points_xyz0 = np.delete(one_inst_point2d, 3, 1)
                polygon_xyz = get_convexhull_xyz0(points_xyz0)
                if polygon_xyz is not None:
                    inst_poly_2d.append(polygon_xyz)
            elif self.settings.boundary_mode == "alpha-shape":
                polygon_num, polygon_xyz = get_alpha_shape_xyz0(np.delete(one_inst_point2d, 3, 1))
                if polygon_num == 1:
                    inst_poly_2d.append(polygon_xyz[0])
                else:
                    polygon_xyz = get_convexhull_xyz0(np.delete(one_inst_point2d, 3, 1))
                    inst_poly_2d.append(polygon_xyz)
        self.inst_poly_2d = inst_poly_2d
        with open(self.settings.output_boundary_path, 'w') as f:
            for i, poly in enumerate(inst_poly_2d):
                f.write(f'polygon {i}\n')
                for point in poly:
                    f.write(f'{point[0]},{point[1]}\n')

    def build_segment_proxy(self):
        print("build proxy")
        with open(self.settings.color_file_path, "a+") as f1:
            for color in self.building_inst_color:
                f1.write(" ".join(map(str, color)) + "\n")
        global_idx = 0
        for i in tqdm(range(self.building_inst_num+1), desc="Processing", unit="item"):
            proxy = Mesh(
                pcd_xyz=self.pcd_inst_dict[i],
                layer_num=self.settings.layer_num,
                area_ration=self.settings.area_threshold,
                save_path=self.settings.obj_save_path,
                faces_global_index=global_idx,
                poly_mode=self.settings.layer_mode,
                z_ground=self.settings.ground_z,
                output_inst_color=self.settings.output_inst_color,
                one_building_inst_color=self.building_inst_color[i]
            )
            if proxy.layer.build_result:
                global_idx += len(proxy.obj.vertex)
    def build_whole_proxy(self):
        global_idx = 0
        proxy = Mesh(
            pcd_xyz=self.input_pcd_array,
            layer_num=self.settings.layer_num,
            area_ration=self.settings.area_threshold,
            save_path=self.settings.obj_save_path,
            faces_global_index=global_idx,
            poly_mode=self.settings.layer_mode,
            z_ground=self.settings.ground_z
        )
        if proxy.layer.build_result:
            global_idx += len(proxy.obj.vertex)


if __name__ == "__main__":
    # exp = Experiment(settings=Option("config/jp_1_inst_config.yaml"))
    # exp = Experiment(settings=Option("config/longhua_inst_config.yaml"))
    # exp = Experiment(settings=Option("config/longhua_inst_config.yaml"))
    exp = Experiment(settings=Option(sys.argv[1]))

