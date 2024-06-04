import numpy as np
from .basic_proess import project_pcd_to_zplane, polygon_area, get_alpha_shape_xyz0, get_convexhull_xyz0
# from .basic_proess import get_list_max_last_index
import logging
import copy


class Layer:

    def __init__(self, pcd_xyz, layer_num, area_ration, poly_mode='alpha-shape', z_ground=None):

        pcd_xyz = np.array(pcd_xyz)
        self.layer_num = layer_num
        self.area_ration = area_ration
        self.poly_mode = poly_mode                      # alpha-shape,convex-hull

        self.z_min = np.min(pcd_xyz[:, 2])
        self.z_max = np.max(pcd_xyz[:, 2])
        self.z_ground = z_ground

        self.logger = None
        self.init_logger()

        layer_xyz_project = self.slicing(pcd_xyz)
        self.build_result = None

        result = self.extract_main_layer_shape(layer_xyz_project)

        if result != 0:
            layer_shape_index, layer_shape_xyz0, layer_shape_areas, main_layer_shape_index, main_layer_shape_xyz0 = result

            stack_layer_shape_xyz0, main_layer_shape_xyz0 = self.main_layer_shift(main_layer_shape_xyz0)
            stack_poly_shape_xyz0, main_poly_shape_xyz0, poly_points_num = \
                self.extract_layer_poly(stack_layer_shape_xyz0, main_layer_shape_xyz0)

            self.main_layer_shape_xyz0 = main_layer_shape_xyz0
            self.stack_layer_shape_xyz0 = stack_layer_shape_xyz0
            self.main_poly_shape_xyz0 = main_poly_shape_xyz0
            self.stack_poly_shape_xyz0 = stack_poly_shape_xyz0
            self.poly_points_num = poly_points_num
            self.build_result = True
        else:
            self.build_result = False

    def init_logger(self):
        self.logger = logging.getLogger('Layer')
        # self.logger.setLevel(logging.DEBUG)
        self.logger.setLevel(logging.ERROR)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)
        if self.z_ground is None:
            self.logger.info('Start, no z_ground')

        self.logger.info('Start! ' + ' layer_num:  ' + str(self.layer_num) + '  area_ration:  ' + str(
            self.area_ration) + '  z_ground:  ' + str(self.z_ground))

    def slicing(self, pcd_xyz):
        pcd_z = pcd_xyz[:, 2]
        z_min = np.min(pcd_z)
        z_max = np.max(pcd_z)
        layer_dist = (z_max - z_min) / self.layer_num
        layer_xyz_project = []
        for i in range(self.layer_num):
            layer_z = z_max - layer_dist * (i + 1)
            layer_up_pcd_index = (pcd_z >= layer_z)
            layer_up_pcd_xyz = pcd_xyz[layer_up_pcd_index]
            layer_up_pcd_xyz0 = project_pcd_to_zplane(layer_up_pcd_xyz, layer_z)
            layer_xyz_project.append(layer_up_pcd_xyz0)

        layer_xyz_project = np.array(layer_xyz_project, dtype=object)
        self.logger.info('Slicing complete!')
        return layer_xyz_project

    def extract_main_layer_shape(self, layer_xyz_project):
        # alpha-shape
        error_layer_num = 0
        main_layer_shape_index = []
        layer_shape_xyz0 = []
        layer_poly_num = []
        layer_shape_areas = []
        layer_shape_index = []
        if self.poly_mode == 'alpha-shape':
            for i in range(len(layer_xyz_project)):
                if get_alpha_shape_xyz0(layer_xyz_project[i]) is not None:
                    main_layer_shape_index.append(i)
                    break
                else:
                    layer_shape_xyz0.append([])
                    layer_poly_num.append(0)

            if len(main_layer_shape_index) == 0:
                print("build alpha shape fail")
                return 0
            else:
                print("error")

            for i in range(main_layer_shape_index[-1], self.layer_num):
                print(i)
                if get_alpha_shape_xyz0(layer_xyz_project[i]) is not None:
                    polygon_num, one_layer_shape_xyz0 = get_alpha_shape_xyz0(layer_xyz_project[i])
                    self.layer_poly_num.append(polygon_num)
                    if polygon_num == 1:
                        one_layer_shape_area = polygon_area(one_layer_shape_xyz0[0])
                        layer_shape_index.append(i)
                        layer_shape_xyz0.append(one_layer_shape_xyz0)
                        layer_shape_areas.append(one_layer_shape_area)
                    else:
                        layer_shape_area = 0
                        for j in range(polygon_num):
                            one_layer_shape_area += polygon_area(one_layer_shape_xyz0[j])
                        layer_shape_index.append(i)
                        layer_shape_xyz0.append(one_layer_shape_xyz0)
                        layer_shape_areas.append(one_layer_shape_area)
                else:
                    layer_shape_index.append(i)
                    layer_shape_xyz0.append([])
                    layer_shape_areas.append(layer_shape_areas[-1])

        elif self.poly_mode == 'convex-hull':
            for i in range(len(layer_xyz_project)):
                if get_convexhull_xyz0(layer_xyz_project[i]) is not None:
                    main_layer_shape_index.append(i)
                    break
                else:
                    layer_shape_xyz0.append([])
                    layer_poly_num.append(0)

            if len(main_layer_shape_index) == 0:
                print("build convexhull fail")
                return 0

            for i in range(main_layer_shape_index[-1], self.layer_num):
                one_layer_shape_xyz0 = get_convexhull_xyz0(layer_xyz_project[i])
                one_layer_shape_area = polygon_area(one_layer_shape_xyz0)
                layer_shape_index.append(i)
                layer_shape_xyz0.append(one_layer_shape_xyz0)
                layer_shape_areas.append(one_layer_shape_area)

        for i in range(len(layer_shape_areas) - 1):
            area_ration = layer_shape_areas[i + 1] / layer_shape_areas[i]
            if area_ration >= self.area_ration:
                main_layer_shape_index[-1] = layer_shape_index[i]
                main_layer_shape_index.append(layer_shape_index[i + 1])
            else:
                main_layer_shape_index[-1] = layer_shape_index[i]

        main_layer_shape_index[-1] = len(layer_xyz_project) - 1

        layer_shape_index = np.array(layer_shape_index)
        layer_shape_xyz0 = np.array(layer_shape_xyz0, dtype=object)
        layer_shape_areas = np.array(layer_shape_areas)

        main_layer_shape_index = np.array(main_layer_shape_index)
        main_layer_shape_xyz0 = np.array(layer_shape_xyz0[main_layer_shape_index])

        self.logger.info('Extract main layers complete!')

        if self.z_ground is None or self.z_ground > self.z_min:
            self.z_ground = self.z_min

        return layer_shape_index, layer_shape_xyz0, layer_shape_areas,\
            main_layer_shape_index, main_layer_shape_xyz0

    def main_layer_shift(self, main_layer_shape_xyz0):
        stack_layer_shape_xyz0 = copy.deepcopy(main_layer_shape_xyz0)

        if self.poly_mode == "convex-hull":
            stack_layer_shape_xyz0[0][:, -1] = self.z_max
            main_layer_shape_xyz0[-1][:, -1] = self.z_ground

            for i in range(1, len(main_layer_shape_xyz0)):
                stack_layer_shape_xyz0[i][:, -1] = main_layer_shape_xyz0[i - 1][0, -1]

        elif self.poly_mode == "alpha-shape":
            for i in range(len(stack_layer_shape_xyz0[0])):
                stack_layer_shape_xyz0[0][i][:, -1] = self.z_max

            for i in range(len(self.main_layer_shape_xyz0[-1])):
                main_layer_shape_xyz0[-1][i][:, -1] = self.z_ground

            for i in range(1, len(main_layer_shape_xyz0)):
                for j in range(len(main_layer_shape_xyz0[i])):
                    stack_layer_shape_xyz0[i][j][:, -1] = main_layer_shape_xyz0[i-1][0][0, -1]

        return stack_layer_shape_xyz0, main_layer_shape_xyz0

    def extract_layer_poly(self, stack_layer_shape_xyz0, main_layer_shape_xyz0):
        stack_poly_shape_xyz0 = []
        main_poly_shape_xyz0 = []
        poly_points_num = []

        if self.poly_mode == "alpha-shape":
            for i in range(len(self.stack_layer_shape_xyz0)):
                for j in range(len(self.stack_layer_shape_xyz0[i])):
                    stack_poly_shape_xyz0.append(stack_layer_shape_xyz0[i][j])
                    main_poly_shape_xyz0.append(main_layer_shape_xyz0[i][j])
                    poly_points_num.append(len(stack_layer_shape_xyz0[i][j]))
        elif self.poly_mode == 'convex-hull':
            for i in range(len(stack_layer_shape_xyz0)):
                stack_poly_shape_xyz0.append(stack_layer_shape_xyz0[i])
                main_poly_shape_xyz0.append(main_layer_shape_xyz0[i])
                poly_points_num.append(len(stack_layer_shape_xyz0[i]))

        return stack_poly_shape_xyz0, main_poly_shape_xyz0, poly_points_num




