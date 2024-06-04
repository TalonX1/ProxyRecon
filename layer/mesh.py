import numpy as np
from .layer import Layer
from .obj import Obj
import copy
from ground.base import get_context
from sect.triangulation import Triangulation
from shapely.geometry import Polygon


class Mesh:
    def __init__(self,
                 pcd_xyz,
                 layer_num,
                 area_ration,
                 save_path,
                 faces_global_index,
                 poly_mode='convex-hull',
                 multi_buildings=True,
                 z_ground=None,
                 output_inst_color=False,
                 one_building_inst_color=[200, 200, 200]
                 ):

        self.poly_mode = poly_mode
        self.layer = Layer(pcd_xyz, layer_num, area_ration, poly_mode, z_ground)
        self.save_path = save_path

        self.output_inst_color = output_inst_color
        self.one_building_inst_color = one_building_inst_color

        self.xy2_global_idx = {}            # 坐标点到全局索引的字典映射{[x,y]:idx}，用于构建拓扑
        self.faces_global_index = faces_global_index  # 面片存储开始索引
        self.obj = None

        # 构建footprint文件
        self.footprint_path = None
        self.footprint_z = []
        self.footprint_xy = []

        if self.layer.build_result:
            self.init_save_path()

    def init_save_path(self):
        if self.save_path.endswith('obj'):
            obj_path = self.save_path
            if self.poly_mode == 'convex-hull':
                vertex, facade, roof = self.triangulation_mesh()
            elif self.poly_mode == 'alpha-shape':
                vertex, facade, roof = self.none_triangulation_mesh()

            obj_vertex_path = obj_path.strip('.obj') + "_vertex.obj"
            obj_face_path = obj_path.strip('.obj') + "_face.obj"
            face = copy.deepcopy(roof)
            face.extend(facade)
            self.obj = Obj(vertex, face, self.faces_global_index, obj_vertex_path, obj_face_path, obj_path, self.output_inst_color, self.one_building_inst_color)
            self.obj.write_info()

        elif self.save_path.endswith('footprint'):
            self.footprint_path = self.save_path
            self.generate_footprint()

    def generate_footprint(self):
        self.footprint_z.append([self.layer.main_layer_shape_xyz0[0, 0, -1], self.layer.z_max])
        self.footprint_xy.append(np.delete(self.layer.main_layer_shape_xyz0[0], 2, 1))
        for i in range(1, len(self.layer.main_layer_shape_xyz0)):
            self.footprint_z.append(
                [self.layer.main_layer_shape_xyz0[i, 0, -1], self.layer.main_layer_shape_xyz0[i - 1, 0, -1]])
            self.footprint_xy.append(np.delete(self.layer.main_layer_shape_xyz0[i], 2, 1))

    def none_triangulation_mesh(self):
        # 添加顶点
        # self.layer.stack_poly_shape_xyz0 为ndarray,内容为层+点
        # for poly in range(len(self.layer.stack_poly_shape_xyz0)):

        vertex = []
        facade = []
        roof = []
        roof_index = [0]

        for poly in range(len(self.layer.stack_poly_shape_xyz0)):
            vertex.extend(self.layer.stack_poly_shape_xyz0[poly])
            vertex.extend(self.layer.main_poly_shape_xyz0[poly])

        for poly in range(len(self.layer.poly_points_num) - 1):
            roof_index.append(roof_index[-1] + 2 * self.layer.poly_points_num[poly])

        for poly in range(len(self.layer.stack_poly_shape_xyz0)):
            start_index = roof_index[poly] + self.faces_global_index                         # 每一个多边形的初始索引
            one_roof_face = []                                                                    # 一个顶面
            for point in range(len(self.layer.stack_poly_shape_xyz0[poly])):
                one_facade_face = []
                one_roof_face.append(start_index + point + 1)                                     # 顶部面片构造
                # 如果不是最后一个点
                if point + 1 != len(self.layer.stack_poly_shape_xyz0[poly]):                      # 侧面面片构造
                    one_facade_face.append(start_index + point + 1)
                    one_facade_face.append(
                        start_index + point + len(self.layer.stack_poly_shape_xyz0[poly]) + 1)
                    one_facade_face.append(
                        start_index + point + len(self.layer.stack_poly_shape_xyz0[poly]) + 2)
                    one_facade_face.append(start_index + point + 2)
                else:
                    one_facade_face.append(start_index + point + 1)
                    one_facade_face.append(
                        start_index + point + 1 + len(self.layer.stack_poly_shape_xyz0[poly]))
                    one_facade_face.append(start_index + 1 + len(self.layer.stack_poly_shape_xyz0[poly]))
                    one_facade_face.append(start_index + 1)
                facade.append(one_facade_face)
            roof.append(one_roof_face)

        return vertex, facade, roof

    def triangulation_mesh(self):
        roof_index_out = [0]
        roof_index_inner = [0]
        vertex = []
        facade = []
        roof = []

        for poly in range(len(self.layer.stack_layer_shape_xyz0)):
            vertex.extend(self.layer.stack_layer_shape_xyz0[poly])
            vertex.extend(self.layer.main_layer_shape_xyz0[poly])

        for poly in range(len(self.layer.stack_layer_shape_xyz0)):
            roof_index_inner.append(roof_index_out[-1] + self.layer.poly_points_num[poly])
            roof_index_out.append(roof_index_out[-1] + 2 * self.layer.poly_points_num[poly])

        for poly_idx in range(len(self.layer.stack_layer_shape_xyz0)):
            out_start_index = roof_index_out[poly_idx] + self.faces_global_index
            inner_start_index = roof_index_inner[poly_idx] + self.faces_global_index
            one_roof_face = []

            if poly_idx != 0:
                inner_poly_xyz0 = self.layer.main_layer_shape_xyz0[poly_idx-1]
                out_poly_xyz0 = self.layer.stack_layer_shape_xyz0[poly_idx]
                assert inner_poly_xyz0[0][-1] == out_poly_xyz0[0][-1]
                inner_poly_xy = np.delete(inner_poly_xyz0, 2, 1).tolist()
                out_poly_xy = np.delete(out_poly_xyz0, 2, 1).tolist()

                if self.layer.poly_mode == 'convex-hull':
                    context = get_context()
                    MyContour, MyPoint = context.contour_cls, context.point_cls
                    MyPolygon = context.polygon_cls

                    inner_poly_contour = MyContour([MyPoint(inner_poly_xy[i][0], inner_poly_xy[i][1]) for i in range(len(inner_poly_xy))])
                    out_poly_contour = MyContour([MyPoint(out_poly_xy[i][0], out_poly_xy[i][1]) for i in range(len(out_poly_xy))])

                    # tris = Triangulation.constrained_delaunay(
                    #     MyPolygon(MyContour([MyPoint(0, 0), MyPoint(5, 0), MyPoint(5, 5), MyPoint(0, 5)]),
                    #             [MyContour([MyPoint(1, 1), MyPoint(4, 1), MyPoint(4, 4), MyPoint(1, 4)])]),
                    #     context=context).triangles()

                    triangles = Triangulation.constrained_delaunay(
                        MyPolygon(out_poly_contour, [inner_poly_contour]), context=context).triangles()

                    if len(triangles) != 0:
                        for triangle in triangles:
                            one_triangle_face_idx = []
                            for idx in range(3):
                                triangle_point_xy = [triangle.vertices[idx].x, triangle.vertices[idx].y]
                                if triangle_point_xy in inner_poly_xy:
                                    triangle_idx = inner_poly_xy.index(triangle_point_xy)
                                    one_triangle_face_idx.append(triangle_idx + inner_start_index + 1)
                                else:
                                    triangle_idx = out_poly_xy.index(triangle_point_xy)
                                    one_triangle_face_idx.append(triangle_idx + out_start_index + 1)
                            one_roof_face.append(one_triangle_face_idx)
                    else:
                        one_poly_face_idx = []
                        for i in range(len(out_poly_xy)):
                            one_poly_face_idx.append(i + out_start_index + 1)
                        one_roof_face.append(one_poly_face_idx)
                elif self.layer.poly_mode == 'alpha_shape':
                    inner_polygon = Polygon(inner_poly_xy)
                    out_polygon = Polygon(out_poly_xy)

                    poly_union = out_polygon.union(inner_polygon)
                    poly_intersection = out_polygon.intersection(inner_polygon)
                    poly_difference = poly_union.difference(poly_intersection)
                    poly_difference_xy = []
                    poly_difference_idx = []
                    for i in range(len(poly_difference)):
                        poly_difference_xy.append(list(poly_difference.geoms[i].exterior.coords))

                    for i in range(len(poly_difference_xy)):
                        one_poly_difference_idx = []
                        for j in range(len(poly_difference_xy[i])):
                            point_idx = self.xy2_global_idx[poly_difference_xy[i][j]]
                            one_poly_difference_idx.append(point_idx)
                        poly_difference_idx.append(one_poly_difference_idx)
            else:
                first_roof_face = []
                for i in range(len(self.layer.stack_layer_shape_xyz0[0])):
                    first_roof_face.append(1+i+self.faces_global_index)
                one_roof_face.append(first_roof_face)

            for point in range(len(self.layer.stack_layer_shape_xyz0[poly_idx])):
                one_facade_face = []
                if point + 1 != len(self.layer.stack_layer_shape_xyz0[poly_idx]):
                    one_facade_face.append(out_start_index + point + 1)
                    one_facade_face.append(
                        out_start_index + point + len(self.layer.stack_layer_shape_xyz0[poly_idx]) + 1)
                    one_facade_face.append(
                        out_start_index + point + len(self.layer.stack_layer_shape_xyz0[poly_idx]) + 2)
                    one_facade_face.append(out_start_index + point + 2)
                else:
                    one_facade_face.append(out_start_index + point + 1)
                    one_facade_face.append(
                        out_start_index + point + 1 + len(self.layer.stack_layer_shape_xyz0[poly_idx]))
                    one_facade_face.append(out_start_index + 1 + len(self.layer.stack_layer_shape_xyz0[poly_idx]))
                    one_facade_face.append(out_start_index + 1)
                facade.append(one_facade_face)
            roof.extend(one_roof_face)

        return vertex, facade, roof
