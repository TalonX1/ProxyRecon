import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from descartes import PolygonPatch
import alphashape
import math
import open3d as o3d
from matplotlib import pyplot as plt


def project_pcd_to_zplane(pcd_xyz, z_height):
    pcd_xyz0 = np.array(pcd_xyz)
    pcd_xyz0[:, 2] = z_height
    return pcd_xyz0


def polygon_area(pcd_xyz0):
    pcd_xy = np.delete(pcd_xyz0, 2, 1)
    poly = Polygon(pcd_xy)
    return poly.area


def get_convexhull_xyz0(pcd_xyz0):
    try:
        pcd_xy = np.delete(pcd_xyz0, 2, 1)
        hull = ConvexHull(pcd_xy)
        convexhull_index = hull.vertices.tolist()
        convexhull_xyz0 = pcd_xyz0[convexhull_index]
        return np.array(convexhull_xyz0)
    except:
        # print("Can not generate convex hull")
        return None


def get_alpha_shape_xyz0(pcd_xyz0):
    one_layer_z = pcd_xyz0[0][2]
    pcd_xy = np.delete(pcd_xyz0, 2, 1)
    try:
        # one_floor_alpha_shape = alphashape.alphashape(pcd_xy, 0.05)
        one_floor_alpha_shape = alphashape.alphashape(pcd_xy, 0.06)
    except:
        # print("Can not generate alpha shape")
        return None
    # print(one_floor_alpha_shape.type)  # Polygon  MultiPolygon
    if one_floor_alpha_shape.type == 'Polygon':
        one_polygon_xyz = []
        one_floor_alpha_shape_xyz = np.insert(list(one_floor_alpha_shape.exterior.coords)[0:-1], 2, one_layer_z, 1)
        one_polygon_xyz.append(one_floor_alpha_shape_xyz)
        return 1, one_polygon_xyz
    elif one_floor_alpha_shape.type == 'MultiPolygon':
        multi_polygon = list(one_floor_alpha_shape.geoms)
        # print(len(multi_polygon))
        multi_polygon_xyz = []
        for i in range(len(multi_polygon)):
            # print(list(list(one_floor_alpha_shape.geoms)[i].exterior.coords))
            one_floor_alpha_shape_xyz = np.insert(list(list(one_floor_alpha_shape.geoms)[i].exterior.coords)[0:-1], 2,
                                                  one_layer_z, 1)
            multi_polygon_xyz.append(one_floor_alpha_shape_xyz)
        # multi_polygon_xyz = np.array(multi_polygon_xyz, dtype=object)
        return len(multi_polygon), np.array(multi_polygon_xyz)
    elif one_floor_alpha_shape.type == 'GeometryCollection':
        return None


def get_list_max_last_index(input_list):
    list_max = max(input_list)
    return [index for (index, value) in enumerate(input_list) if value == list_max][-1]


def show_point_cloud(xyz_points_set):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_points_set)
    o3d.visualization.draw_geometries([point_cloud])


def dbscan_o3d(pcd_xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=10, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 标签为-1的置为黑色
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])


# def on_one_line(pcd_xyz0):
#     delta_x = pcd_xyz0[1][0] - pcd_xyz0[0][0]
#     delta_y = pcd_xyz0[1][1] - pcd_xyz0[0][1]
#     distance_square = delta_x * delta_x + delta_y * delta_y
#     sin_times_cos = delta_x * delta_y/ distance_square
#
#     for j in range(2, len(pcd_xyz0)):
#         dx = pcd_xyz0[j][0] - pcd_xyz0[0][0]
#         dy = pcd_xyz0[j][1] - pcd_xyz0[0][1]
#         if math.fabs(dx * dy / (dx * dx + dy * dy) - sin_times_cos) > 10 ** -9:
#             return False
#
#     return True
#  https://www.tutorialspoint.com/program-to-check-whether-list-of-points-form-a-straight-line-or-not-in-python


def poly_condition(pcd_xyz0):
    pcd_num = len(pcd_xyz0)
    if pcd_num < 3:
        return False
    else:
        delta_x = pcd_xyz0[1][0] - pcd_xyz0[0][0]
        delta_y = pcd_xyz0[1][1] - pcd_xyz0[0][1]
        distance_square = delta_x * delta_x + delta_y * delta_y
        sin_times_cos = delta_x * delta_y / distance_square

        for j in range(2, len(pcd_xyz0)):
            dx = pcd_xyz0[j][0] - pcd_xyz0[0][0]
            dy = pcd_xyz0[j][1] - pcd_xyz0[0][1]
            if math.fabs(dx * dy / (dx * dx + dy * dy) - sin_times_cos) > 10 ** -9:
                return False

        return True


# def point_in_poly(p, poly):
#     one_poly = Polygon(poly)
#     one_point = Point(p)
#     point_in_poly = one_poly.contains(one_point)
#     #     print(one_point.within(one_poly))
#     return point_in_poly

def point_in_poly(p, poly):
    """

    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1, z1 = corner
        x2, y2, z2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def judge_one_list_in_other_list(source_list, target_list):
    count_num = 0
    for i in range(len(source_list)):
        for j in range(len(target_list)):
            if source_list[i] == target_list[j]:
                count_num += 1
    if count_num == len(source_list):
        return True

    else:
        return False


def poly_bounding_box(poly_points):
    x_min = np.min(poly_points, axis=0)[0]
    x_max = np.max(poly_points, axis=0)[0]
    y_min = np.min(poly_points, axis=0)[1]
    y_max = np.max(poly_points, axis=0)[1]

    bounding_box_center = [x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2]

    return bounding_box_center, x_min, x_max, y_min, y_max
