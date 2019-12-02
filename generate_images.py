#!/usr/bin/env python

import argparse
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, Optional

import matplotlib.animation as anim
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.se2 import SE2

from multiprocessing import Pool

_ZORDER = {"AGENT": 15, "AV": 15, "OTHERS": 15}

# def rotate_polygon_about_pt(pts, rotmat, center_pt):
#     """Rotate a polygon about a point with a given rotation matrix.
#     Args:
#         pts: Array of shape (N, 2) representing a polygon or point cloud
#         rotmat: Array of shape (2, 2) representing a rotation matrix
#         center_pt: Array of shape (2,) representing point about which we rotate the polygon
#     Returns:
#         rot_pts: Array of shape (N, 2) representing a ROTATED polygon or point cloud
#     """
#     pts -= center_pt
#     rot_pts = pts.dot(rotmat.T)
#     rot_pts += center_pt
#     return rot_pts


# def get_Rt(agent_traj):
#     def rotation_angle(x,y):
#         angle=np.arctan(abs(y/x))
#         direction= -1* np.sign(x*y)
#         return direction*angle

#     t = agent_traj[0]
#     trajectory_rotation = rotation_angle(agent_traj[19,0],agent_traj[19,1])
#     c, s = np.cos(trajectory_rotation), np.sin(trajectory_rotation)
#     R = np.array([[c,-s], [s, c]])

#     city_to_agent_se2 = SE2(rotation = R, translation = t)
#     return city_to_agent_se2


# def transform_traj(traj, R, t):
#     traj = traj - t
#     traj = np.moveaxis(traj, 1, 0)
#     traj = np.matmul(R, traj)
#     traj = np.moveaxis(traj, 1, 0)
#     return traj


def draw_lane_polygons(ax, lane_polygons, color = "y"):
    for i, polygon in enumerate(lane_polygons):
        ax.plot(polygon[:, 0], polygon[:, 1], color=color, alpha=0.4, zorder=1)


def interpolate_polyline(polyline, num_points):
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))


def viz_sequence(df, name):
    print ("executing sequence visualization")

    lane_centerlines = None
    show = True
    smoothen = False
    time_list = np.sort(np.unique(df["TIMESTAMP"].values))
    city_name = df["CITY_NAME"].values[0]

    x_min = min(df["X"]) - 20
    x_max = max(df["X"]) + 20
    y_min = min(df["Y"]) - 20
    y_max = max(df["Y"]) + 20

    if lane_centerlines is None:
        avm = ArgoverseMap()
        # seq_lane_bbox = avm.city_halluc_bbox_table[city_name]
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]
        full_drivable_area = avm.find_local_driveable_areas([x_min, x_max, y_min, y_max], city_name)
        local_lane_polygons = avm.find_local_lane_polygons([x_min, x_max, y_min, y_max], city_name)

    frames = df.groupby("TRACK_ID")

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    if lane_centerlines is None:
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        lane_centerlines = []
        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline

            if (np.min(lane_cl[:, 0]) < x_max and np.min(lane_cl[:, 1]) < y_max and np.max(lane_cl[:, 0]) > x_min and np.max(lane_cl[:, 1]) > y_min):
                lane_centerlines.append(lane_cl)

    for lane_cl in lane_centerlines:
        ax.plot(lane_cl[:, 0], lane_cl[:, 1], "--", color="grey", alpha=1, linewidth=1, zorder=0)

    draw_lane_polygons(ax, local_lane_polygons)
    draw_lane_polygons(ax, full_drivable_area, color="tab:pink")

    plt.xlabel("Map X")
    plt.ylabel("Map Y")

    color_dict = {"AGENT": "#d33e4c", "OTHERS": "#d33e4c", "AV": "#007672"}
    object_type_tracker: Dict[int, int] = defaultdict(int)
    
    timestamp_20 = None
    for group_name, group_data in frames:
        object_type = group_data["OBJECT_TYPE"].values[0]
        if object_type == "AGENT":
            timestamp_20 = group_data["TIMESTAMP"].values[19]

    for group_name, group_data in frames:
        object_type = group_data["OBJECT_TYPE"].values[0]
        if object_type == "AGENT":
            continue

        cor_x = group_data["X"].values
        cor_y = group_data["Y"].values

        is_timestamp = group_data["TIMESTAMP"].values < timestamp_20

        cor_x = cor_x[is_timestamp]
        cor_y = cor_y[is_timestamp]

        if cor_x.shape[0] == 0:
            continue

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]

        ax.plot(
            cor_x,
            cor_y,
            "-",
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            linewidth=1,
            zorder=_ZORDER[object_type],
        )

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        if object_type == "OTHERS":
            marker_type = "o"
            marker_size = 7
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 7

        ax.plot(
            final_x,
            final_y,
            marker_type,
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            markersize=marker_size,
            zorder=_ZORDER[object_type],
        )

        object_type_tracker[object_type] += 1

    red_star = mlines.Line2D([], [], color="red", marker="*", linestyle="None", markersize=7, label="Agent")
    green_circle = mlines.Line2D([], [], color="green", marker="o", linestyle="None", markersize=7, label="Others")
    black_triangle = mlines.Line2D([], [], color="black", marker="^", linestyle="None", markersize=7, label="AV")

    plt.axis("off")
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    # return ax


if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    afl = ArgoverseForecastingLoader(dataset_dir)
    
    all_files = []
    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        for name in files:
            actual_name = name.split('.')[0]
            df = afl.get(f"{dataset_dir}/{name}").seq_df
            name = f"{dataset_dir}/{actual_name}.png"
            viz_sequence(df, name)

    # print (all_files)
    # agents = 4

    # results = []
    # pool = Pool(agents)
    # with Pool(agents) as pool:
    #     results.append(pool.apply_async(viz_sequence, all_files))

    # for file in all_files:
    #     viz_sequence(file[0], file[1])

    # final_images = []
    # for result in results:
    #     final_images.append(result.get())

    # results = []
    # with Pool(processes=agents) as pool:
    #     results.apppool.starmap(viz_sequence, all_files)

    # seq_path = f"{dataset_dir}/2645.csv"
    # viz_sequence(afl.get(seq_path).seq_df, show=True)
    # seq_path = f"{dataset_dir}/3828.csv"
    # viz_sequence(afl.get(seq_path).seq_df, show=True)
