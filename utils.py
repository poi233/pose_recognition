import argparse
import cv2
import numpy as np
import time
import logging
import os
import analyse
import matplotlib.pyplot as plt
import math
import analyse
import tensorflow as tf
import common
import json

from common import CocoPairsRender, CocoColors, preprocess, estimate_pose, draw_humans
from network_cmu import CmuNetwork
from network_mobilenet import MobilenetNetwork
from networks import get_network
from pose_dataset import CocoPoseLMDB

stage_level = 6
input_height = 368
input_width = 368
model = 'cmu'

# set base paths
video_base = './video'
result_base = './result'


def cv2_2_plt(img):
    b, g, r = cv2.split(img)
    img_show = cv2.merge([r, g, b])
    return img_show


def get_abs(num):
    if num is None:
        return None
    else:
        return abs(num)


def show_images(imgs):
    col = 3
    row = math.ceil(len(imgs) / float(col))
    plt.figure(figsize=(18, 4 * row))
    for index in range(0, len(imgs)):
        img = cv2_2_plt(imgs[index])
        plt.title(index)
        plt.subplot(row, col, index + 1)
        plt.imshow(img)


def draw_smooth_curves(pre_figures, oriented=None, smooth_rate=0.15):
    draw_curves(pre_figures, oriented=oriented)
    figures = {}
    for key, value in pre_figures.items():
        smoothed_data = smooth_data(value, smooth_rate)
        figures[key] = smoothed_data
    if oriented is None:
        index = 0
        row = math.ceil(len(figures) / 2.0)
        col = 2
        plt.figure(figsize=(18, 6 * row))
        for key, figure in figures.items():
            index = index + 1
            plt.subplot(row, col, index)
            plt.title(key)
            plt.plot(figure)
            plt.ylabel(key.split('_')[0], fontsize=12)
            plt.xlabel('frame_num', fontsize=12)
            plt.xlim((0, len(figure)))
            plt.xticks(np.arange(0, len(figure) + 1, math.ceil(len(figure) / 30.0)))
            plt.grid()
    elif oriented not in figures.keys():
        print('No key')
    else:
        plt.title(oriented)
        plt.plot(figures[oriented])
        plt.ylabel(oriented.split('_')[0], fontsize=12)
        plt.xlabel('frame_num', fontsize=12)
        plt.xlim((0, len(figures[oriented])))
        plt.xticks(np.arange(0, len(figures[oriented]) + 1, math.ceil(len(figures[oriented]) / 30.0)))
        plt.grid()


def smooth_data(data, alpha=0.1):
    sm_data = []
    start = 0
    all_none = True
    for i in data:
        if i is not None:
            all_none = False
    if all_none:
        return data
    while data[start] is None:
        sm_data.append(None)
        start = start + 1
    for i in range(0, start):
        sm_data[i] = data[start]
    for i in range(start, len(data)):
        if i == 0:
            if data[i] is None:
                sm_data.append(0)
            else:
                sm_data.append(data[i])
        elif data[i] is None:
            sm_data.append(sm_data[i - 1])
        else:
            to_append = alpha * data[i] + (1. - alpha) * sm_data[i - 1]
            sm_data.append(to_append)
    return sm_data


def process_thetas(human_thetas):
    figures = {}
    for i in range(0, len(human_thetas)):
        if human_thetas[i] == []:
            continue
        for key, value in human_thetas[i][0].items():
            if key not in figures:
                figures[key] = []
            else:
                figures[key].append(value)
    return figures


def get_images(video_name, start_time=None, end_time=None, interval=1):
    input_path = os.path.join(video_base, video_name)
    vc_left = cv2.VideoCapture(input_path)
    fps = vc_left.get(cv2.CAP_PROP_FPS)
    size = (int(vc_left.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vc_left.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    imgs = []
    success, img = vc_left.read()
    count = 1
    while success:
        if start_time is None and end_time is None:
            if count % interval == 0:
                imgs.append(img)
        elif count >= start_time * fps and count <= end_time * fps and count % interval == 0:
            imgs.append(img)
        elif count > end_time * fps:
            break
        success, img = vc_left.read()
        count = count + 1
    vc_left.release()
    return imgs


def get_average(list1, list2):
    assert len(list1) == len(list2)
    l = []
    for i in range(0, len(list1)):
        if list1[i] is None and list2[i] is None:
            l.append(None)
        elif list1[i] is None and list2[i] is not None:
            l.append(list2[i])
        elif list1[i] is not None and list2[i] is None:
            l.append(list1[i])
        else:
            l.append((list1[i] + list2[i]) / 2.0)
    return l


def get_dirivative(l):
    dirivative_list = []
    for i in range(1, len(l)):
        dirivative_list.append(l[i] - l[i - 1])
    return dirivative_list


def process_dirivative(smoothed_data, smooth_rate, threshold=0.1):
    global max
    dirivative_data = smooth_data(get_dirivative(smoothed_data), smooth_rate)
    under = []
    over = []
    # split
    for data in dirivative_data:
        if data is None:
            under.append(None)
            over.append(None)
        elif abs(data) < threshold:
            under.append(data)
            over.append(None)
        else:
            under.append(None)
            over.append(data)
    # 对数据进行进一步精化
    # 对细小噪声进行消除操作
    start = 0
    end = -1
    to_change = []
    for i in range(1, len(over)):
        # 获取一段开始点
        if over[i - 1] is None and over[i] is not None:
            start = i
        # 获取一段的结束点
        elif over[i - 1] is not None and over[i] is None:
            end = i - 1
            # 筛除一些噪音段
            # if end >= start and end - start < 30 and max(map(get_abs, over[start:end + 1])) < 0.5:
            if end >= start and end - start < 30:
                for n in range(start, end + 1):
                    to_change.append(n)
                end = -1
    for n in to_change:
        under[n] = over[n]
        over[n] = None
    # 将间隔很短的段落进行合并降低噪声干扰
    i = 0
    last_end = len(over)
    while i < len(over) - 1:
        if over[i] is not None and over[i + 1] is None:
            last_end = i + 1
            i = i + 1
        elif over[i] is None and over[i + 1] is not None:
            present_start = i
            if present_start >= last_end and present_start - last_end <= 15:
                for change in range(last_end, present_start + 1):
                    over[change] = under[change]
                    under[change] = None
            i = i + 1
        else:
            i = i + 1
    return under, over


def draw_curves_0(figures, oriented=None):
    if oriented is None:
        index = 0
        row = math.ceil(len(figures) / 2.0)
        col = 2
        plt.figure(figsize=(18, 6 * row))
        for key, figure in figures.items():
            index = index + 1
            plt.subplot(row, col, index)
            plt.title(key)
            plt.plot(figure)
            plt.ylabel(key.split('_')[0], fontsize=12)
            plt.xlabel('frame_num', fontsize=12)
            plt.xlim((0, len(figure)))
            plt.xticks(np.arange(0, len(figure) + 1, math.ceil(len(figure) / 30.0)))
            plt.grid()
    elif oriented not in figures.keys():
        print('No key')
    else:
        plt.title(oriented)
        plt.plot(figures[oriented])
        plt.ylabel(oriented.split('_')[0], fontsize=12)
        plt.xlabel('frame_num', fontsize=12)
        plt.xlim((0, len(figures[oriented])))
        plt.xticks(np.arange(0, len(figures[oriented]) + 1, math.ceil(len(figures[oriented]) / 30.0)))
        plt.grid()


def draw_curves(video_name, human_thetas, oriented=common.CalDegree, smooth_rate=0.15, threshold=0.1):
    # set parameters
    total = float(len(oriented) * len(human_thetas))
    col = 2
    row = math.ceil(total / col)
    # set save path
    result_path = os.path.join(result_base, '{}_result'.format(video_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    path = os.path.join(result_path, 'figures')
    if not os.path.exists(path):
        os.makedirs(path)
    # draw figure and save
    for human_id, data in human_thetas.items():
        figure_path = os.path.join(path, 'human_{}'.format(human_id))
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        for key, value in data.items():
            if key in oriented:
                # get data
                origin_data = value
                smoothed_data = smooth_data(value, smooth_rate)
                # draw origin and smoothed data
                plt.figure(figsize=(15, 5))
                plt.title('{}_{}'.format(human_id, key))
                plt.plot(origin_data, 'C0')
                plt.plot(smoothed_data, 'C1')
                plt.ylabel(key.split('_')[0], fontsize=12)
                plt.xlabel('frame_num', fontsize=12)
                plt.legend(['origin', 'smoothed', 'dirivative'], loc='upper right')
                plt.xlim((0, len(value)))
                plt.xticks(np.arange(0, len(value) + 1, math.ceil(len(value) / 30.0)))
                plt.grid()
                # plt.savefig(os.path.join(figure_path, '{}'.format(key)), dpi=300)
                plt.show()
                plt.clf()
                # get dirivative data
                under, over = process_dirivative(smoothed_data, smooth_rate, threshold=threshold)
                # draw dirivative
                plt.figure(figsize=(15, 5))
                plt.title('{}_{}_dirivative'.format(human_id, key))
                plt.plot(under, 'C0')
                plt.plot(over, 'C1')
                plt.ylabel('dirivative', fontsize=12)
                plt.xlabel('frame_num', fontsize=12)
                plt.legend(['under 0.1', 'over 0.1'], loc='upper right')
                plt.xlim((0, len(value)))
                plt.yticks(np.arange(-0.5, 0.5, 0.1))
                plt.xticks(np.arange(0, len(value) + 1, math.ceil(len(value) / 30.0)))
                plt.grid()
                # plt.savefig(os.path.join(figure_path, '{}'.format(key)), dpi=300)
                plt.show()
                plt.clf()


def get_action_time(over, key):
    time = []
    start = 0
    end = -1
    for i in range(0, len(over)):
        if i == 0:
            if over[i] is not None:
                start = i
                continue
            else:
                continue
        # 获取一段开始点
        if over[i - 1] is None and over[i] is not None:
            start = i
        # 获取一段的结束点
        elif over[i - 1] is not None and over[i] is None:
            end = i
            if end > start:
                if key == 'tug' and sum(over[start:end]) / (end - start) < 0:
                    continue
                else:
                    time.append((round((end - start) / 30.0, 3), start, end))
                start = 0
                end = -1

        if i == len(over) - 1 and start != 0:
            time.append((round((len(over) - start) / 30.0, 3), start, len(over)))
    max = 0
    max_id = -1
    for i in range(0, len(time)):
        if time[i][0] > max:
            max_id = i
    action_time = "{},{},{}".format(time[max_id][0], time[max_id][1], time[max_id][2])
    return action_time

# # set base paths
# video_base = './video'
# result_base = './result'
# result_dirs = os.listdir(result_base)
# result_dirs.remove('.DS_Store')
#
# # get json files to result_json
# result_json = {}
# for result_dir in result_dirs:
#     json_path = os.path.join(result_base, '{}/json'.format(result_dir))
#     json_file = os.listdir(json_path)[0]
#     with open(os.path.join(json_path, json_file), 'r') as json_file:
#         json_data = json.load(json_file)
#         for human_id, data in json_data.items():
#             avg_vertical_leg = get_average(data['theta_vertical_8_9'], data['theta_vertical_11_12'])
#             avg_leg = get_average(data['theta_1_8_9'], data['theta_1_11_12'])
#             avg_knee = get_average(data['theta_8_9_10'], data['theta_11_12_13'])
#             avg = get_average(get_average(avg_vertical_leg, avg_leg), avg_knee)
#             json_data[human_id]['avg_vertical_leg'] = avg_vertical_leg
#             json_data[human_id]['avg_leg'] = avg_leg
#             json_data[human_id]['avg_knee'] = avg_knee
#             json_data[human_id]['avg'] = avg
#         result_json[result_dir] = json_data
#
# video_name = "turnback_1"
# key = '{}_result'.format(video_name)
# oriented = [
#     #             'theta_1_8_9',
#     #             'theta_vertical_8_9',
#     #             'distance_8_9',
#     #             'theta_1_11_12',
#     #             'theta_vertical_11_12',
#     #             'distance_11_12',
#     #             'theta_8_9_10',
#     #             'theta_11_12_13',
#     #             'theta_vertical_1_8_11',
#     #             'theta_8_1_11',
#     #             'avg_vertical_leg',
#     #             'avg_leg',
#     #             'avg_knee',
#     'avg',
# ]
#
# draw_curves(video_name, result_json[key], oriented=oriented, smooth_rate=0.1, threshold=0.1)
