import argparse
import cv2
import numpy as np
import time
import logging
import os
import video_transfer
import math


# 计算出相关肢体之间的角度
# e.g. thetas = {degree0:deg0,degree1:deg0},{degree0:deg0,degree1:deg0}
def cal_degrees(human):
    thetas = {}

    if human is None:
        return None

    if 1 in human.keys() and 8 in human.keys() and 9 in human.keys():
        thetas['theta_1_8_9'] = cal_theta_1_8_9(human)
        thetas['theta_vertical_8_9'] = cal_theta_vertical_8_9(human)
        thetas['distance_8_9'] = cal_distance_8_9(human)
    else:
        thetas['theta_1_8_9'] = None
        thetas['theta_vertical_8_9'] = None
        thetas['distance_8_9'] = None

    if 1 in human.keys() and 11 in human.keys() and 12 in human.keys():
        thetas['theta_1_11_12'] = cal_theta_1_11_12(human)
        thetas['theta_vertical_11_12'] = cal_theta_vertical_11_12(human)
        thetas['distance_11_12'] = cal_distance_11_12(human)
    else:
        thetas['theta_1_11_12'] = None
        thetas['theta_vertical_11_12'] = None
        thetas['distance_11_12'] = None

    if 8 in human.keys() and 9 in human.keys() and 10 in human.keys():
        thetas['theta_8_9_10'] = cal_theta_8_9_10(human)
    else:
        thetas['theta_8_9_10'] = None

    if 11 in human.keys() and 12 in human.keys() and 13 in human.keys():
        thetas['theta_11_12_13'] = cal_theta_11_12_13(human)
    else:
        thetas['theta_11_12_13'] = None

    if 1 in human.keys() and 8 in human.keys() and 11 in human.keys():
        thetas['theta_vertical_1_8_11'] = cal_vertical_1_8_11(human)
        thetas['theta_8_1_11'] = cal_theta_8_1_11(human)
    else:
        thetas['theta_vertical_1_8_11'] = None
        thetas['theta_8_1_11'] = None

    if thetas == {}:
        return None

    return thetas


# 左臀部-颈部-右臀部角度
def cal_theta_8_1_11(human):
    return get_theta((0, 0), (-1, 0), get_point(human, 1), get_point(human, 8)) - \
           get_theta((0, 0), (-1, 0), get_point(human, 1), get_point(human, 11))


# 颈部-臀部-膝盖的角度(左)
def cal_theta_1_8_9(human):
    return get_theta(get_point(human, 8), get_point(human, 1), get_point(human, 8), get_point(human, 9))


# 颈部-臀部-膝盖的角度(右)
def cal_theta_1_11_12(human):
    return get_theta(get_point(human, 11), get_point(human, 1), get_point(human, 11), get_point(human, 12))


# 臀部-膝盖与竖直方向朝上的夹角(左)
def cal_theta_vertical_8_9(human):
    return get_theta((0, 0), (0, -1), get_point(human, 8), get_point(human, 9))


# 臀部-膝盖与竖直方向朝上的夹角(右)
def cal_theta_vertical_11_12(human):
    return get_theta((0, 0), (0, -1), get_point(human, 11), get_point(human, 12))


# 臀部-膝盖-脚踝的角度(左)
def cal_theta_8_9_10(human):
    return get_theta(get_point(human, 9), get_point(human, 8), get_point(human, 9), get_point(human, 10))


# 臀部-膝盖-脚踝的角度(右)
def cal_theta_11_12_13(human):
    return get_theta(get_point(human, 12), get_point(human, 11), get_point(human, 12), get_point(human, 13))


def cal_distance_8_9(human):
    return get_distance(get_point(human, 8), get_point(human, 9))


def cal_distance_11_12(human):
    return get_distance(get_point(human, 11), get_point(human, 12))


def cal_vertical_1_8_11(human):
    return get_theta((0, 0), (0, 1), get_point(human, 1), get_point(human, 8)) - \
           get_theta((0, 0), (0, 1), get_point(human, 1), get_point(human, 11))


# 将cos转为角度
def cos_2_degree(cos):
    if cos > 1. or cos < -1.:
        # print(cos)
        return None
    assert cos <= 1. and cos >= -1.
    return math.acos(cos) * 180. / math.pi


# 求两条肢体的角度第v1p1,v1p2为第一条肢体的两点，v2p1,v2p2为第二条肢体的两点
def get_theta(v1p1, v1p2, v2p1, v2p2):
    v1 = get_unit_vector(v1p1, v1p2)
    v2 = get_unit_vector(v2p1, v2p2)
    cos_theta = cal_dot_product(v1, v2)
    return cos_2_degree(cos_theta)


# 求两点之间的距离
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 得到两个点的向量，方向为1->2
def get_vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])


# 求出两个向量的点积
def cal_dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


# 获得点n的坐标
def get_point(human, n):
    if human is None:
        return None
    elif n not in human.keys():
        return None
    else:
        return human[n][1]


# 获取单位向量
def get_unit_vector(p1, p2):
    vector = get_vector(p1, p2)
    distance = get_distance(p1, p2)
    unit_vector = (vector[0] / distance, vector[1] / distance)
    return unit_vector
