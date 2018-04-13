import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import analyse
import tensorflow as tf
import utils
from PIL import Image
import common
import json
import face_detect
import sys

from common import preprocess, estimate_pose, draw_humans
from networks import get_network

# set parameters
stage_level = 6
input_height = 368
input_width = 368
model = 'cmu'
input_node = tf.placeholder(tf.float32, shape=(1, input_height, input_width, 3), name='image')
# set base paths
video_base = './video'
result_base = './result'
# get video names
videos = []
# read network
sess = tf.Session()


def save_json(video_name, dict_obj, obj_name):
    # set json save path
    result_path = os.path.join(result_base, '{}_result'.format(video_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    path = os.path.join(result_path, 'json')
    if not os.path.exists(path):
        os.makedirs(path)
    # save json file
    js_obj = json.dumps(dict_obj)
    file_object = open(os.path.join(path, '{}.json'.format(obj_name)), 'w')
    file_object.write(js_obj)
    file_object.close()


def save_images(video_name, predicted_img):
    result_path = os.path.join(result_base, '{}_result'.format(video_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    path = os.path.join(result_path, 'images')
    if not os.path.exists(path):
        os.makedirs(path)
    frame_count = 1
    for images in predicted_img:
        Image.fromarray(utils.cv2_2_plt(images)).save(os.path.join(path, '{}.jpg'.format(frame_count)))
        frame_count = frame_count + 1


def save_action_time(video_name, human_thetas, smooth_rate=0.1, threshold=0.1):
    # set json save path
    oriented = ['theta_8_1_11', 'avg']
    transfer_dict = {'avg': 'tug', 'theta_8_1_11': 'tt'}
    result_path = os.path.join(result_base, '{}_result'.format(video_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    path = os.path.join(result_path, 'json_time')
    if not os.path.exists(path):
        os.makedirs(path)
    # draw figure and save
    for human_id, data in human_thetas.items():
        time = dict()
        time['videoName'] = video_name
        for key, value in data.items():
            if key in oriented:
                time[transfer_dict[key]] = []
                # get smoothed data
                smoothed_data = utils.smooth_data(value, smooth_rate)
                # get derivate data
                under, over = utils.process_dirivative(smoothed_data, smooth_rate, threshold=threshold)
                # 找出over中连续的不为null的片段
                time[transfer_dict[key]] = utils.get_action_time(over, key)
        # save json file
        js_obj = json.dumps(time)
        file_object = open(os.path.join(path, 'human_thetas_action_time_{}.json'.format(human_id)), 'w')
        file_object.write(js_obj)
        file_object.close()


def save_figures(video_name, human_thetas, oriented=None, smooth_rate=0.1):
    if oriented is None:
        return
    else:
        oriented.append('avg')
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
                origin_data = value
                smoothed_data = utils.smooth_data(value, smooth_rate)
                plt.title('{}_{}'.format(human_id, key))
                plt.plot(origin_data, 'C0')
                plt.plot(smoothed_data, 'C1')
                plt.ylabel(key.split('_')[0], fontsize=12)
                plt.xlabel('frame_num', fontsize=12)
                plt.legend(['origin', 'smoothed'], loc='upper right')
                plt.xlim((0, len(value)))
                plt.xticks(np.arange(0, len(value) + 1, math.ceil(len(value) / 30.0)))
                plt.grid()
                plt.savefig(os.path.join(figure_path, '{}'.format(key)), dpi=300)
                plt.clf()
                cv2.destroyAllWindows()


def get_prediction(img):
    preprocessed = preprocess(img, input_width, input_height)
    pafMat, heatMat = sess.run(
        [net.get_output(name=last_layer.format(stage=stage_level, aux=1)),
         net.get_output(name=last_layer.format(stage=stage_level, aux=2))],
        feed_dict={'image:0': [preprocessed]}
    )
    heatMat, pafMat = heatMat[0], pafMat[0]
    humans = estimate_pose(heatMat, pafMat)
    return humans


def get_predicted_imgs_and_thetas(imgs, trace_person):
    predicted_img = []
    human_thetas = {}
    human_tmp = {}
    frame_count = 0
    trace_human_id = None
    count = 0
    for img in imgs:
        count += 1
        if count % 50 == 0:
            print("processed img:{}".format(count))
        humans = get_prediction(img)
        # 对预测出来的进行删减
        processed_humans = []
        for human in humans:
            # 颈部如果未找到直接舍去
            if 1 not in human.keys():
                continue
            # 如果颈部，鼻子，一只耳朵能够被找到，且存在六个以上点，则保留此数据
            elif 0 in human.keys() and 1 in human.keys() and (16 in human.keys() or 17 in human.keys()) and len(
                    human.keys()) > 6:
                processed_humans.append(human)
            elif 0 not in human.keys() and 2 not in human.keys() and 5 not in human.keys():
                continue
            elif 9 not in human.keys() and 10 not in human.keys() and 12 not in human.keys() and 13 not in human.keys():
                continue
            elif len(human) < 6:
                continue
            else:
                processed_humans.append(human)
        # 在原始数据上标出点并连接，保存
        canvas = draw_humans(img, processed_humans)
        predicted_img.append(canvas)
        former_human_count = len(human_tmp)
        present_human_count = len(processed_humans)
        # 预测出有人物的第一帧图直接加入空字典
        if human_tmp == {}:
            count = 0
            for human in processed_humans:
                # 进行初始化
                human_tmp[count] = []
                for i in range(0, frame_count):
                    human_tmp[count].append(None)
                human_tmp[count].append(human)
                # 计算角度
                human_thetas[count] = {}
                theta_dict = analyse.cal_degrees(human)
                # 对所有可能的角度数据进行初始化
                for key in common.CalDegree:
                    human_thetas[count][key] = []
                    # 为之前没有人物的帧均加入None
                    for i in range(0, frame_count):
                        human_thetas[count][key].append(None)
                    if key in theta_dict.keys():
                        human_thetas[count][key].append(theta_dict[key])
                    else:
                        human_thetas[count][key].append(None)
                count = count + 1
        # 不是第一帧对前一帧的内容进行分析
        else:
            # 使用处理过的预测数据在前一帧的基础上为每一个人添加内容
            for human_id, human_list in human_tmp.items():
                human = None
                # 寻找最后一个最近的非None的数据，如果最近10个数据均是None，则删除对应键
                for former_human in human_list[::-1]:
                    if former_human is not None:
                        human = former_human
                        break
                # 在能找到一个非None的前数据时
                if human is not None:
                    # 以找到的已存数据的最近一帧与新加入的进行对比添加人物至合适位置
                    neck_point = analyse.get_point(human, 1)
                    radius = -1
                    # 如果找到鼻子点，以颈部到鼻子为半径，颈部为原点，在前一帧如果找到点，则判断为同一个人，如果均存在则取距离较近的一个
                    if 0 in human.keys():
                        nose_point = analyse.get_point(human, 0)
                        radius = analyse.get_distance(nose_point, neck_point)
                    # 如果找不到鼻子，则以左肩或者右肩到颈部的举例作为半径
                    elif 5 in human.keys():
                        l_shoulder = analyse.get_point(human, 5)
                        radius = analyse.get_distance(neck_point, l_shoulder)
                    elif 2 in human.keys():
                        r_shoulder = analyse.get_point(human, 2)
                        radius = analyse.get_distance(neck_point, r_shoulder)
                    # 寻找并分析得到合适的人并添加
                    to_add = None
                    min_distance = 1.
                    for processed_human in processed_humans:
                        neck_distance = analyse.get_distance(analyse.get_point(processed_human, 1),
                                                             analyse.get_point(human, 1))
                        if former_human_count < present_human_count:
                            if neck_distance < min_distance and neck_distance < radius:
                                to_add = processed_human
                                min_distance = neck_distance
                        else:
                            if neck_distance < min_distance:
                                to_add = processed_human
                                min_distance = neck_distance
                    human_tmp[human_id].append(to_add)
                    if to_add is not None:
                        processed_humans.remove(to_add)
                    # 添加角度数据
                    theta_dict = analyse.cal_degrees(to_add)
                    for key in common.CalDegree:
                        if theta_dict is None:
                            human_thetas[human_id][key].append(None)
                        elif key in theta_dict.keys():
                            human_thetas[human_id][key].append(theta_dict[key])
                        else:
                            human_thetas[human_id][key].append(None)
                else:
                    human_tmp.pop(human_id)
                    human_thetas.pop(human_id)
                    continue
            # 如果有新的人物加入，在这里进行添加
            for residual_human in processed_humans:
                new_id = max(human_tmp.keys()) + 1
                # 添加人物
                human_tmp[new_id] = []
                for i in range(0, frame_count):
                    human_tmp[new_id].append(None)
                human_tmp[new_id].append(residual_human)
                # 添加计算的角度数据
                human_thetas[new_id] = {}
                theta_dict = analyse.cal_degrees(residual_human)
                for key in common.CalDegree:
                    human_thetas[new_id][key] = []
                    for i in range(0, frame_count):
                        human_thetas[new_id][key].append(None)
                    if key in theta_dict.keys():
                        human_thetas[new_id][key].append(theta_dict[key])
                    else:
                        human_thetas[new_id][key].append(None)
        frame_count = frame_count + 1
    # 寻找需要追踪的人
    human_ids = list(human_tmp)
    if trace_person is not None:
        for i in range(0, len(imgs)):
            for human_id in human_ids:
                head_point = analyse.get_point(human_tmp[human_id][i], 0)
                if head_point is not None:
                    result = face_detect.recognize_person(trace_person, imgs[i], head_point)
                    if result:
                        trace_human_id = human_id
                        return predicted_img, human_thetas, human_tmp, trace_human_id
                    else:
                        human_ids.remove(human_id)
    return predicted_img, human_thetas, human_tmp, trace_human_id


def process_video(video_name, oriented=common.CalDegree, trace_person=None, smooth_rate=0.1):
    print('start processing {}'.format(video_name))
    predicted_img, human_thetas, human, trace_person_id = get_predicted_imgs_and_thetas(
        utils.get_images('{}.MOV'.format(video_name)), trace_person)
    # 对human_thetas再次进行处理
    human_to_delete = []
    for human_id, human_theta in human_thetas.items():
        type_num = len(human_theta)
        valid_num = 0.0
        for type, data in human_theta.items():
            not_none_count = 0.0
            total = len(data)
            for i in data:
                if i is not None:
                    not_none_count = not_none_count + 1
            if not_none_count / total > 0.5:
                valid_num = valid_num + 1
        if valid_num / type_num < 0.5:
            human_to_delete.append(human_id)
    for d in human_to_delete:
        human_thetas.pop(d)
        human.pop(d)
    # add avg in human_thetas
    for human_id, data in human_thetas.items():
        avg_vertical_leg = utils.get_average(data['theta_vertical_8_9'], data['theta_vertical_11_12'])
        avg_leg = utils.get_average(data['theta_1_8_9'], data['theta_1_11_12'])
        avg_knee = utils.get_average(data['theta_8_9_10'], data['theta_11_12_13'])
        avg = utils.get_average(utils.get_average(avg_vertical_leg, avg_leg), avg_knee)
        human_thetas[human_id]['avg_vertical_leg'] = avg_vertical_leg
        human_thetas[human_id]['avg_leg'] = avg_leg
        human_thetas[human_id]['avg_knee'] = avg_knee
        human_thetas[human_id]['avg'] = avg
        # 将处理好的数据存储为json文件
    save_json(video_name, human_thetas, 'human_thetas')
    # 将处理好的动作时间存储为json文件
    save_action_time(video_name, human_thetas, smooth_rate=0.1, threshold=0.1)
    # 存储每一张进行处理过的图片
    save_images(video_name, predicted_img)
    # 绘制并存储相关曲线
    save_figures(video_name, human_thetas, oriented=oriented, smooth_rate=smooth_rate)
    print('end processing {}'.format(video_name))
    return trace_person_id


net, _, last_layer = get_network(model, input_node, sess)
for video in videos:
    print('---------------------------')
    video_name = video.split('.')[0]
    type = video_name.split('_')[0]
    # print(process_video(video_name, trace_person='images/face.JPG'))
    trace_person = process_video(video_name)
