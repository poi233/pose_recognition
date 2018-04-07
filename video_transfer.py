import argparse
import cv2
import numpy as np
import time
import logging
import os
import analyse

import tensorflow as tf

from common import CocoPairsRender, CocoColors, preprocess, estimate_pose, draw_humans
from network_cmu import CmuNetwork
from network_mobilenet import MobilenetNetwork
from networks import get_network
from pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


def video_write(img, humans):
    # display
    canvas = draw_humans(img, humans)
    stand_degrees = analyse.cal_degrees(humans)
    # print(stand_degrees)

    # show text
    y0, dy = 50, 25
    i = 0
    for person in stand_degrees:
        text = "person:"
        y = y0 + dy * i
        cv2.putText(canvas, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        i = i + 1
        for degree_name, degree in person.items():
            text = "{} = {}°".format(degree_name, analyse.cos_2_degree(degree))
            y = y0 + dy * i
            cv2.putText(canvas, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            i = i + 1
    return canvas
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Realtime Webcam')
    parser.add_argument('--videopath', type=str, default="people_1.avi")
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    parser.add_argument('--show-process', type=bool, default=False, help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')

    # set base paths
    video_base = './video'
    result_base = './result'

    # set video name
    videos = os.listdir(video_base)
    # video_name = 'stand3.MOV'

    # video_path = os.path.join(video_base, video_name)
    with tf.Session() as sess:
        # read network
        logging.info('model loading')
        net, _, last_layer = get_network(args.model, input_node, sess)

        for video_name in videos:
            print('{} process start'.format(video_name))
            video_path = os.path.join(video_base, video_name)

            #read video
            videoCapture = cv2.VideoCapture(video_path)
            # ret_val, img = cam.read()

            # 获得码率及尺寸
            zoom_rate = 1
            # fps = videoCapture.get(cv2.CAP_PROP_FPS)
            fps = 1
            size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * zoom_rate),
                    int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT) * zoom_rate))
            # 指定写视频的格式, I420-avi, MJPG-mp4
            videoWriter = cv2.VideoWriter(os.path.join(result_base, video_name), cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

            # 读帧
            success, img = videoCapture.read()
            logging.info('video fps=%f' % (fps))
            logging.info('video image=%dx%d' % (img.shape[1], img.shape[0]))

            # 处理图片大小
            img = cv2.resize(img, None, img, zoom_rate, zoom_rate, interpolation=cv2.INTER_AREA)
            logging.info('resized video image=%dx%d' % (img.shape[1], img.shape[0]))

            # 对每一帧进行操作
            count = 0
            while success:
                if count % 30 == 0 and count != 0:
                    logging.info('video processing %d' % (count))
                #read
                # logging.info('video read+')
                #process
                # logging.info('video preprocess+')
                preprocessed = preprocess(img, args.input_width, args.input_height)
                # logging.info('video process+')
                pafMat, heatMat = sess.run(
                    [
                        net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                        net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
                    ], feed_dict={'image:0': [preprocessed]}
                )
                heatMat, pafMat = heatMat[0], pafMat[0]

                # logging.info('video postprocess+')
                t = time.time()
                humans = estimate_pose(heatMat, pafMat)
                #write
                # logging.info('video write+')
                to_write = video_write(img, humans)
                videoWriter.write(to_write)
                # cv2.waitKey(10)  # 延迟
                success, img = videoCapture.read()  #获取下一帧
                # 缩小图片
                if success:
                    img = cv2.resize(img, None, img, zoom_rate, zoom_rate, interpolation=cv2.INTER_AREA)

                count = count + 1
            logging.info('video {} finished+'.format(video_name))
            videoCapture.release()
