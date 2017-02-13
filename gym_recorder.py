#!/usr/bin/env python3
"""
Records images from a game as well as moves made by the player.
"""

import universe  # register the universe environments
import gym
import random

from PIL import Image
from skimage import feature
import numpy as np
import threading
import os
import time
import tensorflow as tf


# GAME = 'AirRaid'
# GAME = 'DemonAttack'
# GAME = 'SpaceInvaders'
# GAME = 'Pong'
# GAME = 'Asteroids'
# GAME = 'Berzerk'
GAME = 'Freeway'
FILENAME = GAME + '-valid'
# FILENAME = GAME + '-train'
IM_BOX = (0, 0, 160, 192)
# ACTIONS = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'z','z','z','z','z', 'n']
# ACTIONS = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown']
# ACTIONS = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'n']
# ACTIONS = ['n']


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class Agent(object):
    def __init__(self, env):
        self.env = env
        self.ep = 0
        self.t = 0
        self.last_move = random.randint(0, 1)
        self.obs_list = []
        self.rw_list = []
        filename = os.path.join('full_images/', FILENAME + '.tfrecords')
        print('Writing', filename)
        self.writer = tf.python_io.TFRecordWriter(filename)

    def step(self):
        cnt = 0

        total_steps = 30000 if 'train' in FILENAME else 5000
        while cnt < total_steps:
            # time.sleep(0.1)

            print(cnt)
            # action = env.action_space.sample()
            # teleport button in asteroids
            if cnt % 10 == 0:
                action = np.random.randint(0, 3)
            # action_n = [self.gen_action()]
            observation, reward, done, info = self.env.step(action)

            if done:
                env.reset()
                self.t = 0
                self.ep += 1
                continue

            # self.env.render()
            if np.max(observation) < 1:
                continue

            self.obs_list.append(observation)
            self.rw_list.append(reward)

            # if enough frames to merge
            if len(self.obs_list) == 1:
                # print(observation_n)

                self.write_record()
                self.obs_list = []
                self.rw_list = []
                self.t += 1
                cnt += 1


    def gen_action(self):
        presses = []
        draw = random.randint(0, len(ACTIONS)-1)
        # print(self.last_move)
        if self.last_move != draw:
            presses.append(('KeyEvent', ACTIONS[self.last_move], False))
            presses.append(('KeyEvent', ACTIONS[draw], True))
            self.last_move = draw

        return presses

    def write_record_edges_merged(self):
        processed_ims = []
        for i in range(len(self.obs_list)):
            im = Image.fromarray(self.obs_list[i])
            im = im.convert('L')
            # im = im.resize((84, 84), Image.BICUBIC)
            im = im.resize((84, 84), Image.NEAREST)
            self.obs_list[i] = im
            im = np.array(im)
            im = feature.canny(im)
            processed_ims.append(im)

        merged = np.logical_or(*processed_ims).astype('uint8')
        # print(len(merged.tobytes()))

        # im = im.resize((84, 84), Image.NEAREST)
        # im.thumbnail((80, 91), Image.ANTIALIAS)
        # im.thumbnail((84, 84), Image.ANTIALIAS)
        # im.thumbnail((28, 28), Image.ANTIALIAS)

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(84),
            'width': _int64_feature(84),
            'depth': _int64_feature(1),
            'timestep': _int64_feature(self.t),
            'episode': _int64_feature(self.ep),
            'reward': _float_feature(np.sum(self.rw_list)),
            'action': _int64_feature(self.last_move),
            'image_processed': _bytes_feature(merged.tobytes()),
            'image_raw_0': _bytes_feature(self.obs_list[0].tobytes()),
            'image_raw_1': _bytes_feature(self.obs_list[1].tobytes())}))

        self.writer.write(example.SerializeToString())

    def write_record(self):
        im = self.obs_list[0]

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(84),
            'width': _int64_feature(84),
            'depth': _int64_feature(1),
            'timestep': _int64_feature(self.t),
            'episode': _int64_feature(self.ep),
            'reward': _float_feature(np.sum(self.rw_list)),
            'action': _int64_feature(self.last_move),
            'image_raw': _bytes_feature(im.tobytes()),
        }))

        self.writer.write(example.SerializeToString())

    def write_image(self, image):
        im = Image.fromarray(image)
        # im = im.crop(IM_BOX).convert('LA')
        im.thumbnail((86, 86), Image.ANTIALIAS)
        im.save('images_bw/' + FILENAME + str(self.t) + '.png')

    def close(self):
        self.writer.close()


if __name__ == '__main__':

    # env = gym.make('gym-core.' + GAME + '-v0')
    env = gym.make(GAME + "-v0")

    ag = Agent(env)

    # env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()

    ag.step()
    ag.close()
    # env.render()
    # thr = threading.Thread(target=ag.step())
    # thr.setDaemon(True)
    # thr.start()

    # while True:
        # time.sleep(0.1)


