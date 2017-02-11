#!/usr/bin/env python3
"""
Records images from a game as well as moves made by the player.
"""

import universe  # register the universe environments
import gym
import random

from PIL import Image
import numpy as np
import threading
import os
import time
import tensorflow as tf

# GAME = 'AirRaid'
# GAME = 'DevilAttack'
# GAME = 'SpaceInvaders'
# GAME = 'Pong'
GAME = 'Asteroids'
# GAME = 'Berzerk'
FILENAME = GAME + '-valid'
# FILENAME = GAME + '-train'
IM_BOX = (0, 0, 160, 192)
ACTIONS = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'z','z','z','z','z', 'n']
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
        self.last_move = random.randint(0, len(ACTIONS)-1)
        filename = os.path.join('new_images/', FILENAME + '.tfrecords')
        print('Writing', filename)
        self.writer = tf.python_io.TFRecordWriter(filename)

    def step(self):
        cnt = 0

        while cnt < 1000:
            time.sleep(0.01)

            print(cnt)
            action_n = [self.gen_action()]
            observation_n, reward_n, done_n, info = self.env.step(action_n)
            self.env.render()

            if observation_n[0] is not None:
                # print(observation_n)
                im = observation_n[0]['vision']
                rw = reward_n[0]
                # self.write_image(im)

                if np.max(im) < 1:
                    continue

                self.write_record(im, rw)
                self.t += 1
                cnt += 1


            # TODO check logic
            if done_n[0] is True:
                self.t = 0
                self.ep += 1


    def gen_action(self):
        presses = []
        draw = random.randint(0, len(ACTIONS)-1)
        # print(self.last_move)
        if self.last_move != draw:
            presses.append(('KeyEvent', ACTIONS[self.last_move], False))
            presses.append(('KeyEvent', ACTIONS[draw], True))
            self.last_move = draw

        return presses

    def write_record(self, image, reward):
        im = Image.fromarray(image)
        im = im.crop(IM_BOX).convert('L')
        im = im.resize((84, 84), Image.NEAREST)
        # im.thumbnail((80, 91), Image.ANTIALIAS)
        # im.thumbnail((84, 84), Image.ANTIALIAS)
        # im.thumbnail((28, 28), Image.ANTIALIAS)

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(84),
            'width': _int64_feature(84),
            'depth': _int64_feature(1),
            # todo add time delayed
            'timestep': _int64_feature(self.t),
            'episode': _int64_feature(self.ep),
            'reward': _float_feature(reward),
            'action': _int64_feature(self.last_move),
            'image_raw': _bytes_feature(im.tobytes())}))

        self.writer.write(example.SerializeToString())

    def write_image(self, image):
        im = Image.fromarray(image)
        # im = im.crop(IM_BOX).convert('LA')
        im.thumbnail((86, 86), Image.ANTIALIAS)
        im.save('images_bw/' + FILENAME + str(self.t) + '.png')

    def close(self):
        self.writer.close()


if __name__ == '__main__':

    env = gym.make('gym-core.' + GAME + '-v0')

    ag = Agent(env)

    env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()

    ag.step()
    ag.close()
    # env.render()
    # thr = threading.Thread(target=ag.step())
    # thr.setDaemon(True)
    # thr.start()

    # while True:
        # time.sleep(0.1)


