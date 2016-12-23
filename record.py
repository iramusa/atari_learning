#!/usr/bin/env python3
"""
Records images from a game as well as moves made by the player.


"""

import universe  # register the universe environments
import gym
import random

from PIL import Image
import threading
import time

PREFIX = 'DevilAttack_snap_'
IM_BOX = (0, 40, 168, 208)
ACTIONS = ['ArrowLeft', 'ArrowRight', 'z', 'z', 'z', 'z', 'z']


class Agent(object):
    def __init__(self, env):
        self.env = env
        self.t = 0
        self.last_move = random.choice(ACTIONS)

    def step(self):
        while True:
            print(self.t)
            action_n = [self.gen_action()]
            observation_n, reward_n, done_n, info = self.env.step(action_n)
            self.env.render()

            if observation_n[0] is not None:
                # print(observation_n)
                im = observation_n[0]['vision']
                self.write_image(im)

            self.t += 1
            time.sleep(0.01)

    def gen_action(self):
        presses = []
        but = random.choice(ACTIONS)
        if self.last_move != but:
            presses.append(('KeyEvent', self.last_move, False))
            presses.append(('KeyEvent', but, True))
            self.last_move = but

        return presses

    def write_image(self, image):
        im = Image.fromarray(image)
        im = im.crop(IM_BOX).convert('LA')
        im.thumbnail((86, 86), Image.ANTIALIAS)
        im.save('images_bw/' + PREFIX + str(self.t) + '.png')


if __name__ == '__main__':

    env = gym.make('gym-core.DemonAttack-v0')
    ag = Agent(env)

    env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()

    ag.step()

    # env.render()
    # thr = threading.Thread(target=ag.step())
    # thr.setDaemon(True)
    # thr.start()

    # while True:
        # time.sleep(0.1)


