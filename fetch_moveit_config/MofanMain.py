import math
from camera import RGBD
from MofanDDPG import DDPG
from Env import Robot, CubesManager
import copy
import numpy as np

MAX_EPISODES = 900
MAX_EP_STEPS = 5
ON_TRAIN = True


if __name__ == '__main__':
    # set env
    robot = Robot()
    cubm = CubesManager()
    observation_dim = 3
    action_dim = 3
    action_bound = -1, 1

    # set RL method (continuous)
    rl = DDPG(action_dim, observation_dim, action_bound)
    number = 0
    steps = []
    # start training
    for i in range(MAX_EPISODES):

        cubm.reset_cube(rand=True)
        Box_position = cubm.read_cube_pose("demo_cube")
        print "cube position:", Box_position
        robot.Box_position = copy.deepcopy(Box_position)
        now_position = robot.gripper.get_current_pose("gripper_link").pose.position
        now_dis = math.sqrt(math.pow(now_position.x - robot.Box_position[0], 2)
                            + math.pow(now_position.y - robot.Box_position[1], 2)
                            + math.pow(now_position.z - robot.Box_position[2], 2))
        robot.reward = -10 * now_dis
        robot.reset()
        s = robot.get_state()
        ep_r = 0.  # reward of each epoch
        for j in range(MAX_EP_STEPS):

            a = rl.choose_action(np.array(s[0]))

            s_, r, done = robot.step(a)
            number += 1
            print "-------the %i step-------" % number
            rl.store_transition(np.array(s[0]), a, r, np.array(s_[0]))
            # print s_[0]
            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()
            # rl.learn()
            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
        rl.save()


def eval():
    rl.restore()
    robot.render()
    robot.viewer.set_vsync(True)
    s = robot.reset()
    while True:
        robot.render()
        a = rl.choose_action(s)
        s, r, done = robot.step(a)


# if ON_TRAIN:
#     train()
# else:
#     eval()



