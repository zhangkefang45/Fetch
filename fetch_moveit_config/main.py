import cv2
from DQN import DQN
from Env import Robot, CubesManager

MAX_EPISODES = 5000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 50

if __name__ == "__main__":

    robot = Robot()
    s_dim = robot.state_dim
    a_dim = robot.action_dim
    a_bound = robot.action_bound
    cubm = CubesManager()
    rl = DQN()

    for i in range(MAX_EPISODES):
        robot.reset()
        cubm.reset_cube(rand=True)
        Box_position = cubm.read_cube_pose("cube1")
        Box_position[0] -= 0.2
        Box_position[2] -= 0.1
        robot.Box_position = Box_position
        # print(cubm.read_cube_pose("cube1"))
        # print(robot.Box_position)
        s = robot.get_state()
        st = 0
        rw = 0
        while True:
            st += 1
            a = rl.choose_action(s)
            s_, r, done = robot.step(a)
            rw += r
            r = -r
            rl.store_transition(s, a, r, s_)

            if rl.memory_counter > 50:
                rl.learn()

            if done:
                break
        print("the step is {0}".format(st))
        print("the total reward is {0} , and the average is {1}".format(rw, rw/st))
# robot = Robot()
# rl = DQN()
# s = robot.get_state()
# a = rl.choose_action(s)
# s_, r, done = robot.step(a)
# rl.store_transition(s, a, r, s_)
# rl.learn()