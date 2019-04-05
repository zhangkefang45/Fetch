import math
from camera import RGBD
from DQN import DQN
from Env import Robot, CubesManager
import copy

MAX_EPISODES = 5000
MAX_EP_STEPS = 100
MEMORY_CAPACITY = 100

if __name__ == '__main__':

    robot = Robot()
    s_dim = robot.state_dim
    a_dim = robot.action_dim
    a_bound = robot.action_bound
    cubm = CubesManager()
    rl = DQN()

    for i in range(MAX_EPISODES):
        cubm.reset_cube(rand=True)
        Box_position = cubm.read_cube_pose("demo_cube")
        # Box_position[0] -= 0.2
        # Box_position[2] -= 0.1
        print "cube position:", Box_position
        robot.Box_position = copy.deepcopy(Box_position)
        robot.reset()
        now_position = robot.gripper.get_current_pose("gripper_link").pose.position
        now_dis = math.sqrt(math.pow(now_position.x - robot.Box_position[0], 2)
                            + math.pow(now_position.y - robot.Box_position[1], 2)
                            + math.pow(now_position.z - robot.Box_position[2], 2))
        robot.reward = -10 * now_dis
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
            rl.store_transition(s, a, r, s_)  # a.shape =(1, 3)
            # r = float32
            if rl.memory_counter > 100:
                rl.learn()
                print "---------{learning}----------"
            print "memory_counter:", rl.memory_counter
            if done or st > 5:
                break
            s = s_
        print("the step is {0}".format(st))
        print("the total reward is {0} , and the average is {1}".format(rw, rw / st))
        if i % 50 == 0:
            print("this is the {0} turns".format(i))
            print("the total reward is {0} , and the average is {1}".format(rw, rw / st))

# robot = Robot()
# robot.test()
# cum = CubesManager()
# cum.reset_robot()

