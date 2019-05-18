from Env import Robot, CubesManager
import copy
import numpy as np
import os

MAX_PICTURE_NUM = 30000


def read(file_dir="/home/ljt/Desktop/data"):
    x = []
    y = []
    data = []
    number = 0
    for root, dirs, files in os.walk(file_dir):
        for i in files:
            # print "the file data:", np.load("/home/ljt/Desktop/data/"+i)
            x_y = i[1:-11].split(" ")
            x1 = float(x_y[0][:-1])
            y1 = float(x_y[1])
            data1 = np.load("/home/ljt/Desktop/data/"+i)
            x.append(x1)
            y.append(y1)
            number += 1
            data.append(data1)
            if number == 10000:
                break
            # print x1, y1, data1
    return x, y, data


def collect_data():
    # set env
    robot = Robot()
    cubusm = CubesManager()
    number = 0
    steps = []
    for i in range(MAX_PICTURE_NUM):

        cubusm.reset_cube(rand=True)
        Box_position = cubusm.read_cube_pose("demo_cube")
        print "cube position:", str(Box_position)
        joint, view = robot.get_state()
        print "camera image shape:", view.shape
        np.save("/home/ljt/Desktop/ws/src/fetch_moveit_config/data/"+str(Box_position), view)


if __name__ == '__main__':
    read("/home/ljt/Desktop/data")