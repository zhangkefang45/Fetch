#!/usr/bin/env python
# -*- coding: utf-8 -*-
import actionlib
import math
import copy
import rospy, sys
import moveit_commander
import control_msgs.msg
import random

from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from Env import CubesManager
from moveit_python import (MoveGroupInterface,
                           PlanningSceneInterface,
                           PickPlaceInterface)
from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from moveit_python.geometry import rotate_pose_msg_by_euler_angles
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from control_msgs.msg import PointHeadAction, PointHeadGoal
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from moveit_msgs.msg import PlaceLocation, MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# 超参数
CLOSED_POS = 0.0  # The position for a fully-closed gripper (meters).
OPENED_POS = 0.10  # The position for a fully-open gripper (meters).
ACTION_SERVER = 'gripper_controller/gripper_action'


class PointHeadClient(object):

    def __init__(self):
        self.client = actionlib.SimpleActionClient("head_controller/point_head", PointHeadAction)
        rospy.loginfo("Waiting for head_controller...")
        self.client.wait_for_server()

    def look_at(self, x, y, z, frame, duration=1.0):
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(duration)
        self.client.send_goal(goal)
        self.client.wait_for_result()


class GraspingClient(object):

    def __init__(self):
        self.scene = PlanningSceneInterface("base_link")
        self.pickplace = PickPlaceInterface("arm", "gripper", verbose=True)
        self.move_group = MoveGroupInterface("arm", "base_link")

        find_topic = "basic_grasping_perception/find_objects"
        rospy.loginfo("Waiting for %s..." % find_topic)
        self.find_client = actionlib.SimpleActionClient(find_topic, FindGraspableObjectsAction)
        self.find_client.wait_for_server()

    def updateScene(self):
        # find objects
        goal = FindGraspableObjectsGoal()
        goal.plan_grasps = True
        self.find_client.send_goal(goal)
        self.find_client.wait_for_result(rospy.Duration(5.0))
        find_result = self.find_client.get_result()

        # remove previous objects
        for name in self.scene.getKnownCollisionObjects():
            self.scene.removeCollisionObject(name, False)
        for name in self.scene.getKnownAttachedObjects():
            self.scene.removeAttachedObject(name, False)
        self.scene.waitForSync()

        # insert objects to scene
        objects = list()
        idx = -1
        for obj in find_result.objects:
            idx += 1
            obj.object.name = "object%d"%idx
            self.scene.addSolidPrimitive(obj.object.name,
                                         obj.object.primitives[0],
                                         obj.object.primitive_poses[0],
                                         wait = False)
            if obj.object.primitive_poses[0].position.x < 0.85:
                objects.append([obj, obj.object.primitive_poses[0].position.z])

        for obj in find_result.support_surfaces:
            # extend surface to floor, and make wider since we have narrow field of view
            height = obj.primitive_poses[0].position.z
            obj.primitives[0].dimensions = [obj.primitives[0].dimensions[0],
                                            1.5,  # wider
                                            obj.primitives[0].dimensions[2] + height]
            obj.primitive_poses[0].position.z += -height/2.0

            # add to scene
            self.scene.addSolidPrimitive(obj.name,
                                         obj.primitives[0],
                                         obj.primitive_poses[0],
                                         wait = False)

        self.scene.waitForSync()

        # store for grasping
        #self.objects = find_result.objects
        self.surfaces = find_result.support_surfaces

        # store graspable objects by Z
        objects.sort(key=lambda object: object[1])
        objects.reverse()
        self.objects = [object[0] for object in objects]
        #for object in objects:
        #    print(object[0].object.name, object[1])
        #exit(-1)

    def getGraspableObject(self):
        graspable = None
        for obj in self.objects:
            # need grasps
            if len(obj.grasps) < 1:
                continue
            # check size
            if obj.object.primitives[0].dimensions[0] < 0.03 or \
               obj.object.primitives[0].dimensions[0] > 0.25 or \
               obj.object.primitives[0].dimensions[0] < 0.03 or \
               obj.object.primitives[0].dimensions[0] > 0.25 or \
               obj.object.primitives[0].dimensions[0] < 0.03 or \
               obj.object.primitives[0].dimensions[0] > 0.25:
                continue
            # has to be on table
            if obj.object.primitive_poses[0].position.z < 0.5:
                continue
            print(obj.object.primitive_poses[0], obj.object.primitives[0])
            return obj.object, obj.grasps
        # nothing detected
        return None, None

    def getSupportSurface(self, name):
        for surface in self.support_surfaces:
            if surface.name == name:
                return surface
        return None

    def getPlaceLocation(self):
        pass

    def pick(self, block, grasps):
        success, pick_result = self.pickplace.pick_with_retry(block.name,
                                                              grasps,
                                                              support_name=block.support_surface,
                                                              scene=self.scene)
        self.pick_result = pick_result
        return success

    def place(self, block, pose_stamped):
        places = list()
        l = PlaceLocation()
        l.place_pose.pose = pose_stamped.pose
        l.place_pose.header.frame_id = pose_stamped.header.frame_id

        # copy the posture, approach and retreat from the grasp used
        l.post_place_posture = self.pick_result.grasp.pre_grasp_posture
        l.pre_place_approach = self.pick_result.grasp.pre_grasp_approach
        l.post_place_retreat = self.pick_result.grasp.post_grasp_retreat
        places.append(copy.deepcopy(l))
        # create another several places, rotate each by 360/m degrees in yaw direction
        m = 16 # number of possible place poses
        pi = 3.141592653589
        for i in range(0, m-1):
            l.place_pose.pose = rotate_pose_msg_by_euler_angles(l.place_pose.pose, 0, 0, 2 * pi / m)
            places.append(copy.deepcopy(l))

        success, place_result = self.pickplace.place_with_retry(block.name,
                                                                places,
                                                                scene=self.scene)
        return success

    def tuck(self):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        pose = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(joints, pose, 0.02)
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return

    def stow(self):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        pose = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(joints, pose, 0.02)
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return

    def intermediate_stow(self):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        pose = [0.7, -0.3, 0.0, -0.3, 0.0, -0.57, 0.0]
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(joints, pose, 0.02)
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return


class MoveItIkDemo(object):
    MIN_EFFORT = 35  # Min grasp force, in Newtons
    MAX_EFFORT = 100  # Max grasp force, in Newtons

    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)
        # 初始化ROS节点
        rospy.init_node('moveit_demo')
        # 初始化需要使用move group控制的机械臂中的arm group
        arm = moveit_commander.MoveGroupCommander('arm')
        gripper = moveit_commander.MoveGroupCommander('gripper')
        self._client = actionlib.SimpleActionClient(ACTION_SERVER, control_msgs.msg.GripperCommandAction)
        self._client.wait_for_server(rospy.Duration(10))
        # 获取终端link的名称
        end_effector_link = arm.get_end_effector_link()
        # 获取场景中的物体
        head_action = PointHeadClient()
        grasping_client = GraspingClient()
        # 向下看
        head_action.look_at(1.2, 0.0, 0.0, "base_link")
        grasping_client.stow()
        self.open()
        grasping_client.updateScene()
        rospy.sleep(5)
        # 设置目标位置所使用的参考坐标系
        reference_frame = 'base_link'
        arm.set_pose_reference_frame(reference_frame)

        # 当运动规划失败后，允许重新规划
        arm.allow_replanning(True)

        # 设置位置(单位：米)和姿态（单位：弧度）的允许误差
        arm.set_goal_position_tolerance(0.01)
        arm.set_goal_orientation_tolerance(0.05)

        # # 设置机械臂工作空间中的目标位姿，位置使用x、y、z坐标描述，
        # # 姿态使用四元数描述，基于base_link坐标系
        target_pose = PoseStamped()
        target_pose.header.frame_id = reference_frame
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = 0.6
        target_pose.pose.position.y = 0.2
        target_pose.pose.position.z = 0.75
        target_pose.pose.orientation.x = 0
        target_pose.pose.orientation.y = 0
        target_pose.pose.orientation.z = 0
        target_pose.pose.orientation.w = 0

        # 设置机器臂当前的状态作为运动初始状态
        arm.set_start_state_to_current_state()

        # 设置机械臂终端运动的目标位姿
        arm.set_pose_target(target_pose, end_effector_link)

        # 规划运动路径
        traj = arm.plan()

        # 按照规划的运动路径控制机械臂运动
        arm.execute(traj)
        rospy.sleep(2)

        l = gripper.get_current_pose("l_gripper_finger_link").pose.position
        r = gripper.get_current_pose("r_gripper_finger_link").pose.position
        print(math.sqrt(pow(l.x-r.x, 2)+pow(l.y-r.y, 2)+pow(l.z-r.z,2)))
        self.close()
        rospy.sleep(2)
        l = gripper.get_current_pose("l_gripper_finger_link").pose.position
        r = gripper.get_current_pose("r_gripper_finger_link").pose.position
        print(math.sqrt(pow(l.x-r.x, 2)+pow(l.y-r.y, 2)+pow(l.z-r.z,2)))

        grasping_client.stow()
        #
        # # 控制机械臂终端向右移动5cm
        # # arm.shift_pose_target(1, -0.05, end_effector_link)
        # # arm.go()
        # # rospy.sleep(1)
        # #
        # # # 控制机械臂终端反向旋转90度
        # # arm.shift_pose_target(3, -1.57, end_effector_link)
        # # arm.go()
        # # rospy.sleep(1)
        # #
        # # # 控制机械臂回到初始化位置
        # # arm.set_named_target('home')
        # # arm.go()
        #
        # # 关闭并退出moveit
        # moveit_commander.roscpp_shutdown()
        # moveit_commander.os._exit(0)

    def open(self):
        """Opens the gripper.
        """
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = OPENED_POS
        self._client.send_goal_and_wait(goal, rospy.Duration(10))

    def close(self, width=0.0, max_effort=MAX_EFFORT):
        """Closes the gripper.

        Args:
            width: The target gripper width, in meters. (Might need to tune to
                make sure the gripper won't damage itself or whatever it's
                gripping.)
            max_effort: The maximum effort, in Newtons, to use. Note that this
                should not be less than 35N, or else the gripper may not close.
        """
        assert CLOSED_POS <= width <= OPENED_POS
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = width
        goal.command.max_effort = max_effort
        self._client.send_goal_and_wait(goal, rospy.Duration(10))



if __name__ == "__main__":
    test = CubesManager()
    i = 1
    while i<10:
        print(i)
        test.reset_cube(rand=True)
        rospy.sleep(1)
        i+=1


