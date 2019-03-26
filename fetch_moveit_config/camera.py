#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse, cv2, math, os, rospy, sys, threading, time
from pprint import pprint
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from cv_bridge import CvBridge, CvBridgeError

import tf
import tf2_ros
import tf2_geometry_msgs

class RGBD(object):

    def __init__(self):
        """Similar to the HSR version, but with Fetch topic names."""
        topic_name_c = 'head_camera/rgb/image_raw'
        topic_name_i = 'head_camera/rgb/camera_info'
        topic_name_d = 'head_camera/depth_registered/image_raw'
        topic_name_p = 'head_camera/depth_registered/points'

        self._bridge = CvBridge()
        self._input_color_image = None
        self._input_depth_image = None
        self._input_point_cloud2 = None
        self._info = None
        self.is_updated = False

        self._sub_color_image = rospy.Subscriber(topic_name_c, Image, self._color_image_cb)
        self._sub_depth_image = rospy.Subscriber(topic_name_d, Image, self._depth_image_cb)
        self._sub_point_cloud2 = rospy.Subscriber(topic_name_p, PointCloud2, self._point_cloud2_cb)
        self._sub_info = rospy.Subscriber(topic_name_i, CameraInfo, self._info_cb)

    def _point_cloud2_cb(self, data):
        try:
            # self._input_depth_image = self._bridge.imgmsg_to_cv2(
            #         data, desired_encoding="passthrough")
            self._input_point_cloud2 = data
            # print("get dep")
            # print(self._input_depth_image)
        except Exception as e:
            rospy.logerr(e)

    def _color_image_cb(self, data):
        try:
            # color = self._bridge.imgmsg_to_cv2(data, "bgr8")
            # b, g, r = cv2.split(color)
            # cv2_img = cv2.merge([r, g, b])
            # self._input_color_image = cv2_img
            self._input_color_image = self._bridge.imgmsg_to_cv2(data, "bgr16")
            self.color_time_stamped = data.header.stamp
            self.is_updated = True
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)


    def _depth_image_cb(self, data):
        try:
            # self._input_depth_image = self._bridge.imgmsg_to_cv2(
            #         data, desired_encoding="passthrough")
            self._input_depth_image = self._bridge.imgmsg_to_cv2(data,'32FC1')
            # print("get dep")
            # print(self._input_depth_image)
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def _info_cb(self,data):
        try:
            self._info = data
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def read_point_cloud(self):
        return self._input_point_cloud2

    def read_color_data(self):
        return self._input_color_image

    def read_depth_data(self):
        return self._input_depth_image

    def read_info_data(self):
        return self._info


if __name__ == '__main__':
    rospy.init_node('take_photo')
    rgb = RGBD()
    i=1
    while i <= 150:
        img = rgb.read_depth_data()
        if img is None:
            continue
        cv2.imwrite("images/"+str(i)+".png", img)

        print i
        i=i+1
