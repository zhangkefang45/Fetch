import rospy
import os
import rospy
import rospkg
import subprocess
import roslaunch
from std_srvs.srv import Trigger, TriggerResponse

# import roslaunch
# import rospy
#
# rospy.init_node('en_Mapping', anonymous=True)
# uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
# roslaunch.configure_logging(uuid)
# launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/zhangkefang/catkin_ws/src/Fetch/Fetch/fetch_gazebo/launch/pickplace_playground.launch"])
# launch.start()
# rospy.loginfo("started")
import rospy
import subprocess
import signal
import random

x = random.uniform(-0.3, 0.1)
rospy.sleep(1)
y = random.uniform(-0.2, 0.2)
print(x,y)
child = subprocess.Popen(["roslaunch","fetch_gazebo","pickplace_playground.launch", "x:=%.2f"%(x), "y:=%.2f"%(y)])
child.wait() #You can use this line to block the parent process untill the child process finished.
print("parent process")
print(child.poll())

rospy.loginfo('The PID of child: %d', child.pid)
print ("The PID of child:", child.pid)
