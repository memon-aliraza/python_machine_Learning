#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @filename data_extration.py
    @author Ali R. Memon
    @date 15.05.2018

    @brief extract images and measurement point list from ROSBAG file

    @copyright (c) Ibeo Automotive Systems GmbH, Hamburg, Germany
"""

import os
import sys
import cv2
import csv
import re
import time
import os.path
import rospy
import rosbag
import datetime
import numpy as np
import pandas as pd
from os import path
from datetime import date
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()


def main():
    
    
   
    folders = ['data/images', 'data/lidar', 'data/specs', 'data/odometry']

    for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

            else:

                print 'Directories already exists'
                sys.exit()
    data_ext()

def ntp_to_unix_stamp(stamp):
 
    secs = (stamp.seconds - 0x83AA7E80) * 1000
    nsecs = (int(stamp.fractions * 1.0e9 / np.uint64(1 << 32)) / 1e9) * 1000
    milliseconds = int(secs + nsecs)

    return milliseconds
    
def data_ext():

    image_topic_name = "/image0"
    measurment_point_list_topic_name = "/measurement_point_list"
   
    bag_files = os.listdir("/home/IBEO.AS/arm/Documents/ibeo_rosbag/20180618_FLIPFineCalibrationEdges/")
    
    for file in bag_files:
        inbag_name = file
        bag = rosbag.Bag("/home/IBEO.AS/arm/Documents/ibeo_rosbag/20180618_FLIPFineCalibrationEdges/"+inbag_name)

        print "Reading bag file: " , inbag_name
        
        for topic, msg, t in bag.read_messages():
            if(topic == image_topic_name):
                

                try:
                    imgframe = bridge.compressed_imgmsg_to_cv2(msg)
    
                except CvBridgeError, e:
                    print(e)

                else:
                    stamp = ntp_to_unix_stamp(msg.header.creation_system_timestamp)
                    cv2.imwrite('data/images/'+str(stamp)+'.jpeg', imgframe)

            else:
                if(topic == measurment_point_list_topic_name):
                    for k in msg.meas_point_list:
    
                        mpl_stamp = ntp_to_unix_stamp(msg.measurement_timestamp_begin)
    
                        with open('data/lidar/' + str(mpl_stamp) + '.csv', 'ab') as file:
                            writer = csv.writer(file)
                            #writer.writerow([np.float32(k.position.x), np.float32(k.position.y), np.float32(k.position.z), re.sub('[,"()]', '',str(k.measurement_property))])

                            #9.706824,1.9801139,-1.4684542,"(216.0,)"
                            writer.writerow([msg])

    bag.close()
    print ("Extraction finished!")

if __name__ == "__main__":
    main()
