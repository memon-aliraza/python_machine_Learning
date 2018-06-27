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
import time
import os.path
import rospy
import rosbag
import datetime
import argparse
import numpy as np
import pandas as pd
from os import path
from datetime import date
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def dir_struct():
    """
    Create directory structure if not already exists
    """

    folders = ['data/images', 'data/lidar', 'data/specs', 'data/odometry']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        else:
            print 'Directories already exists'
            sys.exit()
            

def ntp_to_unix_stamp(stamp):
    """
        Convert NTP to unix timestamp in milliseconds
        """

    secs = (stamp.seconds - 0x83AA7E80) * 1000
    nsecs = (int(stamp.fractions * 1.0e9 / np.uint64(1 << 32)) / 1e9) * 1000
    milliseconds = int(secs + nsecs)

    return milliseconds


def data_ext(bagfile, image_topic, mpl_topic):
    """
    The method receives the bag file and associated topics. Extract and saves data within pre-defined directory structure.
    """

    dir_struct()
    bridge = CvBridge()
    bag = rosbag.Bag(bagfile)

    for topic, msg, t in bag.read_messages():
        if(topic == image_topic):

            try:
                #return 0
                imgframe = bridge.compressed_imgmsg_to_cv2(msg)

            except CvBridgeError, e:
                print(e)

            else:
                #return 0
                stamp = ntp_to_unix_stamp(msg.header.creation_system_timestamp)
                cv2.imwrite('data/images/'+str(stamp)+'.jpeg', imgframe)

        else:
            if(topic == mpl_topic):
                #return 0
                for k in msg.meas_point_list:

                    mpl_stamp = ntp_to_unix_stamp(msg.measurement_timestamp_begin)

                    with open('data/lidar/' + str(mpl_stamp) + '.csv', 'ab') as file:
                        writer = csv.writer(file)
                        writer.writerow([np.float32(k.position.x), np.float32(k.position.y), np.float32(k.position.z)])
                        writer.writerow([(k.position.z), (k.radial_distance), (k.measurement_property)])

    bag.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_file', '-i', type=str, help='Location of ROS Bag File')
    parser.add_argument('--image_topic', '-img', type=str, help='Image Topic Name')
    parser.add_argument('--meas_point_list', '-mpl', type=str, help='Measurement Point List Topic Name')
    # data_extraction.py --bag_file Documents/ibeo_rosbag/20180618_FLIPFineCalibrationEdges/2018-06-18-14-32-52.bag.processed.aligned.bag --image_topic /image0 --meas_point_list /measurement_point_list

#    parser.add_argument('bag_file', type=str, help='Location of ROS Bag File')
#    parser.add_argument('image_topic', type=str, help='Image Topic Name')
#    parser.add_argument('meas_point_list', type=str, help='Measurement Point List Topic Name')
#    data_extraction.py Documents/ibeo_rosbag/20180618_FLIPFineCalibrationEdges/2018-06-18-14-32-52.bag.processed.aligned.bag  /image0  /measurement_point_list


    args = parser.parse_args()
    data_ext(args.bag_file, args.image_topic, args.meas_point_list)
    
