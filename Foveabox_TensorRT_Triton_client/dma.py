#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 15:01:41 2021

@author: qy
"""

import tarfile
import os
import sys
import re
import shutil
import random

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import numpy as np
import time
import cv2
import shutil
import subprocess
from datetime import datetime
from matplotlib.patches import BoxStyle
from tqdm import tqdm


def extract_tar_gz_file(tar_path, target_path):
    
    '''
    extract tar.gz file to the target path
    print operation info when finished
    
    @param[IN]: tar.gz file path
    @param[IN]: target path

    '''
    try:
        print("start extracting: ", tar_path)
        print("target path: ", target_path)
        tar = tarfile.open(tar_path, "r:gz")
        file_names = tar.getnames()
        for file_name in tqdm(file_names):
            tar.extract(file_name, target_path)
        tar.close()
        print("extract finish!")
    except:
        print("extract file failed:", tar_path)


def read_and_convert(file_path, target_file, img_format="jpg"):
    
    '''
    convert image type from yuyv422 to the input image_format
    
    @param[IN]: source image folder path, each image name should end 
                with ".yuyv422"
    @param[IN]: target folder path, converted image will be saved here
    @param[IN]: image type to convert, currentlly support tiff and jpeg, 
                default type is jpg
    '''
    
    HEIGHT = 720
    WIDTH = 1280
    sizes = 0
    amounts = 0
    
    path_lists = os.listdir(file_path)
    img_lists = []
    for file in path_lists:
        if file.endswith('.yuyv422'):
            img_lists.append(file) 
    n = len(img_lists)

    for i in range(n):
        timeStamp16_s = img_lists[i].split('_')[0]
        yuyv_file = os.path.join(file_path, img_lists[i])
        yuyv_data = np.fromfile(yuyv_file, dtype='u1')[0:HEIGHT*WIDTH*2].reshape(HEIGHT, WIDTH, -1)
        bgr_data = cv2.cvtColor(yuyv_data, cv2.COLOR_YUV2RGB_YVYU)
        if img_format == 'tiff':
            target_name = target_file+'/'+timeStamp16_s+'.tif'
        else:
            target_name = target_file+'/'+timeStamp16_s+'.jpg'
        cv2.imwrite(target_name, bgr_data)
        amounts += 1
        sizes += os.path.getsize(target_name)
    return


class DollyMonitor():
    
    def __init__(self, file_path, target_path="", work_path="", dm_path=""):
        
        self.FILE_PATH = file_path
        if target_path == "":
            file = file_path.split("/")    
            # target_path = "./" + "/".join(file[:-1])
            target_path = "/".join(file[:-1])
        self.TARGET_PATH = target_path
        self.WORK_PATH = work_path
        self.DM_PATH = dm_path
        self.CV_PERC_LOG = ""
        self.PLANNER_LOG = ""
        self.NAVI_TINY_LOG = ""
        self.SLAVE_PATH = ""
        self.KEY_INFO = {"CV_PERC_INFO" : -1,
                         "PLANNER_INFO" : -1,
                         "DETECT_INFO"  : -1,
                         "NAVI_TINY_INFO" : -1}
        
        return

    def data_prepare(self):
        
        '''
        data preparation, make sure the key info(log/image) ready
        step 1: - check log file ready. extract tar.gz if necessary
                - make dolly_monitor folder to store data
                - extract dump image file and convert to jpeg format
        step 2: - get candidate log (uos_cv_perception.log + uos_planner.log)
                
        
        '''
        # prepare data

        # step 1.1 extract tar file if necessary
        if not os.path.exists(self.FILE_PATH):
            print("file path not exist: ", self.FILE_PATH)
        if not os.path.exists(self.TARGET_PATH):
            print("file path not exist: ", self.TARGET_PATH)
        
        if self.FILE_PATH[-6:] == "tar.gz":
            print("start extracting logfile: ", self.FILE_PATH)
            if re.search("log_(.*)_(.*)_", self.FILE_PATH, re.M|re.I):
                log_name = re.search("(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", 
                                    self.FILE_PATH,re.M|re.I)# 2021-12-01-09-21-51
            else:
                log_name = re.search("log_(.*)_(.*)", self.FILE_PATH, re.M | re.I)
            log_time = log_name.group(1)
            log_path = os.path.join(self.TARGET_PATH, "log_{}".format(log_time))
            print(log_path)
            if not os.path.exists(log_path):
                extract_tar_gz_file(self.FILE_PATH, self.TARGET_PATH)
            else:
                print("log has been extracted!")
            self.WORK_PATH = log_path
        else:
            self.WORK_PATH = self.FILE_PATH
        print("work path: ", self.WORK_PATH)

        if not os.path.exists(self.WORK_PATH):
            print(self.WORK_PATH + " is not exist!")
            self.WORK_PATH = ""
            return 
        
        self.SLAVE_PATH = os.path.join(self.WORK_PATH, "slave_uos")
        # create folder for dolly monitor
        self.DM_PATH = os.path.join(self.WORK_PATH, "dolly_monitor")
        if not os.path.exists(self.DM_PATH):
            os.mkdir(self.DM_PATH)
        
        # step 1.2 extract dump image file if necessary
        
        DUMP_IMG_TAR_PATH = os.path.join(self.SLAVE_PATH, "dump_images.tar.gz")
        
        if not os.path.exists(DUMP_IMG_TAR_PATH):
            print(DUMP_IMG_TAR_PATH + " is not exist!")
            self.WORK_PATH = ""
            return

        IMG_PATH = os.path.join(self.SLAVE_PATH, "dump_images")
        if not os.path.exists(IMG_PATH) and os.path.exists(DUMP_IMG_TAR_PATH):
            extract_tar_gz_file(DUMP_IMG_TAR_PATH, self.SLAVE_PATH)
        
        def image_prepare(src_path, dst_path):
            if os.path.exists(src_path) and (not os.path.exists(dst_path)):
                cap_path_lists = os.listdir(src_path)
                if len(cap_path_lists) > 0:
                    test_file = cap_path_lists[0]
                else:
                    test_file = ""
                if test_file.endswith('.jpg') or test_file.endswith('.jpeg'):
#                    os.symlink(src_path, dst_path)
                    shutil.move(src_path, dst_path)
                else:
                     os.mkdir(dst_path)
                     read_and_convert(src_path, dst_path, 'jpeg')

        # step 1.3 get human dump image and convert to jpg format
        print("get human dump image and convert it to jpeg format")
        cap4_src_path = os.path.join(IMG_PATH, "image_capturer_4/human_det_images")
        cap5_src_path = os.path.join(IMG_PATH, "image_capturer_5/human_det_images")
        
        if not os.path.exists(cap4_src_path) and not os.path.exists(cap5_src_path):
            print(cap4_src_path +" and " + cap5_src_path + " is not exist!")
            self.WORK_PATH = ""
            return

        if os.path.exists(cap4_src_path):
            cap4_dst_path = os.path.join(self.DM_PATH, "human_dump_cap4")
            image_prepare(cap4_src_path, cap4_dst_path)
        else:
            print(cap4_src_path + " is not exist!")

        if os.path.exists(cap5_src_path):
            cap5_dst_path = os.path.join(self.DM_PATH, "human_dump_cap5")
            image_prepare(cap5_src_path, cap5_dst_path)
        else:
            print(cap5_src_path + " is not exist!")
        
        print("image process finished!")
        
        # step 2 prepare candidate log
        max_log_num = 20
        # step 2.1 uos_cv_preception log
        self.CV_PERC_LOG = os.path.join(self.DM_PATH, "uos_cv_perception.log")
        if not os.path.exists(self.CV_PERC_LOG):
            perception_log_candidates = []
    
            for i in range(max_log_num, 0, -1):
                candidate = os.path.join(self.SLAVE_PATH, "uos_cv_perception.{}.log".format(i))
                perception_log_candidates.append(candidate)
            perception_log_candidates.append(os.path.join(self.SLAVE_PATH, "uos_cv_perception.log"))
            f = open(self.CV_PERC_LOG, "w")
            for candi in perception_log_candidates:
                if os.path.exists(candi):
                    print("find perception log: ", candi)
                    for line in open(candi, errors = "ignore"):
                        f.writelines(line)
            f.close()
        else:
            print("find merged uos_cv_perecption.log already!")
            
        # step 2.2 uos_planner.log
        self.PLANNER_LOG = os.path.join(self.DM_PATH, "uos_planner.log")
        
        if not os.path.exists(self.PLANNER_LOG):
            planner_log_candidates = []
            for i in range(max_log_num, 0, -1):
                candidate = os.path.join(self.WORK_PATH, "uos_planner.{}.log".format(i))
                planner_log_candidates.append(candidate)
            planner_log_candidates.append(os.path.join(self.WORK_PATH, "uos_planner.log"))
            f = open(self.PLANNER_LOG, "w")
            for candi in planner_log_candidates:
                if os.path.exists(candi):
                    print("find planner log: ", candi)
                    for line in open(candi, errors = "ignore"):
                        f.writelines(line)
            f.close()
        else:
            print("find merged uos_planner.log already!")

        # step 2.3 uos_navi tiny log (only get navi_log_output_data info)
        # down sample the info
        self.NAVI_TINY_LOG = os.path.join(self.DM_PATH, "uos_navigation_tiny.log")
        
        if not os.path.exists(self.NAVI_TINY_LOG):
            navi_tiny_log_candidates = []
            for i in range(max_log_num, 0, -1):
                navi_orig = os.path.join(self.WORK_PATH, "uos_navigation.{}.log".format(i))
                navi_tiny = os.path.join(self.WORK_PATH, "uos_navi_tiny.{}.log".format(i))
                if os.path.exists(navi_orig):
                    cmd = 'cat ' + navi_orig + ' | grep "output_data" > ' + navi_tiny
                    p_ParseNavi = subprocess.Popen(cmd, shell=True)
                    p_ParseNavi.wait()
                    navi_tiny_log_candidates.append(navi_tiny)
            navi_orig = os.path.join(self.WORK_PATH, "uos_navigation.log")
            navi_tiny = os.path.join(self.WORK_PATH, "uos_navi_tiny.log")
            cmd = 'cat {} | grep "output data" > {}'.format(navi_orig, navi_tiny)
            p_ParseNavi = subprocess.Popen(cmd, shell=True)
            p_ParseNavi.wait()
            navi_tiny_log_candidates.append(navi_tiny)
            print("navi tiny: ", navi_tiny_log_candidates)
            f = open(self.NAVI_TINY_LOG, "w")
            cnt = 0
            sample_rate = 50
            for candi in navi_tiny_log_candidates:
                if os.path.exists(candi):
                    print("find navi_tiny log: ", candi)
                    for line in open(candi, errors = "ignore"):
                        if cnt % sample_rate == 0:
                            f.writelines(line)
                        cnt += 1
            f.close()            
        else:
            print("find merged uos_navigation_tiny.log already!")

    def get_key_info(self):
        
        '''
        extract key info from log
        for uos_planner.log: ts_struct, timestamp, action
        for uos_cv_perception.log: ts_struct, timestamp, class_id, track_id
        '''
        # step 2 get key info
        
        print("start getting key info")
        # step 2.1 key info for planner log
        print("get info from planner log")
        planner_log = open(self.PLANNER_LOG)
        line = planner_log.readline()
        planner_info = []
        while line:
            match_obj = re.search("(.*) Human (.*) error zone.", line, re.M | re.I)
            if match_obj is not None:
                ts_struct = match_obj.group(1).split(" INFO ")[0]
                ts_str = time.mktime(time.strptime(ts_struct.split(".")[0], "%y-%m-%d %H:%M:%S"))
                ts_str = str(ts_str)[:-1] + ts_struct.split(".")[1]
                action = match_obj.group(2)
                cur_info = [ts_struct, ts_str, action]
                planner_info.append(cur_info)
                # print(line)
            line = planner_log.readline()
        planner_log.close()
        print("Done!")
        self.KEY_INFO["PLANNER_INFO"] = planner_info
        
        # step 2.2 key info for cv_perception log
        print("get info from perception log")
        perc_log = open(self.CV_PERC_LOG)
        line = perc_log.readline()
        perc_info = []
        pre_line = ""
        while line:
            match_obj = re.search("Iou_tracker_sequence: (.*)", line, re.M|re.I)
            if match_obj is not None:
                try:
                    # parse the log format
                    if re.match("(.*) Iou_tracker_sequence: (.*)", line):
                        ts_struct = line.split(" INFO")[0]
                    else:
                        ts_struct = pre_line.split(" INFO")[0]
                    ts_str = time.mktime(time.strptime(ts_struct.split(".")[0], "%y-%m-%d %H:%M:%S"))
                    ts_str = str(ts_str)[:-1] + ts_struct.split(".")[1]
                    match_info = match_obj.group(1).split(" ")
                    class_id = match_info[8]
                    track_id = match_info[9]
                    cur_info = [ts_struct, ts_str, class_id, track_id]
                    perc_info.append(cur_info)
                    # print("ts: ", ts_struct)
                    # print("line", line) 
                    # print("class id, track id: ", class_id, track_id)
                except:
                    print("process data failed")
            pre_line = line
            line = perc_log.readline()
        perc_log.close()
        self.KEY_INFO["CV_PERC_INFO"] = perc_info
        print("Done!")
        
        return 
    
    def human_invade_analysis(self):
        
        '''
        human invade analysis
        - find out how many times did human detect happen during error zone
          WARNING!! if a miss detect and a correct detect happen at same time, 
          this algorithm can not figure it out
        - plot the result
        '''
        print("human invade analysis...")
        # analysis
        planner_info = self.KEY_INFO["PLANNER_INFO"]
        perc_info = self.KEY_INFO["CV_PERC_INFO"]
        if planner_info == -1:
            print("planner info init failed! please check")
            return
        if perc_info == -1:
            print("uos_cv_perc info init failed! please check")
            return

        if planner_info == []:
            print("No human detected in error zone")
            return
        elif perc_info == []:
            print("Can not find object detected info")
            return

        df_planner = pd.DataFrame(planner_info)
        df_planner.columns = ["Time", "Timestamp", "Action"]
        df_planner["Timestamp"] = df_planner["Timestamp"].astype("float")

        df_perc = pd.DataFrame(perc_info)
        df_perc.columns = ["Time", "Timestamp", "Class", "Track_id"]
        df_perc["Timestamp"] = df_perc["Timestamp"].astype("float")
        df_perc["Class"] = df_perc["Class"].astype("int")
        df_perc["Track_id"] = df_perc["Track_id"].astype("int")
                     
        # draw info in human detect err area
        planner_ts = np.array(df_planner["Timestamp"])
        # planner_action = np.array(df_planner["Action"])
        # reshape planner_ts (planner_ts: [enter time, leave time, enter time, leave time, ...], enter time and leave time come in pairs)
        error_zone = np.reshape(planner_ts[:int(len(planner_ts)//2*2)], (-1, 2))   # [:, 0]: enter time, [:, 1]: leave time

        if len(error_zone) == 0:
            print("Did not find human enter error zone!")
            return

        df_perc["human_in_error_zone"] = 0.0
        k = 0  # current index for error_zone
        cur_start = error_zone[k][0]  # errorzone startpoint index
        cur_end = error_zone[k][1]  # errorzone endpoint index
        error_count = 0  # error happen times
        cur_perc_info_start = -1  # perc_info start index belong to the same error_zone[k]
        errorzone_record_list = []  # [:, 0]:errorzone startpoint index, [:, 1]:errorzone endpoint index, [:, 2]:error_count
        time_delta = 1
        i = 0
        pbar = tqdm(total=df_perc.shape[0])
        while i < df_perc.shape[0]:
            if cur_start - time_delta <= df_perc["Timestamp"][i] <= cur_end + time_delta:
                if df_perc['Class'][i] == 5:  # Human
                    if cur_perc_info_start == -1:
                        cur_perc_info_start = i
                    error_count += 1
                i += 1
            elif df_perc["Timestamp"][i] > cur_end:
                if cur_perc_info_start != -1:
                    df_perc.loc[cur_perc_info_start:i, "human_in_error_zone"] = error_count
                    errorzone_record_list.append([cur_perc_info_start, i-1, error_count])  # i-1: perc_info end index belong to the same error_zone[k]
                    cur_perc_info_start = -1
                    error_count = 0
                k += 1
                if k >= len(error_zone):
                    break
                cur_start = error_zone[k][0]
                cur_end = error_zone[k][1]
            else:
                i += 1
            pbar.update(1)
        pbar.close()
                
        start_time = df_perc["Time"][0]
        end_time = df_perc["Time"][df_perc.shape[0]-1]

        # data_plot_bar
        x_gap = []
        x_index = []
        for j in range(0, df_perc.shape[0], df_perc.shape[0]//40):
            time_string = (time.asctime(time.localtime(df_perc["Timestamp"][j]))).split(' ')[4]
            x_gap.append(time_string)
            x_index.append(j)
        head_title = "start_time: "+start_time+"   end_time: "+end_time+"\n detect time zone number: "+str(len(error_zone))
        plt.title(head_title, fontdict={'size': 10})
        plt.xlabel("Timestamp", fontdict={'size': 8})
        plt.ylabel("human_number_in_error_zone", fontdict={'size': 8})
        plt.plot(range(df_perc.shape[0]), df_perc["human_in_error_zone"], linewidth=0.01)
        #plt.xticks(range(df_perc.shape[0]),())#It takes a lot of time 
        plt.xticks([])
        plt.xticks(x_index, x_gap, rotation=45)
        plt.xticks(size=3)
        plt.yticks(size=3)
        color = ['blue', 'green', 'red']
        for i in range(len(errorzone_record_list)):
            x_list = range(df_perc.shape[0])[errorzone_record_list[i][0]:errorzone_record_list[i][1]]
            plt.fill_between(x_list, 0, errorzone_record_list[i][2], facecolor=color[i % 3], alpha=1)
        zero_list = [0]*df_perc.shape[0]
        plt.plot(range(df_perc.shape[0]), zero_list, '-b', alpha=0.5)
        plt.savefig(os.path.join(self.DM_PATH, "Distribution_of_hunman-number_in_invasion-time.jpg"),
                    format="jpg", dpi=1200, bbox_inches='tight')
        #plt.show()
        plt.clf()
        plt.close()

        return
    
    def get_human_invade_time_zone(self):
        '''
        find out the time zone when planner report human enter error zone
        '''
        planner_info = self.KEY_INFO["PLANNER_INFO"]
        if planner_info == -1:
            print("planner info init failed! please check")
            return
        #determine the whether human enter the error zone
        if len(planner_info) == 0:
            print('\n')
            print("No human enter error zone, exit the program!")
            sys.exit()
        planner_ts = np.array(planner_info)[:, 1]
        # reshape planner_ts (planner_ts: [enter time, leave time, enter time, leave time, ...], enter time and leave time come in pairs)
        error_zone = np.reshape(planner_ts[:int(len(planner_ts) // 2 * 2)], (-1, 2)).astype(np.float)  # [:, 0]: enter time, [:, 1]: leave time
        error_zone = error_zone.tolist()
        return error_zone

    def get_detect_info(self):
        '''
        get key detect information.
        the track result has following info:
        ---------------------------------------
        stream_id, timestamp, top_left_x, top_left_y, bot_right_x, 
        bot_right_y,pos_x, pos_y, vel_x, vel_y, obj_class, track_id
        ---------------------------------------
            
        '''
        
        print("getting detection info")
        if self.CV_PERC_LOG == "":
            print("can not find perception log in dolly monitor dir")
            return
        perc_log = open(self.CV_PERC_LOG)
        line = perc_log.readline()
        detect_info = []
        cur_info = []
        track_info = []
        track_flag = False
        ts_flag = False
        stream_flag = False
        old_stream_id = -1
        old_ts = -1
        stream_id = -1
        ts = -1
        while line:
            match_obj = re.search("Iou_tracker_sequence: (.*)", line, re.M|re.I)
            if match_obj is not None:
                try:
                    match_info = match_obj.group(1).split(" ")
                    top_left_x = float(match_info[0])
                    top_left_y = float(match_info[1])
                    bot_right_x = float(match_info[2]) 
                    bot_right_y = float(match_info[3])
                    pos_x = float(match_info[4])
                    pos_y = float(match_info[5])
                    vel_x = float(match_info[6])
                    vel_y = float(match_info[7])
                    obj_class = int(match_info[8])
                    track_id = int(match_info[9])
                    track_flag = True
                    track_info.append([top_left_x, top_left_y, bot_right_x, bot_right_y,
                                       pos_x, pos_y, vel_x, vel_y, obj_class, track_id])
                except:
                    print("process data failed")
            ts_info = re.search("image_timestamp: (.*)", line, re.M | re.I)
            if ts_info is not None:
                # parse the log format
                ts_info = ts_info.group(1).split("_")[0]
                try: 
                    old_ts = ts
                    ts = float(ts_info)
                    ts_flag = True
                except:
                    print("process image ts failed")
            stream_info = re.search("_stream_id: (.*)", line, re.M | re.I)
            if stream_info is not None:
                try:
                    old_stream_id = stream_id
                    stream_id = int(stream_info.group(1))
                    stream_flag = True
                except:
                    print("process stream id failed")
                    
            if ts_flag and stream_flag:
                if track_flag:
                    for i in range(len(track_info)):
                        cur_info = [old_stream_id, old_ts] + track_info[i]
                        detect_info.append(cur_info)
                track_flag = False
                ts_flag = False
                stream_flag = False
                track_info = []
                    
            line = perc_log.readline()
        perc_log.close()
        self.KEY_INFO["DETECT_INFO"] = detect_info
        
        return

    def get_log_imglists(self):

        '''
        get lists of detection info of dollymonitor model
        the result contains:
            1. images that contains human
            2. human of detection result
        '''

        cap4_src_path = os.path.join(self.DM_PATH, "human_dump_cap4")
        cap5_src_path = os.path.join(self.DM_PATH, "human_dump_cap5")
        dst4_path = os.path.join(self.DM_PATH, "detect4_res")
        dst5_path = os.path.join(self.DM_PATH, "detect5_res")
        if not os.path.exists(dst4_path):
            os.mkdir(dst4_path)
        if not os.path.exists(dst5_path):
            os.mkdir(dst5_path)
        if self.KEY_INFO["DETECT_INFO"] == -1:
            print("can not find detection result")
            return
        
        detect_info = self.KEY_INFO["DETECT_INFO"]
        img4_name = os.listdir(cap4_src_path)
        img4_list = []  # format[[ts, path, im]]
        for i in range(len(img4_name)):
            if img4_name[i][-4:] == ".jpg":
                img4_list.append([float(img4_name[i][:-4].split("_")[0]), cap4_src_path + "/" + img4_name[i]])

        img5_name = os.listdir(cap5_src_path)
        img5_list = []
        for i in range(len(img5_name)):
            if img5_name[i][-4:] == ".jpg":
                img5_list.append([float(img5_name[i][:-4].split("_")[0]), cap5_src_path + "/" + img5_name[i]])
        
        # sort
        detect_info.sort(key=lambda x: x[1])
        if len(img4_list) != 0:
            img4_list.sort(key=lambda x: x[0])
        if len(img5_list) != 0:
            img5_list.sort(key=lambda x: x[0])
        # matching method

        def find_min_max_detect_time(ts, left, right):
            # return the index
            
            if left > right:
                print("input left and right error!")
                return -1
            if left == right:
                return left
            m = (left + right) // 2
            # print(left, right, m)
            if detect_info[m][1] <= ts:
                return find_min_max_detect_time(ts, m+1, right)
            else:
                return find_min_max_detect_time(ts, left, m)
        
        def get_candidate(img_list, stream):
            print('match images and detect_info for stream {}'.format(stream))
            length = len(detect_info)
            for i in tqdm(range(len(img_list))):
                # print(i)
                candidate = img_list[i][0]
                if candidate > detect_info[-1][1]:
                    img_list[i].append([])
                    continue
                ts_min_max_idx = find_min_max_detect_time(candidate, 0, len(detect_info)-1)
                
                l = ts_min_max_idx-1
                r = ts_min_max_idx

                while r < length and (detect_info[r][1] - candidate) <= 1:
                    if detect_info[r][0] == stream:
                        break
                    else:
                        r += 1
        
                while l >= 0 and (candidate - detect_info[l][1]) <= 1:
                    if detect_info[l][0] == stream:
                        break
                    else:
                        l -= 1
            
                # print((detect_info[r][1] > candidate) , (candidate > detect_info[l][1]))
                l_ts_diff = candidate - detect_info[l][1]  # left side
                r_ts_diff = detect_info[r][1] - candidate  # right side
#                if 1 < min(l_ts_diff, r_ts_diff):
                if (l >= 0 and 0.2 < min(l_ts_diff, r_ts_diff)) or (l < 0 and r_ts_diff > 0.2):
                    img_list[i].append([])
                    # ts diff should be in +-200ms
                    # print("failed to find best error case", l_ts_diff, r_ts_diff)
                    continue
                if l >= 0 and l_ts_diff < r_ts_diff:
                    #pick small side
                    # print("pick small side", l_ts_diff)
                    cur_ts = detect_info[l][1]
                    cur_info = []
                    if detect_info[l][0] != stream:
                        img_list[i].append([])
                        continue
                    while l >= 0 and detect_info[l][1] == cur_ts and detect_info[l][0] == stream:
                        cur_info.append(detect_info[l])
                        l -= 1
                    img_list[i].append(cur_info)
                else:
                    #pick large side
                    # print("pick large side", r_ts_diff)
                    cur_ts = detect_info[r][1]
                    cur_info = []
                    if detect_info[r][0] != stream:
                        img_list[i].append([])
                        continue
                    while r < length and detect_info[r][1] == cur_ts and detect_info[r][0] == stream:
                        cur_info.append(detect_info[r])
                        r += 1
                    img_list[i].append(cur_info)
            return img_list
        img4_list = get_candidate(img4_list, 4)
        img5_list = get_candidate(img5_list, 5)

        #draw ans save
        img_lists = [img4_list, img5_list]
        # print(img_lists)
        dst_lists = [dst4_path, dst5_path]
        return img_lists,dst_lists

    def plot_detect_res(self):
        
        '''
        plot detection result
        the result contains:
            1. image + detect bbox
            2. IPM result
            3. timestamp and localtime
        the timestamp of the image and the timestamp of the perceotion log
        '''
        
        print("plot detection result, this step could spend some time")
        img_lists,dst_lists = self.get_log_imglists()
        invade_zone = self.get_human_invade_time_zone()
        print('match images and invade_zone')
        for k in range(len(img_lists)):
            img_list = img_lists[k]
            dst_path = dst_lists[k]
            for i in tqdm(range(len(img_list))):
                info = img_list[i]
                #skip empty info
                ts = info[0]
#                print(ts)
                if len(info[2]) == 0:
                    continue
                #if planner did not find human invade, skip
                is_human_invade = False
#                print("ts: ", str(datetime.fromtimestamp(ts)))
                bias = 5
                for zone in invade_zone:
                    start_ts = zone[0]-bias
                    end_ts = zone[1]+bias
                    if start_ts < ts < end_ts:
                        is_human_invade = True
                        break
#                        print(datetime.fromtimestamp(ts), datetime.fromtimestamp(start_ts), datetime.fromtimestamp(end_ts))
                if not is_human_invade:
                    continue
                
                det_res = info[2]
                img_orig = cv2.imread(info[1])
                img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                fig = plt.figure()
                # ax1 = fig.add_subplot(1,2,1)
                # ax2 = fig.add_subplot(1,2,2)
                gs = gridspec.GridSpec(5, 5)
                ax1 = fig.add_subplot(gs[:, :-1])
                ax2 = fig.add_subplot(gs[1:3, 4])
                ax3 = fig.add_subplot(gs[3:, 4])
                #insert new pos info
                ax4 = fig.add_subplot(gs[4:, 4])
                ax4.axis('off')

                ax2.spines['top'].set_color('none')
                ax2.spines['right'].set_color('none')
                ax2.xaxis.set_ticks_position('bottom')
                ax2.spines['bottom'].set_position(('data', 0))
                ax2.yaxis.set_ticks_position('left')
                ax2.spines['left'].set_position(('data', 0))
                max_dis = 5
                # if len(det_res)>1:
                #     print("here!!!")
                #factors to caculate the ratio for each texts
                counter = len(det_res)
                ratio_factor = 1/counter
                is_human_find = False #only draw image with human
                for ele in det_res:
                    #randomize the rgb value for bbox and xy plot
                    rgb = (random.random(), random.random(), random.random())
                    r = int(rgb[0]*255)
                    g = int(rgb[1]*255)
                    b = int(rgb[2]*255)

                    top_left_x = int(ele[2])
                    top_left_y = int(ele[3])
                    bot_right_x = int(ele[4])
                    bot_right_y = int(ele[5])
                    pose_x = int(ele[6])
                    pose_y = int(ele[7])
                    obj_class = int(ele[10])
                    track_id = int(ele[11])

                    cv2.line(img, (top_left_x, top_left_y), (top_left_x, bot_right_y), (r, g, b), 5)
                    cv2.line(img, (bot_right_x, top_left_y), (bot_right_x, bot_right_y), (r, g, b), 5)
                    cv2.line(img, (top_left_x, top_left_y), (bot_right_x, top_left_y), (r, g, b), 5)
                    cv2.line(img, (top_left_x, bot_right_y), (bot_right_x, bot_right_y), (r, g, b), 5)
                    if obj_class == 5:
                        obj = "human"
                        is_human_find = True
                    else:
                        obj = str(obj_class)
                    text_obj = "Object class: " + obj
                    text_track = "track id: " + str(track_id)


                    #add new display feature to specify each object
                    #color match xy coordinates and bbox
                    #display coordinates bbox info
                    top_left_pos = '(' + str(top_left_x) + ',' + str(top_left_y) + ')'
                    bottom_right_pos = '(' + str(bot_right_x) + ',' + str(bot_right_y) + ')'
                    xy_info = 'xy: ' + '(' + str(pose_x) + ', ' + str(pose_y) + ') '
                    pos_info = xy_info + 'bbox: ' + top_left_pos + bottom_right_pos

                    #bbox style setup
                    boxstyle = BoxStyle("Round", pad=0.5)
                    props = {'boxstyle': boxstyle,
                             'facecolor': rgb,
                             'linestyle': 'solid',
                             'linewidth': 1,
                             'edgecolor': 'black'}
                    ax4.text(0, ratio_factor*counter, "id: " + str(track_id) + ' ' + pos_info, horizontalalignment='left',
                             verticalalignment='bottom',
                             fontsize=3,
                             bbox=props)
                    counter -= 1

                    cv2.putText(img, text_obj, (top_left_x, top_left_y+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)   
                    cv2.putText(img, text_track, (top_left_x, top_left_y+30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

                    ax2.scatter(pose_x, pose_y, c=[rgb])
                    max_dis = max(abs(pose_x), abs(pose_y), max_dis)
                if not is_human_find:
                    plt.close(fig)
                    continue
                ax1.imshow(img)
                ax2.set_xticks([-1 * max_dis, max_dis])
                ax2.set_yticks([-1 * max_dis, max_dis])
                ax2.annotate("Heading", xy=(0, max_dis), xytext=(0, max_dis+1), textcoords='offset points')
                
                ax3.set_xticks([-5, 5])
                ax3.set_yticks([-5, 5])
                ax3.axis('off')
                ax3.set_xticks([])
                ax3.set_yticks([])
                time = str(datetime.fromtimestamp(float(info[0]))).split()
                
                ax3.annotate(time[0], xy=(-5, 2), xytext=(-5, 2), textcoords='offset points')
                ax3.annotate(time[1], xy=(-5, 1), xytext=(-5, 1), textcoords='offset points')
                
                plt.savefig(dst_path + '/' + str(info[0]) + '.jpg', format='jpg',dpi=800)
                # print("save fig: " + dst_path + '/' + str(info[0]) + '.jpg')
                plt.close(fig)

        return
    
    def get_dump_distribution(self):
        
        print("get dump distribution")
        cap4_path = self.DM_PATH + "/human_dump_cap4"
        cap5_path = self.DM_PATH + "/human_dump_cap5"
        cap_list = [cap4_path, cap5_path]
        img_list = []
        for cap in cap_list:
            img_name = os.listdir(cap)
            for i in range(len(img_name)):
                if img_name[i][-4:] == ".jpg":
                    img_list.append(float(img_name[i][:-4].split("_")[0]))
        img_list.sort()
        df_cap = pd.DataFrame(img_list)
        df_cap.columns = ["time"]
        df_cap["vague_ts"] = df_cap["time"].astype("int")
        
        if self.NAVI_TINY_LOG == "" or not os.path.exists(self.NAVI_TINY_LOG):
            print("can not find navi tiny log")
            return
        df_navi_tiny = pd.read_csv(self.NAVI_TINY_LOG, "\t", header = None)
        # df_navi_tiny = pd.read_csv(self.NAVI_TINY_LOG, "\t", header = None, error_bad_lines=False)
        df_navi_tiny = df_navi_tiny[0].str.split(" ", expand = True)
        df_navi_tiny[0] = df_navi_tiny[0].str.split(":", expand = True)[1]
        df_navi_tiny.columns = ["east", "north", "height", "alpha", "beta", "theta", "position_state", "conf",
                                "time", "longitude", "latitude", "read_rcs_duration", "fusion_duration"]
        df_navi_tiny = df_navi_tiny[df_navi_tiny["conf"] == "1.00"]
        df_navi_key = pd.DataFrame(df_navi_tiny, columns = ["east", "north", "time"], dtype=np.float)
        df_navi_key["vague_ts"] = df_navi_key["time"].astype("int")
        df_cap_in_navi = pd.merge(df_navi_key, df_cap, on = "vague_ts")
        
        plt.scatter(df_navi_key["east"], df_navi_key["north"], c = '#3399FF', marker = '.')
        plt.scatter(df_cap_in_navi["east"], df_cap_in_navi["north"], c='r',marker='.')
        plt.title("Dump image location distribution")
        plt.xlabel("East")
        plt.ylabel("North")
        plt.legend(("Vehicle Trace", "Dump Spot"))
        plt.savefig(os.path.join(self.DM_PATH, 'dump_distribution.jpg'), format='jpg', dpi=1200)
        plt.close()
        print("job done")
        return
    
    def dump_info(self, name = ""):
        
        '''
        TODO: dump processed infomation to avoid recalculation
        '''
        if name not in self.KEY_INFO or self.KEY_INFO[name] == -1:
            print("get dump key info failed: ", name)
            return
        SUPPORT_DUMP_MATRIX = []
        if name not in SUPPORT_DUMP_MATRIX:
            print("current info is not in support dump info matrix: ", name)
            return
        
    def get_det_info_for_data_closed_loop(self):
        
        cap4_src_path = self.DM_PATH + "/human_dump_cap4"
        cap5_src_path = self.DM_PATH + "/human_dump_cap5"

        if self.KEY_INFO["DETECT_INFO"] == -1:
            print("can not find detection result")
            return
        
        detect_info = self.KEY_INFO["DETECT_INFO"]
        img4_name = os.listdir(cap4_src_path)
        img4_list = [] # format[[ts, path, im]]
        for i in range(len(img4_name)):
            if img4_name[i][-4:] == ".jpg":
                img4_list.append([float(img4_name[i][:-4].split("_")[0]), cap4_src_path + "/" + img4_name[i]])


        img5_name = os.listdir(cap5_src_path)
        img5_list = []
        for i in range(len(img5_name)):
            if img5_name[i][-4:] == ".jpg":
                img5_list.append([float(img5_name[i][:-4].split("_")[0]), cap5_src_path + "/" + img5_name[i]])
        
        # sort
        detect_info.sort(key=lambda x: x[1])
        img4_list.sort(key=lambda x: x[0])
        img5_list.sort(key=lambda x: x[0])
        # matching method

        def find_min_max_detect_time(ts, left, right):
            # return the index
            
            if left > right:
                print("input left and right error!")
                return -1
            if left == right:
                return left
            m = (left + right) // 2
            # print(left, right, m)
            if detect_info[m][1] <= ts:
                return find_min_max_detect_time(ts, m+1, right)
            else:
                return find_min_max_detect_time(ts, left, m)
        
        def get_candidate(img_list, stream):
            length = len(detect_info)
            for i in range(len(img_list)):
                print(i)
                candidate = img_list[i][0]
                if candidate > detect_info[-1][1]:
                    img_list[i].append([])
                    continue
                ts_min_max_idx = find_min_max_detect_time(candidate, 0, len(detect_info)-1)
                
                l = ts_min_max_idx-1
                r = ts_min_max_idx


                while (r<length and (detect_info[r][1] - candidate) <= 1):
                    if detect_info[r][0] == stream:
                        break
                    else:
                        r+=1
        
                while (l>=0 and (candidate - detect_info[l][1]) <= 1):
                    if detect_info[l][0] == stream:
                        break
                    else:
                        l-=1
            
                # print((detect_info[r][1] > candidate) , (candidate > detect_info[l][1]))

                r_ts_diff = detect_info[r][1] - candidate #left side
                l_ts_diff = candidate - detect_info[l][1] #right side
#                if 1 < min(l_ts_diff, r_ts_diff):
                if 0.2 < min(l_ts_diff, r_ts_diff):
                    img_list[i].append([])
                    # ts diff should be in +-200ms
                    print("failed to find best error case", l_ts_diff, r_ts_diff)
                    continue
                if l_ts_diff < r_ts_diff:
                    #pick small side
                    #print("pick small side", l_ts_diff)
                    cur_ts = detect_info[l][1]
                    cur_info = []
                    if detect_info[l][0] != stream:
                        img_list[i].append([])
                        continue
                    while (l >= 0 and detect_info[l][1] == cur_ts and detect_info[l][0] == stream):
                        cur_info.append(detect_info[l])
                        l-=1
                    img_list[i].append(cur_info)
                else:
                    #pick large side
                    print("pick large side", r_ts_diff)
                    cur_ts = detect_info[r][1]
                    cur_info = []
                    if detect_info[r][0] != stream:
                        img_list[i].append([])
                        continue
                    while (r < length and detect_info[r][1] == cur_ts and detect_info[r][0] == stream):
                        cur_info.append(detect_info[r])
                        r+=1
                    img_list[i].append(cur_info)
            return img_list
        
        img4_list = get_candidate(img4_list, 4)
        img5_list = get_candidate(img5_list, 5)
        df_img4_list = pd.DataFrame(img4_list, columns=["time_stamp", "img_path", "track_info"])
        df_img5_list = pd.DataFrame(img5_list, columns=["time_stamp", "img_path", "track_info"])
        df_img4_list.to_csv(self.DM_PATH + "/img4_track_res.csv")
        df_img5_list.to_csv(self.DM_PATH + "/img5_track_res.csv")

       
        
        
        
        return
            
        
#%%    
'''
  
if __name__ == "__main__":
    
    file_path = sys.argv[1]
    DM = DollyMonitor(file_path)
    DM.data_prepare()
    DM.get_key_info()
    # Test.human_invade_analysis()
    DM.get_detect_info()
    DM.plot_detect_res()
    DM.get_dump_distribution()
'''