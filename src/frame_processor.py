# Author: Vaibhav Ganatra (t-vaganatra at microsoft dot com))

from constants import Constants
import cv2
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from mire_detection_graph import mire_locations
from mire_segmentation.get_center import segment_and_get_center
from numpy.lib.stride_tricks import sliding_window_view
import os
import pandas as pd
from scipy.signal import find_peaks
from utils import crop_around_center, dict_list, plot_supports
from typing import List

class FrameProcessor:
    def __init__(self, mire_seg_model_path=Constants.SEG_MODEL_PATH):
        self.model_path = mire_seg_model_path
        self.seg_model = segment_and_get_center(mire_seg_model_path)

    def find_signal(self, mire_intensities: dict, threshold_perc: float):
        """
        prepare the signal for break detection
        """
        signals = dict()
        for mire_num, data in mire_intensities.items():
            signals[mire_num] = dict()

            # angles for mire_intensities
            angle = np.asarray(list(data.keys()), dtype=np.float32)

            # mire_intensities for the corresponding angles
            intensities = np.asarray(list(data.values()), dtype=np.float32)

            signal_subset = intensities[angle < 180]
            mean_signal_intensity = np.mean(signal_subset)

            # finding a large window mean_filtered signal
            padded = np.pad(intensities, Constants.WINDOW_SIZE//2, 'reflect')
            view_ = sliding_window_view(padded, Constants.WINDOW_SIZE)
            mean_intensities = np.mean(view_, axis=1)

            intensities = np.convolve(intensities, np.ones(5)/5, mode='same')

            # diff from large window mean_filtered signal
            diff = mean_intensities - intensities

            # enhancing peaks
            # diff = np.concatenate([diff[-1:], diff, diff[:1]])
            # diff = np.convolve(diff, np.asarray(Constants.PEAK_ENHANCEMENT_CONVOLUTION), mode='valid')

            signals[mire_num] = {
                'angles': angle,
                'signal': diff,
                "threshold" : threshold_perc * mean_signal_intensity
                # 0.25 if sharpness > 54, 0.19 if sharpness > 20, else 0.15
            }
        return signals
    
    def find_supports(self, mire_signals: dict, out_dir: str):
        """
        find supports by finding angles which have the maximum number of breaks
        """
        support_breaks = []
        for mire_num, data in mire_signals.items():
            # skipping outer mires since they may add noise due to eyelashes
            if mire_num > 15:
                continue
            # skip black mires
            if mire_num % 2 == 0:
                continue
            angles = data['angles']
            signal = data['signal']
            threshold = data['threshold']

            plt.plot(angles, signal)

            left_support_angles = angles[(angles>150) & (angles<210)]
            left_support_points = signal[(angles>150) & (angles<210)]
            right_support_angles = angles[(angles>330) | (angles<30)]
            right_support_points = signal[(angles>330) | (angles<30)]

            # threshold = fetch_break_thresholds(mire_num)

            left_peaks = find_peaks(left_support_points, height=threshold)[0]
            plt.scatter(left_support_angles[left_peaks], left_support_points[left_peaks], c='r', label='left support peaks', marker='x')
            for peak in left_peaks:
                support_breaks.append({
                    'mire_num': mire_num,
                    'angle': left_support_angles[peak],
                    'support' : "left"
                })
            
            right_peaks = find_peaks(right_support_points, height=threshold)[0]
            plt.scatter(right_support_angles[right_peaks], right_support_points[right_peaks], c='g', label='right support peaks', marker='x')
            for peak in right_peaks:
                support_breaks.append({
                    'mire_num': mire_num,
                    'angle': right_support_angles[peak] - 360 if right_support_angles[peak] > 330 else right_support_angles[peak],
                    'support' : "right"
                })
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(out_dir, 'support_breaks_{}.png'.format(mire_num)))
            plt.close()
        support_breaks = pd.DataFrame(support_breaks, columns=['mire_num', 'angle', 'support'])

        left_angles = []
        right_angles = []
        for row_ in support_breaks.iterrows():
            _ , angle, side = row_[1]
            subset = support_breaks[abs(support_breaks['angle'] - angle) < 5]
            # checking if there are at least 3 breaks (including the current break) at similar angles
            if len(subset) > 3:
                if side == "left":
                    left_angles.append(angle)
                else:
                    right_angles.append(angle)
        left_support = np.mean(left_angles) if len(left_angles) > 0 else 180
        right_support = np.mean(right_angles) if len(right_angles) > 0 else 0
        return left_support, right_support
    
    def find_peaks(self, mire_signals: dict, left_support: float, right_support: float, out_dir: str, frame_num: float, mire_locations: dict):
        """
        finds peaks in the mire signals
        """
        breaks = []

        n_mires_full = 0
        for mire_num, data in mire_signals.items():
            angles = data['angles']
            signal = data['signal']
            n_points = int((left_support - 9 - right_support)/Constants.JUMP)

            if right_support + 4.5 < 0:
                right_support_ = right_support + 364.5
                angle_subset = angles[(angles > right_support_) | (angles < left_support - 4.5)]
                signal_subset = signal[(angles > right_support_) | (angles < left_support - 4.5)]
            else:
                angle_subset = angles[(angles > right_support + 4.5) & (angles < left_support - 4.5)]
                signal_subset = signal[(angles > right_support + 4.5) & (angles < left_support - 4.5)]
            
            if len(signal_subset) > 0.99 * n_points and mire_num % 2 == 1:
                n_mires_full = max(n_mires_full, mire_num)
        
        print(f"Found {n_mires_full} mires with full signal")
        if n_mires_full == 0:
            return pd.DataFrame(columns=['mire_num', 'angle', 'frame', 'time'])

        edge_angles = dict()
        for mire_num in mire_signals.keys():
            edge_angles[mire_num] = []
            angles = mire_signals[mire_num]['angles']
            location = np.asarray(list(mire_locations[mire_num].values()), dtype=np.float32)
            if mire_num % 2 == 0:
                continue
            if right_support + 4.5 < 0:
                right_support_ = right_support + 364.5
                angle_subset = angles[(angles > right_support_) | (angles < left_support - 4.5)]
                location_subset = location[(angles > right_support_) | (angles < left_support - 4.5)]
            else:
                angle_subset = angles[(angles > right_support + 4.5) & (angles < left_support - 4.5)]
                location_subset = location[(angles > right_support + 4.5) & (angles < left_support - 4.5)]
            dist = np.sqrt(np.sum(np.square(location_subset[1:] - location_subset[:-1]), axis=1))
            jumps = np.where(dist > 10)[0]
            if len(jumps) > 0:
                for idx in jumps:
                    edge_angles[mire_num].append(angle_subset[idx])
                    edge_angles[mire_num].append(angle_subset[idx+1])

            edge_angles[mire_num] = np.asarray(edge_angles[mire_num])            
        
        for mire_num, data in mire_signals.items():
            angles = data['angles']
            signal = data['signal']

            # new added code start
            # if len(signal) < 45 / Constants.JUMP:
            #     continue

            if right_support + 4.5 < 0:
                right_support_ = right_support + 364.5
                angle_subset = angles[(angles > right_support_) | (angles < left_support - 4.5)]
                signal_subset = signal[(angles > right_support_) | (angles < left_support - 4.5)]
            else:
                angle_subset = angles[(angles > right_support + 4.5) & (angles < left_support - 4.5)]
                signal_subset = signal[(angles > right_support + 4.5) & (angles < left_support - 4.5)]

            

            # threshold = fetch_break_thresholds(mire_num)
            threshold = data['threshold']
            peaks = find_peaks(signal_subset, height=threshold)[0]

            plt.plot(angle_subset, signal_subset)
            plt.scatter(angle_subset[peaks], signal_subset[peaks], c='r', label='peaks', marker='x')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(out_dir, 'mire_{}.png'.format(mire_num)))
            plt.close()

            for peak in peaks:
                if mire_num % 2 == 1 and mire_num >= n_mires_full:
                    if len(edge_angles[mire_num]) > 0:
                        diff = min(np.abs(edge_angles[mire_num] - angle_subset[peak]))
                    else:
                        diff = 100
                    if mire_num != 25 and len(edge_angles[mire_num + 2]) > 0:
                        diff2 = min(np.abs(edge_angles[mire_num + 2] - (angle_subset[peak])))
                        diff = min(diff, diff2)                        
                    if diff <= 5:
                        continue
                breaks.append({
                    'mire_num': mire_num,
                    'angle': angle_subset[peak],
                    'frame': frame_num,
                    'time' : frame_num / 30.0
                })
        return breaks

    def find_breaks(self, frame: np.array, mire_intensities: dict, mire_locations: dict, center: List, out_dir: str, frame_num: float, threshold_perc: float):
        """
        finds candidate tear film breaks in a frame
        """
        # Step-1: Extract signal
        signals = self.find_signal(mire_intensities, threshold_perc)
        
        # Step-2: Find supports in the frame
        support_out_dir = os.path.join(out_dir, 'supports')
        os.makedirs(support_out_dir, exist_ok=True)
        left, right = self.find_supports(signals, support_out_dir)
        plot_supports(frame, left, right, center, support_out_dir)

        # Step-3: Find the peaks in the frame
        signal_out_dir = os.path.join(out_dir, 'signals')
        os.makedirs(signal_out_dir, exist_ok=True)
        peaks = self.find_peaks(signals, left, right, signal_out_dir, frame_num, mire_locations)
        if len(peaks) == 0:
            logging.info("No breaks found in the frame: {}".format(frame_num))
            return pd.DataFrame(columns=['mire_num', 'angle', 'frame', 'time'])

        # Step-4: Create a dataframe with columns: mire_num, angle, frame, time
        for break_ in peaks:
            mire_num, angle, _ , _ = break_.values()
            x,y = mire_locations[mire_num][round(angle, 1)]
            cv2.circle(frame, (int(x), int(y)), 10, (0,0,255), 1)
        cv2.imwrite(os.path.join(out_dir, 'frame_with_breaks.png'), frame)
        peaks = pd.DataFrame(peaks, columns=['mire_num', 'angle', 'frame', 'time'])
        return peaks

    def preprocess_frame(self, frame: np.array, out_dir: str, crop_dims: List = [500,500]):
        """
        preprocesses the frame - detect image center, crop around center and segment mires
        """
        img = frame.copy()
        dims = img.shape[:2]

        img_temp = img
        center = (dims[0] // 2, dims[1] // 2)
        img_temp = crop_around_center(img_temp.copy(), center=center, crop_dims=[1000, 1000])
        mask, _ = self.seg_model.segment(img_temp)
        center = self.seg_model.get_center(mask)

        center = (int(center[0] + img.shape[0] // 2 - 500), int(center[1] + img.shape[1] // 2 - 500))         
        center = (int(center[0]), int(center[1]))

        img_temp = crop_around_center(img.copy(), center=center, crop_dims=crop_dims)
        img_gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

        # Obtain mire_mask from the image
        img_gray_with_channels = np.dstack((img_gray, img_gray, img_gray))
        mire_mask, _ = self.seg_model.segment(img_gray_with_channels)
        center = self.seg_model.get_center(mire_mask)
        center = (int(center[0]), int(center[1]))
        mire_mask[mire_mask > 0] = 1

        mire_mask *= 255
        mire_mask = 255 - mire_mask
        mire_mask = mire_mask.astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, "frame.png"), img_temp)
        cv2.imwrite(os.path.join(out_dir, "mire_mask.png"), mire_mask)
        cv2.imwrite(os.path.join(out_dir, "frame_gray.png"), img_gray)

        return img_temp, img_gray, mire_mask, center

    def extract_mire_intensities(self, frame: np.array, out_dir: str):
        """
        given a frame, extract frame center, mire locations and intensities
        """
        if os.path.isfile(out_dir + "/mire_locations.json") and os.path.isfile(out_dir + "/mire_intensities.json") and os.path.isfile(out_dir + "/frame.png") and os.path.isfile(out_dir + "/center.json"):
            img = cv2.imread(out_dir + "/frame.png")
            logging.info("Found mire locations and intensities in: {}, loading..".format(out_dir))
            mire_locs = json.load(open(out_dir + "/mire_locations.json", "r"))
            mire_intensities = json.load(open(out_dir + "/mire_intensities.json", "r"))
            mire_locs = {int(k):v for k,v in mire_locs.items()}
            for mire_ in mire_locs.keys():
                mire_locs[mire_] = {float(k):v for k,v in mire_locs[mire_].items()}
            mire_intensities = {int(k):v for k,v in mire_intensities.items()}
            for mire_ in mire_intensities.keys():
                mire_intensities[mire_] = {float(k):v for k,v in mire_intensities[mire_].items()}                
            center = json.load(open(out_dir + "/center.json", "r"))["center"]
            center = (int(center[0]), int(center[1]))
        else:
            img, img_gray, mire_mask, center = self.preprocess_frame(frame, out_dir)
            logging.info("Center of the frame: {}".format(center))
            mire_locs, mire_intensities = mire_locations(img_gray, mire_mask, center=center, n_mires=Constants.N_MIRES, start_angle=Constants.START_ANGLE, end_angle=Constants.END_ANGLE, jump=Constants.JUMP, out_dir=out_dir)
            json.dump(mire_locs, open(out_dir + "/mire_locations.json", "w"), default=dict_list)
            json.dump(mire_intensities, open(out_dir + "/mire_intensities.json", "w"), default=dict_list)
            json.dump({"center": center}, open(out_dir + "/center.json", "w"))
        
        return img, center, mire_locs, mire_intensities
