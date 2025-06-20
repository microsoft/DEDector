# Author: Vaibhav Ganatra (t-vaganatra at microsoft dot com))

import argparse
from constants import Constants
import cv2
from frame_processor import FrameProcessor
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm
from typing import List
from utils import calculate_frame_sharpness

class DEDector:
    def __init__(self):
        self.frame_processor = FrameProcessor()

    def detect_blinks(self, frames: List[np.array], out_dir: str, fps: float):
        """
        detects blink times in the video
        """
        logging.info("Detecting blinks in the video...")
        signal = []
        durations = []
        frame_count = 0
        for frame_count, frame in enumerate(frames):
            signal.append(frame.std())
            durations.append((frame_count + 1) / fps)
        

        signal = np.array(signal)      
        durations = np.array(durations)
        # pad signal with reflection of signal
        signal = np.pad(signal, 2, mode="reflect")
        # mean-filter the signal
        signal = np.convolve(signal, np.ones(5)/5, mode = "valid")

        # normalize the signal
        mean, std = signal.mean(), signal.std()
        signal = (signal - mean) / std
        # plot normalized signal
        self.plot_signal(durations, signal, "Time (secs)", "Std. Deviation", "Blink Detection", "Normalized Std. Deviation", out_dir, save=False)


        diff = signal[2:] - signal[:-2]
        self.plot_signal(durations[2:], diff, "Time (secs)", "Std. Deviation", "Blink Detection Aux", "Gradients", out_dir, scatter_plot=False, save=False)
        peaks = find_peaks(diff, height=0.5, distance = 5)[0]
        logging.info("Found blinks at: {}".format(peaks))
        # plot peaks
        self.plot_signal(durations[2:][peaks], diff[peaks], "Time (secs)", "Std. Deviation", "Blink Detection", "Blinks", out_dir, scatter_plot=True, save=True)
        plt.close()
        return durations[2:][peaks]
    
    def plot_signal(self, x: List, y: List, xlabel: str, ylabel: str, title: str, label: str, out_dir: str, scatter_plot: bool = False, save: bool = False):
        """
        custom plotting function to plot the provided signal
        """
        if scatter_plot:
            plt.scatter(x, y, label=label, marker="x", color="red")
        else:
            plt.plot(x, y, label=label)
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        title = title.replace(" ", "_").lower()
        if save:
            logging.info("Saving plot to: {}".format(os.path.join(out_dir, f"{title}.png")))
            plt.savefig(os.path.join(out_dir, f"{title}.png"))
    
    def collate_breaks(self, frame_to_process: List[int], out_dir: str):
        # consolidate all breaks found in individual frames into a single dataframe
        all_breaks = []
        for frame_num in frame_to_process:
            break_csv_path = os.path.join(out_dir, f"frame_{frame_num}", "breaks.csv")
            if not os.path.exists(break_csv_path):
                logging.warning("Breaks file not found for frame: {}, skipping...".format(frame_num))
                continue
            breaks = pd.read_csv(break_csv_path)
            all_breaks.append(breaks)
        if len(all_breaks) == 0:
            logging.warning("No breaks found in any frames, skipping...")
            return pd.DataFrame()
        all_breaks = pd.concat(all_breaks)
        all_breaks = all_breaks.sort_values(by=['time'])
        all_breaks = all_breaks.reset_index(drop=True)
        all_breaks.to_csv(os.path.join(out_dir, 'breaks.csv'), index=False)
        logging.info("Collated breaks from all frames, saved to: {}".format(os.path.join(out_dir, 'breaks.csv')))
        return all_breaks

    def filter_breaks(self, data: pd.DataFrame):
        """
        apply break filtering
        """
        filtered_breaks = []
        data = data.sort_values(by=['time'])
        data["angle"] = data["angle"].apply(lambda x : x - 360 if x > 270 else x)

        for break_ in data.iterrows():
            mire, angle, frame, time = break_[1]

            # ignoring black mires
            if mire % 2 == 0:
                continue
    
            # persistence filtering: removing breaks present for more than 5 seconds in outer mires and near supports
            if mire >= 13 or angle >= 150 or angle <= 30:
                subset = data[(data["mire_num"] == mire) & (abs(data["angle"] - angle) <= 5)]
                if len(subset["frame"].unique()) > 5 * Constants.SAMPLES_PER_SECOND:
                    continue

            # radial filtering: removing breaks that are radially adjacent
            if mire >= 15 or (angle < 20 or angle > 160):
                subset = data[(data["mire_num"] == mire + 2) | (data["mire_num"] == mire - 2)]
                subset = subset[(abs(subset["angle"] - angle) <= 6) & (subset["time"] == time)]
                if len(subset) != 0:
                    continue

            # proximity filtering: removing breaks that are seen within 20 degrees of each other
            if mire >= 13:
                subset = data[(data["mire_num"] == mire) & (data["time"] == time) & (abs(data["angle"] - angle) <= 20) & (abs(data["angle"] - angle) >= 2)]
                if len(subset) >= 1:
                    # if mire == _mire and abs(angle - _angle) < 5:
                        # print("Continuing due to 50 degree window")
                    continue
            
            # merge filtering: removing merge breaks 
            subset = data[((data["mire_num"] == mire + 1) | (data["mire_num"] == mire - 1)) & (data["time"] == time) & (abs(data["angle"] - angle) <= 2)]
            if len(subset) != 0:
                continue

            # eyelash-region filtering
            if mire >= 19 and (angle >= 60 and angle <= 120):
                continue
            if mire >= 15 and (angle >= 60 and angle < 119):        
                continue
            # specific filtering for large eyelashes
            if (mire == 11 and abs(angle - 105) <= 6) or (mire==13 and abs(angle - 106) <= 6):
                continue

            filtered_breaks.append({
                "mire_num": mire,
                "angle": angle,
                "frame": frame,
                "time": time
            })

        filtered_breaks = pd.DataFrame(filtered_breaks)
        return filtered_breaks

    def check_temporal_validity(self, mire: float, angle: float, frame: float, in_dir: str, sharpness: float):
        """
        given a mire break, check if it is temporally valid
        """
        if mire % 2 == 0:
            return True

        signals = []
        times = []
        frames = os.listdir(in_dir)
        frames = [int(f.split('_')[-1]) for f in frames if os.path.isdir(os.path.join(in_dir, f))]
        frames.sort()
        for frame_ in frames:
            try:
                intensities = json.load(open(os.path.join(in_dir, "frame_" + str(int(frame_)), 'mire_intensities.json')))
                signal_value = intensities[str(int(mire))][str(round(angle,1))]
                if angle == 0:
                    signal_value += intensities[str(int(mire))][str(round(360,1))] + intensities[str(int(mire))][str(round(0.5,1))]
                elif angle == 360:
                    signal_value += intensities[str(int(mire))][str(round(359.5,1))] + intensities[str(int(mire))][str(round(0,1))]
                else:
                    signal_value += intensities[str(int(mire))][str(round(angle - 0.5,1))] + intensities[str(int(mire))][str(round(angle + 0.5,1))]
                if mire > 15 and signal_value > 200:
                    continue
                signals.append((signal_value / 3.0))
                times.append(frame_ / 30.0)
            except:
                continue
        break_time = frame / 30.0

        if mire <= 7:
            prominence_threshold = 10
        elif mire <= 15:
            prominence_threshold = 8.5
        else:
            # 8 if blurred else 7
            prominence_threshold = 7 if sharpness > Constants.BLUR_THRESHOLD else 8

        try:
            padded = np.pad(signals, (7,7), 'reflect')
            filtered_signal = np.convolve(padded, np.ones(15)/15, mode='valid')
            plt.plot(times, filtered_signal)
            plt.savefig(f"{in_dir}/frame_{int(frame)}/temporal_signal_{mire}_{angle}.png")
            plt.close()

            peaks = find_peaks(-1 * filtered_signal, prominence = prominence_threshold, wlen = 40)
            peak_times = []
            for i,peak_ in enumerate(peaks[0]):
                left_base, right_base = peaks[1]['left_bases'][i], peaks[1]['right_bases'][i]
                if right_base - left_base < 8:
                    continue
                peak_times.append(times[peak_])
            if len(peak_times) == 0:
                return False
            else:
                peak_times = np.array(peak_times)
                diff = min(abs(peak_times - break_time))
                if diff <= 1:
                    return True
                else:
                    return False

        except:
            return False


    def calculate_nibut(self, filtered_breaks: pd.DataFrame, in_dir: str, mean_sharpness: str):
        """
        calculate the NIBUT (Non-Invasive Breakup Time) from the filtered breaks
        """
        for break_ in filtered_breaks.iterrows():
            mire, angle, frame , time = break_[1]

            # <=1 for clear, <1 for blurred
            if mean_sharpness > Constants.BLUR_THRESHOLD: 
                subset = filtered_breaks[(filtered_breaks["mire_num"] == mire) & (filtered_breaks["time"] >= time) & (filtered_breaks["time"] <= time + 1) & (filtered_breaks["angle"] >= angle - 5) & (filtered_breaks["angle"] <= angle + 5)]
            else:
                subset = filtered_breaks[(filtered_breaks["mire_num"] == mire) & (filtered_breaks["time"] >= time) & (filtered_breaks["time"] < time + 1) & (filtered_breaks["angle"] >= angle - 5) & (filtered_breaks["angle"] <= angle + 5)]

            n_frames = 8
            unique_frames = subset["frame"].unique()
            if len(unique_frames) >= n_frames:
                temporal_validity = self.check_temporal_validity(mire, angle, frame, in_dir, mean_sharpness)
                if temporal_validity:
                    print(f"Found tear filme break at: {time}, Mire: {mire}, Angle: {angle}")
                    return time
                else:
                    continue
        print("No Tear Breakup Found")
        return -1

    def calculate_video_sharpness(self, start_frame: float, end_frame: float, in_dir: float):
        """
        Calculate video sharpness based on the variance of the laplacian of the frames
        """
        mean_sharpness = 0
        frame_count = 0
        for frame_ in range(start_frame, end_frame):
            if os.path.exists(os.path.join(in_dir, "frame_" + str(int(frame_)), "frame.png")):
                frame = cv2.imread(os.path.join(in_dir, "frame_" + str(int(frame_)), "frame.png"))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = calculate_frame_sharpness(frame)
                mean_sharpness += sharpness
                frame_count += 1
        mean_sharpness /= frame_count
        logging.info("Mean sharpness of the video: {}".format(mean_sharpness))
        return mean_sharpness

    
    def process_video(self, video_path: str, out_dir: str):
        """
        end-to-end video processing function
        """

        logging.info("Reading video from path: {}".format(video_path))
        cap = cv2.VideoCapture(video_path)

        frames = []
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        logging.info("Loaded video: Video duration: {} secs".format(video_duration))

        # Step-1: Blink detection
        blink_secs = self.detect_blinks(frames, out_dir, fps)
        blink_rate = len(blink_secs) / video_duration        
        logging.info("Blink rate: {} blinks/sec".format(blink_rate))
        logging.info("Blinks occured at: {}".format(blink_secs))

        logging.info("Last blink occured at: {} secs".format(blink_secs[-1] if len(blink_secs) > 0 else "No blinks detected"))

        # Step-2,3: Extract mire intensities and find breaks
        video_name = os.path.basename(video_path).split('.mp4')[0]
        logging.info("Extracting mire intensities from the video...")
        sampling_rate = int(fps) // Constants.SAMPLES_PER_SECOND

        blink_secs = [s for s in blink_secs if s < Constants.MAX_EVAL_DURATION]
        if len(blink_secs) == 0:
            logging.warning("No blinks detected in the video, skipping mire extraction.")
            return

        start_frame = int((max(blink_secs)+1) * fps)
        logging.info(f"Examining frames after {max(blink_secs)} secs, frame number {start_frame}")
        end_frame = start_frame + int(Constants.VIDEO_TIME * fps)

        frames_to_process = []

        logging.info("Extracting mire intensities from video: {}".format(video_name))
        for frame_count, frame in tqdm(enumerate(frames)):

            if ((frame_count+1) % sampling_rate != 0) or (frame_count+1) <= start_frame:
                # Skipping frames before the last blink and frames that are not sampled
                continue
            if (frame_count+1) > end_frame:
                break

            # frame_out_dir = os.path.join(out_dir, f"frame_{int(frame_count+1)}")
            frame_out_dir = out_dir + "/" + f"frame_{int(frame_count+1)}"
            if not os.path.exists(frame_out_dir):
                os.makedirs(frame_out_dir)
            try:
            # rotate the frame if it is in landscape mode
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                logging.info(frame.shape)
                frame_cropped, center, mire_locs, mire_intensities = self.frame_processor.extract_mire_intensities(frame, frame_out_dir)
                logging.info("Extracted mire intensities from frame: {}".format(frame_count + 1))
                frames_to_process.append(frame_count + 1)


            except:
                logging.error("Error occured while processing frame: {}".format(frame_count))
                continue

        mean_sharpness = self.calculate_video_sharpness(start_frame, end_frame, out_dir)
        if mean_sharpness > Constants.BLUR_THRESHOLD:
            threshold_perc = Constants.SHARP_INTENSITY_PERCENT
        elif mean_sharpness > Constants.EXTREME_BLUR_THRESHOLD:
            threshold_perc = Constants.MEDIUM_INTENSITY_PERCENT
        else:
            threshold_perc = Constants.BLURRED_INTENSITY_PERCENT


        for frame_count, frame in tqdm(enumerate(frames)):

            if ((frame_count+1) % sampling_rate != 0) or (frame_count+1) <= start_frame:
                # Skipping frames before the last blink and frames that are not sampled
                continue
            if (frame_count+1) > end_frame:
                break

            # frame_out_dir = os.path.join(out_dir, f"frame_{int(frame_count+1)}")
            frame_out_dir = out_dir + "/" + f"frame_{int(frame_count+1)}"
            if not os.path.exists(frame_out_dir):
                os.makedirs(frame_out_dir)
            try:
                frame_cropped, center, mire_locs, mire_intensities = self.frame_processor.extract_mire_intensities(frame, frame_out_dir)
                breaks = self.frame_processor.find_breaks(frame_cropped, mire_intensities, mire_locs, center, frame_out_dir, frame_count + 1, threshold_perc)
                breaks.to_csv(os.path.join(frame_out_dir, "breaks.csv"), index=False)
                logging.info("Found breaks in frame: {}".format(frame_count + 1))

                frames_to_process.append(frame_count + 1)


            except:
                logging.error("Error occured while processing frame: {}".format(frame_count))
                continue

        logging.info("Processed {} frames from the video, collating breaks".format(len(frames_to_process)))
        # Step-4: Collate breaks from all frames
        all_breaks = self.collate_breaks(frames_to_process, out_dir)            

        # Step-5: Filter breaks
        all_breaks = self.filter_breaks(all_breaks)

        # Step-6: Calculate NIBUT
        nibut_timestamp = self.calculate_nibut(all_breaks, out_dir, mean_sharpness)

        if nibut_timestamp == -1:
            logging.warning("No NIBUT found in the video.")
        else:
            nibut = nibut_timestamp - max(blink_secs) 
            with open(os.path.join(out_dir, "nibut.json"), 'w') as f:
                json.dump({
                    "blink": max(blink_secs),
                    "tear_break": nibut_timestamp, 
                    "nibut": nibut}, 
                    f)
    

    
def main():
    """
    func main - main function to run the DEDector pipeline
    @param args - command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to the video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing videos")
    parser.add_argument("--frame", type=str, help = "Path to the frame image")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save the results")

    args = parser.parse_args()
    assert args.video or args.frame or args.video_dir, "Please provide either a video file or a frame image or a directory containing videos."

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(args.out_dir, "dedector.log"), filemode='w')

    dedector = DEDector()

    if args.video_dir:
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
        for video_file in video_files:
            video_path = os.path.join(args.video_dir, video_file)
            logging.info("Processing video: {}".format(video_path))
            video_name = video_file.split('.mp4')[0]
            out_dir = os.path.join(args.out_dir, video_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            dedector.process_video(video_path, out_dir)
            break
    elif args.video:
        video_path = args.video
        video_name = os.path.basename(video_path).split('.mp4')[0]
        args.out_dir = os.path.join(args.out_dir, video_name)
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        logging.info("Processing video: {}".format(video_path))
        dedector.process_video(video_path, args.out_dir)
    elif args.frame:
        pass
    logging.info("Processing completed. Results saved to: {}".format(args.out_dir))

if __name__ == "__main__":
    main()