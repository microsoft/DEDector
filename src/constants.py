# Author: Vaibhav Ganatra (t-vaganatra at microsoft dot com))

class Constants:
    SAMPLES_PER_SECOND = 2
    SEG_MODEL_PATH = "./mire_segmentation/segment_and_get_center_epoch_557_iter_14.pkl"
    N_MIRES = 26
    JUMP = 0.5
    START_ANGLE = 0
    END_ANGLE = 360
    WINDOW_SIZE = 51
    VIDEO_TIME = 2
    PEAK_ENHANCEMENT_CONVOLUTION = [3,20,3]
    MAX_EVAL_DURATION = 15

    BLUR_THRESHOLD = 53
    EXTREME_BLUR_THRESHOLD = 20

    SHARP_INTENSITY_PERCENT = 0.25
    MEDIUM_INTENSITY_PERCENT = 0.19
    BLURRED_INTENSITY_PERCENT = 0.15
