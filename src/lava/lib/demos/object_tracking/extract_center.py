"""
Function to create bounding box annotations for DAVIS 2016 dataset.
The videos and annotations for object segmentation can be found at:
https://davischallenge.org/davis2016/code.html
"""


import numpy as np
import cv2 as cv
import glob
import os

def get_unique_numbers(numbers):
    unique = []
    for number in numbers:
        if number not in unique:
            unique.append(number)
    return unique

gt_path =  r"C:\Users\AYCA\Desktop\intel_backup\DAVIS\Annotations\480p"
result_name = r"\bounding_box.txt"

gt_names = []
lines = []
infos = []


objects = os.listdir(gt_path+ r'\\')

for object in objects:
    print(object, " is being processed.")
    gt_complete_path = gt_path + r'\\' + object + r'\\'

    frames = [cv.imread(file) for file in sorted(glob.glob(gt_complete_path + "*"))]

    for frame in frames:
        frame_gray = frame[:,:,1].reshape((480, 854))

        # boundingRect function returns to 4 values, location of the right top and the
        # length of the edges (x_loc, y_loc, width, height)
        bb = cv.boundingRect(frame_gray)
        infos.append(str(bb[0]) + "," + str(bb[1]) + "," +
                     str(bb[2]) + "," + str(bb[3]) + "," + object)

    with open(gt_complete_path + result_name, "w") as f:
        for info in infos:
            f.write(info + "\n")
        f.close()
    infos = []
