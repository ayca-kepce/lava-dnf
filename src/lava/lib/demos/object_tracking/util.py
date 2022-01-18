# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (process models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from matplotlib.animation import FuncAnimation


def determine_scale_factor(template_normalized):
    num_weight_bits = 8
    scale_factor = int(np.log2(np.abs(np.max(template_normalized)))) - num_weight_bits + 1
    return scale_factor


def grayscale_conversion(image):
    # check if the image is a numpy array, convert if not
    if not np.ndarray == type(np.ndarray):
        image = np.array(image)

    # check if the frame is in gray scale, convert if not
    if len(image.shape) == 3:
        if image.shape[-1] == 3 or image.shape[-1] == 4:
            image = cv.cvtColor(image.astype('float32'), cv.COLOR_BGR2GRAY)
        else:
            raise ValueError("Please provide the frame either RGB or grayscale.")
    elif len(image.shape) > 3:
        raise ValueError("Please provide the frame either RGB or grayscale.")
    return image


def scale_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)

    # resize image
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    return resized


def normalize_image(image):
    normalized_image = np.zeros_like(image)
    cv.normalize(image, normalized_image, 0, 255, cv.NORM_MINMAX)
    return normalized_image


def draw_rectangule(frame, saliency_map, template_shape):
    """ Marks the frames with tracking rectangle
        based on the saliency map."""

    # find location with the maximum value in the saliency map
    _, _, _, max_loc = cv.minMaxLoc(saliency_map)

    # determine the coordinates of the rectangle (see above)
    rect_tl = [max_loc[0] - int(template_shape[1]/2),
               max_loc[1] - int(template_shape[0]/2)]
    rect_br = [max_loc[0] + int(template_shape[1]/2),
               max_loc[1] + int(template_shape[0]/2)]

    image_color_marked = np.copy(frame)
    cv.rectangle(image_color_marked,
                 pt1=rect_tl,
                 pt2=rect_br,
                 color=(255, 0, 0),  # color red in RGB
                 thickness=10)       # line thickness

    return image_color_marked


def generate_animation(frames, saliency_map, save_video_name: str):

    tot_frames = len(frames)
    max_match = np.max(saliency_map)

    # set up the plot structure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 15))
    for i in range(axes.shape[0]):
        axes[i].set_axis_off()
    plt.tight_layout()

    # initialize plots
    ax_saliency_map = axes[0].imshow(saliency_map[0], vmin=0, vmax=max_match)
    axes[0].set_title("Saliency Maps", fontsize=30)
    ax_frames = axes[1].imshow(frames[0], vmin=0, vmax=255)
    axes[1].set_title("Frames", fontsize=30)

    def update_plot(i):
        """Updates all plots with data from time i"""
        ax_saliency_map.set_data(saliency_map[i])
        ax_frames.set_data(frames[i])
        return ax_saliency_map, ax_frames

    # create an animated plot
    animation = FuncAnimation(fig,
                              update_plot,
                              frames=tot_frames,
                              interval=100,
                              repeat=False,
                              blit=True)

    # save the animation as a video file
    animation.save(save_video_name)
