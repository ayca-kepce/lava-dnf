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


def grayscale_conversion(images):
    converted_images =[]
    # check if the frame is in gray scale, convert if not
    if len(images[0].shape) == 3:
        if images[0].shape[-1] == 3 or images[0].shape[-1] == 4:
            for image in images:
                converted_images.append(cv.cvtColor(image.astype('float32'), cv.COLOR_BGR2GRAY))
        else:
            raise ValueError("Please provide the frame either RGB or grayscale.")
    elif len(images[0].shape) > 3:
        raise ValueError("Please provide the frame either RGB or grayscale.")
    return converted_images


def scale_image(images, scale_factor):
    resized = []
    width = int(images[0].shape[1] * scale_factor)
    height = int(images[0].shape[0] * scale_factor)
    dim = (width, height)

    # resize image
    for image in images:
        resized.append(cv.resize(image, dim, interpolation=cv.INTER_AREA))
    return resized


def normalize_image(image):
    normalized_image = np.zeros_like(image)
    cv.normalize(image, normalized_image, 0, 255, cv.NORM_MINMAX)
    return normalized_image


def draw_rectangule(frames, saliency_maps, template_shape):
    """ Marks the frames with tracking rectangle
        based on the saliency map."""
    image_color_marked = []
    for saliency_map,frame in zip(saliency_maps,frames):
        # find location with the maximum value in the saliency map
        _, _, _, max_loc = cv.minMaxLoc(saliency_map)

        # determine the coordinates of the rectangle (see above)
        rect_tl = [max_loc[0] - int(template_shape[1]/2),
                   max_loc[1] - int(template_shape[0]/2)]
        rect_br = [max_loc[0] + int(template_shape[1]/2),
                   max_loc[1] + int(template_shape[0]/2)]

        cv.rectangle(frame,
                     pt1=rect_tl,
                     pt2=rect_br,
                     color=(255, 0, 0),  # color red in RGB
                     thickness=10)       # line thickness
        image_color_marked.append(frame)

    return image_color_marked


def generate_animation(frames, output_maps, saliency_maps, save_video_name: str):

    tot_frames = len(frames)
    max_match = np.max(saliency_maps)

    # set up the plot structure
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 15))
    for i in range(axes.shape[0]):
        axes[i].set_axis_off()
    plt.tight_layout()

    # initialize plots
    ax_saliency_map = axes[0].imshow(saliency_maps[0], vmin=0, vmax=max_match)
    axes[0].set_title("Saliency Maps", fontsize=30)
    ax_frames = axes[1].imshow(output_maps[0], vmin=0, vmax=255)
    axes[1].set_title("Output Maps", fontsize=30)
    ax_frames = axes[2].imshow(frames[0], vmin=0, vmax=255)
    axes[2].set_title("Frames", fontsize=30)

    def update_plot(i):
        """Updates all plots with data from time i"""
        ax_saliency_map.set_data(saliency_maps[i])
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
