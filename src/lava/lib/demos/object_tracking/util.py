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

def template_resize(template):
    """Function to make the size of the template compatible
       with the padding of lava Conv. Adds a row or/and a column
       of zeros if the dimension is even."""
    template_shape = template.shape
    if not np.mod(template_shape[0], 2):
        template = np.pad(template, ((0, 1), (0, 0)))
    if not np.mod(template_shape[1], 2):
        template = np.pad(template, ((0, 0), (1, 0)))
    return template

def grayscale_conversion(images):
    """Function to check the shape of the frames and convert
       to grayscale if they are not in grayscale."""
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


def scale_images(images, **kwargs):
    """Function to scale the images given the target
       dimension or target size."""
    dimension = kwargs.pop("dimension", None)

    if dimension is None:
        scale_factor = kwargs.pop("scale_factor", 1)
        width = int(images[0].shape[1] * scale_factor)
        height = int(images[0].shape[0] * scale_factor)
        dimension = (width, height)
    resized = []

    # resize image
    for image in images:
        resized.append(cv.resize(image, dimension, interpolation=cv.INTER_AREA))
    return resized


def draw_rectangule(frames, saliency_maps, template_shape, convolution_type='valid'):
    """ Function to draw a rectangle centered on the maximum location
        of provided the saliency map."""
    image_color_marked = []
    for saliency_map,frame in zip(saliency_maps,frames):
        # find location with the maximum value in the saliency map
        _, _, _, max_loc = cv.minMaxLoc(saliency_map)

        # determine the coordinates of the rectangle
        if convolution_type == 'same':
            rect_tl = [max_loc[0] - int(template_shape[1]/2),
                   max_loc[1] - int(template_shape[0]/2)]
            rect_br = [max_loc[0] + int(template_shape[1]/2),
                   max_loc[1] + int(template_shape[0]/2)]
        elif convolution_type == 'valid':
            rect_tl = [max_loc[0],
                       max_loc[1]]
            rect_br = [rect_tl[0] + template_shape[1],
                       rect_tl[1] + template_shape[0]]

        cv.rectangle(frame,
                     pt1=rect_tl,
                     pt2=rect_br,
                     color=(255, 0, 0),  # color red in RGB
                     thickness=3)       # line thickness
        image_color_marked.append(frame)

    return image_color_marked


def generate_animation(frames, output_maps, saliency_maps, save_video_name: str):
    """Function to create and save an animation of the frames, output maps and saliency
       maps of the tracking algorithm to the given directory."""
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


def generate_animation_compare(saliency_maps1, frames_rect1, saliency_maps2, frames_rect2, save_video_name: str):
    """Function to create and save an animation of the frames and saliency maps for Lava and OpenCV
       tracking algorithms to the given directory."""
    if len(saliency_maps1) == 0:
        raise ValueError("Please provide saliency maps for Lava implementation. Check whether the flag 'save_saliency_maps' is true.")

    tot_frames = len(saliency_maps1)
    max_match1 = np.max(saliency_maps1)
    max_match2 = np.max(saliency_maps2)

    min_match1 = np.min(saliency_maps1)
    min_match2 = np.min(saliency_maps2)


    # set up the plot structure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 15), sharex=True, sharey=True)
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_axis_off()
    plt.tight_layout()

    # initialize plots
    ax_saliency_maps1 = axes[0, 0].imshow(saliency_maps1[0], vmin=min_match1, vmax=max_match1)
    axes[0, 0].set_title("Saliency Map Lava", fontsize=25)
    plt.colorbar(ax_saliency_maps1, ax=axes[0,0], shrink=0.6)
    ax_frames_rect1 = axes[0, 1].imshow(frames_rect1[0], vmin=0, vmax=255)
    axes[0, 1].set_title("Tracking Lava", fontsize=25)
    ax_saliency_maps2 = axes[1, 0].imshow(saliency_maps2[0], vmin=min_match2, vmax=max_match2)
    axes[1, 0].set_title("Saliency Map OpenCV", fontsize=25)
    plt.colorbar(ax_saliency_maps2, ax=axes[1,0], shrink=0.6)
    ax_frames_rect2 = axes[1, 1].imshow(frames_rect2[0], vmin=0, vmax=255)
    axes[1, 1].set_title("Tracking OpenCV", fontsize=25)
    plt.tight_layout()


    def update_plot(i):
        """Updates all plots with data from time i"""
        ax_saliency_maps1.set_data(saliency_maps1[i])
        ax_frames_rect1.set_data(frames_rect1[i])
        ax_saliency_maps2.set_data(saliency_maps2[i])
        ax_frames_rect2.set_data(frames_rect2[i])
        return ax_saliency_maps1, ax_frames_rect1, ax_saliency_maps2, ax_frames_rect2

    # create an animated plot
    animation = FuncAnimation(fig,
                              update_plot,
                              frames=tot_frames,
                              interval=400,
                              repeat=False,
                              blit=True)

    # save the animation as a video file
    animation.save(save_video_name)


def precision(answers, groundtruths, threshold):
    """Function to calculate and save the L2 norm of the prediction to the ground
       truth for every frame, and calculate the precision score for the sequence."""
    frame_scores = []

    for answer, groundtruth in zip(answers, groundtruths):
        precision_score = np.linalg.norm(np.array(answer)-np.array(groundtruth))
        precision_bool = precision_score<=threshold
        frame_scores.append((precision_score, precision_bool))

    PRE = (100*np.sum(frame_scores, axis=0)[1])/len(frame_scores)
    return frame_scores, PRE

def get_answers(saliency_maps, **kwargs):
    """Function to get the saliency maps and finds the maximum value."""
    convolution_type = kwargs.pop('convolution_type', 'valid')
    if convolution_type == 'valid':
        template_shape = kwargs.pop('template_shape')
    elif convolution_type == 'same':
        template_shape = (0,0)
    answers = []
    for saliency_map in saliency_maps:
        # find location with the maximum value in the saliency map
        _, _, _, max_loc = cv.minMaxLoc(saliency_map)

        answers.append((np.array((max_loc[0], max_loc[1])) + (int((template_shape[1]+1)/2), int((template_shape[0]+1)/2))))
    return answers


def read_results(results_path):
    """Function to read the results and return a list of arrays
       holding the x and y values of the location."""
    results = []
    lines = []
    with open(results_path) as f:
        lines = f.readlines()

    for line in lines:
        results_read = line.strip().split(',')
        results.append((int(float(results_read[0])), int(float(results_read[1]))))
    return  results


def write_results(results, results_path):
    """Function to get the maximum activation locations
       and write these to new lines in a txt file."""
    with open(results_path, "w") as f:
        for line in results:
            total = str(line[0]) + ',' + str(line[1]) + "\n"
            f.write(total)

def write_precision(frame_scores, PRE, precision_path):
    """Function to write L2 norms of each frame to a txt file,
       and at the end of each line the precision score of the total sequence
       is written."""
    with open(precision_path, "w") as f:
        for line in frame_scores:
            total = str(line[0]) + ',' + str(line[1]) + ',' + str(PRE) +"\n"
            f.write(total)

def crop_template_from_frame(template_path, gt_path):
    """Function to crop the template from the frame given the
       ground truth information. The frame should be provided in grayscale."""
    with open(gt_path, "r") as f:
        first_line = f.readline()
        line_read = first_line.strip().split(',')
        x, y, width, height = int(line_read[0]), int(line_read[1]), int(line_read[2]), int(line_read[3])

        first_frame = cv.imread(template_path + '*000001.jpg')
        template = first_frame[y:y+height, x:x+width]
        plt.imshow(template)
        plt.show()
        cv.imwrite(template_path + "template.png", template)
    return template


def crop_template_from_scaled_frame(frame, scale_factor, template_path, gt_path):
    """Function to crop the template from the scaled frame given the
       ground truth and scaling factor information. The frame should
       be provided in grayscale."""
    with open(gt_path, "r") as f:
        first_line = f.readline()
        line_read = first_line.strip().split(',')
        x, y, width, height = int(int(line_read[0])*scale_factor), int(int(line_read[1])*scale_factor), int(int(line_read[2])*scale_factor), int(int(line_read[3])*scale_factor)
        original_template_shape = (int(line_read[3]), int(line_read[2]))

    template = frame[y:y+height, x:x+width]
    plt.imshow(template)
    plt.show()

    cv.imwrite(template_path + "template.png", template)
    return template, original_template_shape


def determine_scale_factor(template_path, gt_path, first_frame_name):
    """Functions that makes an estimate of the correct match score and returns a scaling factor
       which scales down the frame as if the average pixel value stays the same, while the summation
       of the whole kernel is normalized to the maximum possible value in terms of bit precision."""
    with open(gt_path, "r") as f:
        first_line = f.readline()
        line_read = first_line.strip().split(',')
        x, y, width, height = int(line_read[0]), int(line_read[1]), int(line_read[2]), int(line_read[3])

        first_frame = cv.imread(template_path + first_frame_name)
        template = first_frame[y:y + height, x:x + width]
        template = grayscale_conversion([template])[0]
    max_possible_value = 2**23 -1

    max_value_given_template = np.sum(np.multiply(template,template))

    if max_value_given_template > max_possible_value:
        scale_factor = np.sqrt(max_possible_value/max_value_given_template)
    else:
        scale_factor = 1

    return scale_factor
