# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:20:10 2022

@author: martinkenny
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import median
from skimage.transform import rotate
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label

def distance(x0, y0, x1, y1):
    x_diff = x1 - x0
    y_diff = y1 - y0
    squared = x_diff*x_diff + y_diff*y_diff
    d = squared ** 0.5
    return d

def closest_point(pt_x, pt_y, xvals, yvals):
    distances = [distance(pt_x, pt_y, x, y) for x,y in zip(xvals, yvals)]
    idx = np.argmin(distances)
    return xvals[idx], yvals[idx]

def get_points(mask):
    h, w = mask.shape
    props = regionprops(label(mask))
    xvals = [p.centroid[1] for p in props if 20 < p.area < 116]
    yvals = [p.centroid[0] for p in props if 20 < p.area < 116]
    origin = closest_point(0, h, xvals, yvals)
    bottom_right = closest_point(w, h, xvals, yvals)
    top_left = closest_point(0, 0, xvals, yvals)
    return origin, bottom_right, top_left

def rotate_mask(mask):
    origin, bottom_right, top_left = get_points(mask)
    ax["B"].imshow(mask)
    ax["B"].scatter(*bottom_right, c="red")
    ax["B"].scatter(*origin, c="green")
    ax["B"].axis("off")
    # construct triangle and use inverse tan to find the rotation angle
    bottom_length = abs(origin[0] - bottom_right[0])
    triangle_height = abs(origin[1] - bottom_right[1])
    anti_clockwise = origin[1] > bottom_right[1]
    angle = math.degrees(math.atan(triangle_height / bottom_length))
    deg_sym = u'\N{DEGREE SIGN}'
    ax["B"].set_title(f"Rotated image\nOffset angle: {angle:.2f}{deg_sym}")
    # set point about which to rotate the x axis
    if anti_clockwise:
        point = bottom_right
    else:
        point = origin
    # rotate mask        
    if anti_clockwise:
        angle = 360 - angle
    rotated = rotate(mask, angle, resize=True, center=point)
    origin, bottom_right, top_left = get_points(rotated)
    return rotated, origin, top_left, bottom_right

def coords_to_raw_data(coords, origin, top_left, bottom_right, time_in_s,
                       plot=True):
    minus10 = top_left[1]
    plus110 = origin[1]
    minute0 = origin[0]
    minute7 = bottom_right[0]
    sec_per_px = (time_in_s / (minute7 - minute0))
    perc_agg_per_px = (120 / (plus110 - minus10))
    
    # subtract minima and divide to get units
    # coords[0] is time, coords[1] is % aggregation
    if coords.shape[0] % 2 != 0:
        coords = np.concatenate((coords, np.expand_dims(coords[-1, :], axis=0)))
    time = coords[:, 1] - coords[:, 1].min() #minute0
    agg = coords[:, 0] - minus10
    
    time = time * sec_per_px
    agg = agg * perc_agg_per_px - 10
    # correct for y-axis label at 0 %
    # time, agg = fix_erratic_sequences(time, agg)
    return time, agg

def fix_erratic_sequences(x, y):
    block_idx = (x > 40) * (x < 58)
    block_fill = y[x < 40].mean()
    idx = np.where(block_idx)
    y[idx] = block_fill
    return x, y

if __name__ == "__main__":
    # modify these parameters
    filename = "test_600dpi_grayscale.tiff"
    top_idx = 199       # x at top left in ImageJ
    bottom_idx = 2610   # y at top left in ImageJ
    left_idx = 1030     # x at bottom right in ImageJ
    right_idx = 3850    # y at bottom right in ImageJ
    time_seconds = 450
    savename = "savename.csv"
    
    # filename = "test_not_angled.tiff"
    # top_idx = 199       # x at top left in ImageJ
    # bottom_idx = 2610   # y at top left in ImageJ
    # left_idx = 1030     # x at bottom right in ImageJ
    # right_idx = 3850    # y at bottom right in ImageJ
    # time_seconds = 240
    
    filename = "test_angled.tiff"
    top_idx = 272       # x at top left in ImageJ
    bottom_idx = 3000   # y at top left in ImageJ
    left_idx = 1199   # x at bottom right in ImageJ
    right_idx = 3824    # y at bottom right in ImageJ
    time_seconds = 240
    
    img = imread(filename)
    filtered = median(img[top_idx:bottom_idx, left_idx:right_idx])
    
    ax = plt.figure(figsize=(7, 7), constrained_layout=True).subplot_mosaic(
        """
        AC
        BC
        """
        )
    ax["A"].imshow(filtered)
    ax["A"].axis("off")
    ax["A"].set_title("Input image")
    mask = filtered < 240
    rotated, *points = rotate_mask(mask)
    
    skeleton = skeletonize(rotated)
    props = regionprops(label(skeleton))
    areas = [p.area for p in props]
    idx = np.argmax(areas)
    coords = props[idx].coords
    agg_curve = coords[coords[:, 1].argsort()]
    time, agg = coords_to_raw_data(agg_curve, *points, time_seconds)
    ax["C"].plot(time, agg)
    ax["C"].set_ylabel("% aggregation")
    ax["C"].set_xlabel("Time (seconds")
    df = pd.DataFrame({"time (seconds)" : time, "% aggregation" : agg})
    # df.to_csv(savename, index=False)