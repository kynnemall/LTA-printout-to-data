# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:20:10 2022

@author: martinkenny
"""

import math
import numpy as np
import streamlit as st
import plotly.express as px
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

def rotate_mask(mask, plot=False):
    origin, bottom_right, top_left = get_points(mask)
    if plot:
       plt.imshow(mask)
       plt.scatter(*bottom_right, c="red")
       plt.scatter(*origin, c="green")
    # construct triangle and use inverse tan to find the rotation angle
    bottom_length = abs(origin[0] - bottom_right[0])
    triangle_height = abs(origin[1] - bottom_right[1])
    anti_clockwise = origin[1] > bottom_right[1]
    angle = math.degrees(math.atan(triangle_height / bottom_length))
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

def coords_to_raw_data(coords, origin, top_left, bottom_right, time_in_s=450):
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
    return time, agg

def get_rotated(img):
    filtered = median(img)
    mask = filtered < 240
    rotated, *points = rotate_mask(mask)
    return rotated, points

def extract_curve(rotated, points, timelength):    
    skeleton = skeletonize(rotated)
    props = regionprops(label(skeleton))
    areas = [p.area for p in props]
    idx = np.argmax(areas)
    coords = props[idx].coords
    agg_curve = coords[coords[:, 1].argsort()]
    time, agg = coords_to_raw_data(agg_curve, *points, timelength)
    return time, agg
    
# App setup
st.set_page_config(page_title="LTA graph extractor",
                   page_icon="chart_with_upwards_trend",
                   layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# layout app in columns
col1, col2 = st.columns(2)
upload = col1.file_uploader("Upload scanned LTA curve",
                          type=("tif", "tiff", "png"))
time_in_seconds = col2.number_input("Time (seconds)")

if upload is not None and time_in_seconds > 0:
    img = imread(upload)
    h, w = img.shape
    col3, col4 = st.columns(2)
    # col3.image(img, caption='Uploaded Graph', use_column_width=True)
    
    with st.spinner("Correcting graph rotation"):
        rotated, *points = get_rotated(img)
    # col4.image(rotated, caption="Rotation Corrected", use_column_width=True)

    with st.spinner("Extracting curve data"):
        time, aggregation = extract_curve(rotated, *points, time_in_seconds)
    fig = px.line(x=time, y=aggregation, title="Extracted Aggregation Curve")
    fig.update_layout(xaxis_title="Time (minutes)")
    st.plotly_chart(fig)