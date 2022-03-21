#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:49:04 2022

@author: martin
"""

import os
import numpy as np
import pandas as pd
from process import process_graph

def run_tests():
    """
    Load images and respective extracted data,
    run the image processing sequence, and compare the resulting data

    Returns
    -------
    None.

    """
    os.chdir("test_images")
    test_data = [{"filepath" : "test_600dpi_grayscale.tiff",
                 "top" : 199, "bottom" : 2610, "left" : 1030,
                 "right" : 3850, "time_seconds" : 450},
                 {"filepath" : "test_not_angled.tiff",
                  "top" : 199, "bottom" : 2610, "left" : 1030,
                  "right" : 3850, "time_seconds" : 450},
                 {"filepath" : "test_angled.tiff",
                  "top" : 272, "bottom" : 3000, "left" : 1199,
                  "right" : 3824, "time_seconds" : 240},
                 {"filepath" : "test_30_degrees.tiff",
                  "top" : 511, "bottom" : 3688, "left" : 705,
                  "right" : 4258, "time_seconds" : 315},
                 {"filepath" : "test_45_degrees.tiff",
                  "top" : 500, "bottom" : 4159, "left" : 690,
                  "right" : 4162, "time_seconds" : 315}
                 ]
    test_dfs = ["test_600dpi_grayscale.csv", "test_not_angled.csv",
                "test_angled.csv", "test_30_degrees.csv",
                "test_45_degrees.csv"]
    
    for i, (df, test) in enumerate(zip(test_dfs, test_data), 1):
        data = process_graph(**test, savename="", test=True)
        true_data = pd.read_csv(df)
        if np.allclose(data, true_data):
            print(f"Passed test number {i}")
        else:
            print(f"Failed test number {i}")
    os.chdir("..")
