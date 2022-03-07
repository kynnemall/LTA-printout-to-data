#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:11:46 2022

@author: martin
"""

from tests import run_tests
from process import show_image, process_graph

decision = input("""
Type "a" and press enter to view the graph and note down the
(x, y) coordinates of the top (upper left) and bottom (lower right) 
of the graph.

Type "test" to run the program to tests to ensure the workflow works
as desired.

Otherwise, type nothing and press enter to proceed to image processing.
""")
                 
if decision == "a":
    filepath = input("Paste the full path to the input image\n")
    show_image(filepath)
elif decision == "test":
    run_tests()
else:
    filepath = input("Paste the full path to the input image\n")
    top = input("Enter the Y coordinate of the upper left and press Enter\n")
    left = input("Enter the X coordinate of the upper left and press Enter\n")
    bottom = input("Enter the Y coordinate of the bottom right and press Enter\n")
    right = input("Enter the X coordinate of the bottom right and press Enter\n")
    time_seconds = input("Enter the time in seconds of the X axis and press Enter\n")
    savename = input("Enter a filename to save the raw data and press Enter\n")
    process_graph(filepath, int(top), int(bottom), int(left), int(right),
                  int(time_seconds), savename)
