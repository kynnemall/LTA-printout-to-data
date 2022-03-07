#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:11:46 2022

@author: martin
"""

from tests import run_tests
from process import show_image, process_graph

decision = input("""
Type one of the followings command and press Enter to run that command:

1. "start" to view the graph and note down the (x, y) coordinates of the top
(upper left) and bottom (lower right) of the graph.

2. "extract" to run the image processing procedure.

3. "test" to run the program tests to ensure the workflow works as desired.

Press CTRL + C to exit the program.
""")

if decision == "start":
    filepath = input("Paste the full path to the input image\n")
    show_image(filepath)
elif decision == "test":
    run_tests()
elif decision == "extract":
    filepath = input("Paste the full path to the input image\n")
    top = input("Enter the Y coordinate of the upper left and press Enter\n")
    left = input("Enter the X coordinate of the upper left and press Enter\n")
    bottom = input("Enter the Y coordinate of the bottom right and press Enter\n")
    right = input("Enter the X coordinate of the bottom right and press Enter\n")
    time_seconds = input("Enter the time in seconds of the X axis and press Enter\n")
    savename = input("Enter a filename to save the raw data and press Enter\n")
    process_graph(filepath, int(top), int(bottom), int(left), int(right),
                  int(time_seconds), savename)
else:
    print("Invalid option. Program closing.")
