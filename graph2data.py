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

1. "start" to view the graph and note down the (X, Y) coordinates of the top
(upper left) and bottom (lower right) of the graph.
- Zoom in on the graph, ensuring that even the comments data are maintained in
the view window. The coordinates will then be printed out for you.
- By clicking the "home" icon, you can reset the view and zoom in again
if the first set of coordinates cut off some of the data

2. "extract" to run the image processing procedure.
You should zoom in on the top left results graph to verify that the red and 
green dots are at the ends of the X axis.
If they are not, re-run the "start" command to select better coordinates.
One suggestion is to use a larger value for the bottom right Y coordinate.

3. "test" to run the program tests to ensure the workflow is running correctly
on your machine.

Press CTRL + C to exit the program.
""")

time = """
Enter the time in seconds from the leftmost dot to the rightmost dot
of the X axis and press Enter\n
"""

if decision == "start":
    filepath = input("Paste the full path to the input image\n")
    show_image(filepath)
elif decision == "test":
    run_tests()
elif decision == "extract":
    filepath = input("Paste the full path to the input image\n")
    left = input("Enter the upper left X coordinate and press Enter\n")
    right = input("Enter the bottom right X coordinate and press Enter\n")
    top = input("Enter the upper left Y coordinate and press Enter\n")
    bottom = input("Enter the bottom right Y coordinate and press Enter\n")
    time_seconds = input(time)
    savename = input("Enter a filename to save the raw data and press Enter\n")
    process_graph(filepath, int(top), int(bottom), int(left), int(right),
                  int(time_seconds), savename)
else:
    print("Invalid option. Program closing.")