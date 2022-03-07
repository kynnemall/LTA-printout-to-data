<img src="https://github.com/kynnemall/LTA-printout-to-data/blob/main/example.png" width="600"/>

# LTA-printout-to-data
A terminal app that takes high-quality images and extracts aggregation curves so you can create custom graphs with the raw data.

### How it works
1. The app processes 600 DPI TIFF or PNG images to achieve accurate data recovery from graphs containing a single aggregation curve.
2. Segmentation techniques are used to estimate tilt in the graph and align the axes with the image edges.
3. Coordinates of processed masks are used to extract the raw data and remove artefacts.
4. The raw data is then saved in a CSV format which can be opened with Excel or other data processing software.

### Running the program
1. Simply open a terminal, run `python graph2data.py` and follow the instructions onscreen.
2. Before you can extract the graph data, you need to specify coordinates to allow the program to work on a subsection of the image you provide
(this will ensure faster processing)
3. Once the image of your graph shows up, you can zoom in and hover over the top left and bottom right, noting down the (x, y) coordinates to provide in the processing step.
4. Now you can run the image processing step and get your raw data!
