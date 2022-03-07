<img src="https://github.com/kynnemall/LTA-printout-to-data/blob/main/example.png" width="600"/>

# LTA-printout-to-data
A terminal app that takes high-quality images and extracts aggregation curves so you can create custom graphs with the raw data.

### How it works
The app was developed with scanned 600 DPI tiff images to ensure maximum data quality. 600 DPI PNG images will also work.

### Running the program
Simply open a terminal, run `python graph2data.py` and follow the instructions.
1. Before you can extract the graph data, you need to specify coordinates to allow the program to work on a subsection of the image you provide
(this will ensure faster processing)
2. Once the image of your graph shows up, you can zoom in and hover over the top left and bottom right, noting down the (x, y) coordinates to provide in the processing step.

