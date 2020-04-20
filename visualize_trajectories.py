import pandas
import matplotlib.pyplot as plt
import glob
import numpy

# for plotting whycon.png
from pylab import *
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png

# get all logs
log_directory = "/home/joshua/Documents/flight_data/*"
filenames = glob.glob(log_directory)

# create 3d plot
figure = plt.figure()
axes = figure.gca(projection="3d")

# plot landing pad
# whycon_png = get_sample_data("/home/joshua/Pictures/whycon.png", asfileobj=True)
# img = read_png(whycon_png)
# x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
# axes.plot_surface(x, y, 10, rstride=5, cstride=5, facecolors=img)

fn = get_sample_data("/home/joshua/Pictures/whycon.png", asfileobj=True)
img = read_png(fn)
x, y = ogrid[0:img.shape[0], 0:img.shape[1]]

x = x.astype(float) * 1.0/225 - 0.5
y = y.astype(float) * 1.0/225 + 49.5

ax = gca(projection='3d')
axes.plot_surface(x, y, 0, rstride=5, cstride=5, facecolors=img)

for filename in filenames:
    
    # import flight data
    flight_data = pandas.read_csv(filename)

    # get only useful trajectories
    # landing_trajectory = flight_data.where((flight_data["whycon_detected"] == 1) or (flight_data["apriltag_detected"] == 1))
    landing_trajectory = flight_data.query("whycon_detected == 1 | apriltag_detected == 1")

    # plot 3d trajectory
    axes.plot(landing_trajectory["iris_position_x"], landing_trajectory["iris_position_y"], landing_trajectory["iris_position_z"])

    axes.set_xlabel("East")
    axes.set_ylabel("North")
    axes.set_zlabel("Up")

plt.show()