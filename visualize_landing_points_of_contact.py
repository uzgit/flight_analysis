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
axes = figure.gca()

whycon_image = plt.imread("/home/joshua/Pictures/whycon.png")
axes.imshow(whycon_image, interpolation="bicubic", extent=[-0.5,0.5,49.5,50.5])

# fn = get_sample_data("/home/joshua/Pictures/whycon.png", asfileobj=True)
# img = read_png(fn)
# x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
#
# x = x.astype(float) * 1.0 / 225 - 0.5
# y = y.astype(float) * 1.0 / 225 + 49.5
#
# axes.plot_surface(x, y, 0, rstride=5, cstride=5, facecolors=img)

for filename in filenames:
    # import flight data
    flight_data = pandas.read_csv(filename)

    # get only useful trajectories
    # landing_trajectory = flight_data.where((flight_data["whycon_detected"] == 1) or (flight_data["apriltag_detected"] == 1))
    landing_trajectory = flight_data.query("whycon_detected == 1 | apriltag_detected == 1").iloc[-1]

    # plot 3d trajectory
    axes.scatter(landing_trajectory["iris_position_x"], landing_trajectory["iris_position_y"])

    axes.set_xlabel("East")
    axes.set_ylabel("North")
    # axes.set_zlabel("Up")

plt.show()