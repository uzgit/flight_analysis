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
x, y = ogrid[0:img.shape[1], 0:img.shape[0]]

x = x.astype(float) * 1.0 / 225 - 0.5
y = y.astype(float) * 1.0 / 225 + 49.5

ax = gca(projection='3d')
# axes.plot_surface(x, y, 0, rstride=5, cstride=5, facecolors=img)

colors = ["black", "purple", "blue", "green", "yellow", "red"]

times = []
descent_edges = []

for filename in filenames:
    # import flight data
    flight_data = pandas.read_csv(filename)

    # get only useful trajectories
    # landing_trajectory = flight_data.where((flight_data["whycon_detected"] == 1) or (flight_data["apriltag_detected"] == 1))
    landing_trajectory = flight_data.query("whycon_detected == 1 | apriltag_detected == 1")

    start_time = 0
    end_time   = 0

    for i in range(5):

        segment = landing_trajectory.query("landing_phase == %s" % i)
    # plot 3d trajectory
        axes.plot(segment["landing_pad_position_x"] - segment["iris_position_y"], segment["landing_pad_position_y"] - segment["iris_position_x"], segment["iris_position_z"] - segment["landing_pad_position_z"], color = colors[i])

        if( i == 4 ):
            start_time = segment.iloc[0].time
            # print(segment.head())
        elif( i == 0 ):
            end_time = segment.iloc[0].time

    times.append(end_time - start_time)

    axes.set_xlabel("East")
    axes.set_ylabel("North")
    axes.set_zlabel("Up")

    axes.set_xlim3d(-10, 10)

plt.show()

print(times)