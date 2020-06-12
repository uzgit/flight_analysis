import pandas
import matplotlib.pyplot as plt
import glob
import numpy

# for plotting whycon.png
from pylab import *
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png

# get all logs
log_directory = "/home/joshua/Documents/moving_land_testing_5ms/*.csv"
filenames = glob.glob(log_directory)

print(filenames)

# create 3d plot
figure = plt.figure(figsize=(15, 11))
axes = figure.gca(projection="3d")

# plot landing pad
# whycon_png = get_sample_data("/home/joshua/Pictures/whycon.png", asfileobj=True)
# img = read_png(whycon_png)
# x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
# axes.plot_surface(x, y, 0, rstride=5, cstride=5, facecolors=img)

fn = get_sample_data("/home/joshua/Pictures/whycon.png", asfileobj=True)
img = read_png(fn)
x, y = ogrid[0:img.shape[1], 0:img.shape[0]]

x = x.astype(float) / 225.0 - 0.5
y = y.astype(float) / 225.0 - 0.5

# ax = gca(projection='3d')
axes.plot_surface(x, y, 0, rstride=5, cstride=5, facecolors=img)

colors = ["black", "purple", "blue", "green", "yellow", "red"]

times = []
energies = []
descent_edges = []

for filename in filenames:
    # import flight data
    flight_data = pandas.read_csv(filename)

    # get only useful trajectories
    # landing_trajectory = flight_data.where((flight_data["whycon_detected"] == 1) or (flight_data["apriltag_detected"] == 1))
    # landing_trajectory = flight_data.query("whycon_detected == 1 | apriltag_detected == 1")
    landing_trajectory = flight_data.query("landing_phase != 5")

    # print(landing_trajectory.head())

    axes.plot(landing_trajectory["landing_pad_position_y"] - landing_trajectory["iris_position_y"],
              -landing_trajectory["landing_pad_position_x"] + landing_trajectory["iris_position_x"],
              landing_trajectory["iris_position_z"] - landing_trajectory["landing_pad_position_z"])

    times.append(landing_trajectory.time.iloc[-1] - landing_trajectory.time.iloc[0])
    energies.append(landing_trajectory.energy_consumed.iloc[-1] - landing_trajectory.energy_consumed.iloc[0])

    # for i in range(5):
    #
    #     segment = landing_trajectory.query("landing_phase != %s" % 5)
    # # plot 3d trajectory
    #     axes.plot(segment["landing_pad_position_x"] - segment["iris_position_y"], segment["landing_pad_position_y"] - segment["iris_position_x"], segment["iris_position_z"] - segment["landing_pad_position_z"], color = colors[i])

        # if( i == 4 ):
        #     start_time = segment.iloc[0].time
        #     # print(segment.head())
        # elif( i == 0 ):
        #     end_time = segment.iloc[0].time

    axes.view_init(elev=30, azim=20)
    axes.set_xlabel("East")
    axes.set_ylabel("North")
    axes.set_zlabel("Up")

    axes.set_xlim3d(-20, 20)
    axes.set_ylim3d(-20, 20)

plt.savefig("/home/joshua/Documents/moving_land_testing_5ms/moving_land_testing_5mps.png", bbox_inches="tight", pad_inches=-0.2, transparent=True)
# plt.show()

print(numpy.mean(times), numpy.std(times))
print(numpy.mean(energies), numpy.std(energies))