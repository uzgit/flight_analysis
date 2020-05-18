import pandas
import matplotlib.pyplot as plt
import glob
import numpy

# for plotting whycon.png
from pylab import *

# get all logs
log_directory = "/home/joshua/Documents/flight_data/*"
filenames = glob.glob(log_directory)
filename_format = "flight_data_({},{},{})_({},{},{})_({},{},{}).csv"

def parse_filename(filename):

    items = filename.split(",")
    kp3 = float(items[0].split("(")[1])
    ki3 = float(items[1])
    kd3 = float(items[2].split(")")[0])
    kp2 = float(items[2].split("(")[1])
    ki2 = float(items[3])
    kd2 = float(items[4].split(")")[0])
    kp1 = float(items[4].split("(")[1])
    ki1 = float(items[5])
    kd1 = float(items[6].split(")")[0])

    return [ kp3, ki3, kd3, kp2, ki2, kd2, kp1, ki1, kd1 ]

colors = ["black", "purple", "blue", "green", "yellow", "red"]

times = []
descent_edges = []

descent_y = numpy.arange(0, 15, 0.1)
descent_x = 0.17 * numpy.exp( 0.36 * descent_y )

for filename in filenames:

    print("Processing: %s..." % filename)

    fig = plt.figure(figsize=(5, 3))
    axes = fig.gca()

    # import flight data
    flight_data = pandas.read_csv(filename)

    # get only useful trajectories
    # landing_trajectory = flight_data.query("(whycon_detected == 1 | apriltag_detected == 1) & (landing_phase != 5)")
    landing_trajectory = flight_data.query("(landing_phase != 5)")
    landing_trajectory["plane_distance"] = np.sqrt((landing_trajectory.landing_pad_position_x - landing_trajectory.iris_position_x) ** 2 + (landing_trajectory.landing_pad_position_y - landing_trajectory.iris_position_y) ** 2)
    # landing_trajectory["distance_to_apriltag_marker"] = numpy.sqrt(landing_trajectory.iris_position_x ** 2 + (landing_trajectory.iris_position_y - 50) ** 2 + landing_trajectory.iris_position_z ** 2)

    # landing_trajectory["contiguous"] = landing_trajectory.time.shift(-1) - landing_trajectory.time < 1
    landing_trajectory["contiguous"] =  (landing_trajectory.time.shift(-1) - landing_trajectory.time < 5)
#(landing_trajectory.landing_phase.shift(-1) <= landing_trajectory.landing_phase) and

    # print(landing_trajectory)
    slices = landing_trajectory.query("contiguous == False").index.values

    start_time = 0
    end_time   = 0

    for i in range(len(slices) - 1):
        single_landing = landing_trajectory.loc[slices[i]:slices[i+1]]

        for ii in range(4):

            segment = single_landing.query("landing_phase == %s" % ii)
            if( len(segment) > 0 and segment.iloc[0].landing_phase == 2 ):
                segment = segment.iloc[1:]

            axes.plot(segment.plane_distance, segment.iris_position_z, color=colors[ii])

        times.append(end_time - start_time)

    axes.plot(descent_x, descent_y, color="black", linestyle="dashed")

    axes.set_xlim(0, 21)
    axes.set_ylim(0, 15)

    gains = parse_filename(filename)
    text = "$kp_3=%0.1f, ki_3=%0.1f, kd_3=%0.1f$\n$kp_2=%0.1f, ki_2=%0.1f, kd_2=%0.1f$\n$kp_1=%0.1f, ki_1=%0.1f, kd_1=%0.1f$" % (gains[0], gains[1], gains[2], gains[3], gains[4], gains[5], gains[6], gains[7], gains[8])
    plt.figtext(0.65, 0.2, text, wrap=True, horizontalalignment='center', fontsize=10)
    save_filename = filename.split("/")[-1]
    plt.savefig("/home/joshua/Pictures/figures/pid_tests/" + save_filename.split(".csv")[0] + ".png", bbox_inches="tight", pad_inches=0.05, transparent=True)

# plt.show()

print(times)