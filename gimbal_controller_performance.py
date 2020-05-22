import pandas
import matplotlib.pyplot as plt
import numpy
import glob

# flight_data_19_04_2020_16_08_42.csv
log_directory = "/home/joshua/Documents/gimbal_and_pose_estimation_testing (orig with oscillation)/"
filenames = glob.glob(log_directory + "*.csv")

x = 0
y = 1
z = 2
gimbal_x = 3
gimbal_y = 4

plots = []
for i in range(5):
    figure = plt.figure(figsize=(7,4))
    plots.append(figure.gca())

for filename in filenames:

    flight_data = pandas.read_csv(filename)

    start_time = flight_data["time"].iloc[0]
    flight_data["d_t"] = flight_data["time"] - start_time

    plots[x].plot(flight_data["d_t"], flight_data["landing_pad_camera_pose_position_x"])
    plots[y].plot(flight_data["d_t"], flight_data["landing_pad_camera_pose_position_y"])
    plots[z].plot(flight_data["d_t"], flight_data["landing_pad_camera_pose_position_z"])

    plots[gimbal_x].plot(flight_data["d_t"], flight_data["gimbal_x_position"])
    plots[gimbal_y].plot(flight_data["d_t"], flight_data["gimbal_y_position"])

    # break

for plot in plots:
    plot.set_xlabel("Time since landing platform recognition (s)")

plots[x].set_ylabel("X displacement of landing platform camera pose (m)")
plots[y].set_ylabel("Y displacement of landing platform camera pose (m)")
plots[z].set_ylabel("Z displacement of landing platform camera pose (m)")
plots[gimbal_x].set_ylabel("Gimbal yaw angle (rad)")
plots[gimbal_y].set_ylabel("Gimbal pitch angle (rad)")

# plots[x].savefig("/home/joshua/Pictures/figures/gimbal_x_position.png", bbox_inches="tight", pad_inches=0.05, transparent=True)

plt.show()