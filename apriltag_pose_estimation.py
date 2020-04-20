import pandas
import matplotlib.pyplot as plt
import numpy
import glob

# flight_data_19_04_2020_16_08_42.csv
log_directory = "/home/joshua/Documents/flight_data/*.csv"
filenames = glob.glob(log_directory)
# filenames = [log_directory[:-1] + '/flight_data_19_04_2020_18_24_09.csv']

temp = plt.figure()
temp_ax = temp.gca()
apriltag_pose_estimate_figure = plt.figure()
apriltag_pose_estimate_axes = apriltag_pose_estimate_figure.gca(projection="3d")
apriltag_estimate_error_figure = plt.figure()
apriltag_estimate_error_axes = apriltag_estimate_error_figure.gca()
apriltag_error_vs_angular_velocity_figure = plt.figure()
apriltag_error_vs_angular_velocity_axes = apriltag_error_vs_angular_velocity_figure.gca()
apriltag_error_vs_camera_angular_velocity_figure = plt.figure()
apriltag_error_vs_camera_angular_velocity_axes = apriltag_error_vs_camera_angular_velocity_figure.gca()
apriltag_error_vs_camera_angular_acceleration_figure = plt.figure()
apriltag_error_vs_camera_angular_acceleration_axes = apriltag_error_vs_camera_angular_acceleration_figure.gca()
# apriltag_error_vs_camera_angular_jerk_figure = plt.figure()
# apriltag_error_vs_camera_jerk_acceleration_axes = apriltag_error_vs_camera_angular_jerk_figure.gca()

all_data = pandas.DataFrame()

num_readings = 0
num_outliers = 0
for filename in filenames:
    flight_data = pandas.read_csv(filename)

    # get only useful trajectories
    landing_trajectory = flight_data.where(flight_data["apriltag_detected"] == 1)

    # use ENU
    landing_trajectory["distance_to_apriltag_marker"] = numpy.sqrt(landing_trajectory.iris_position_x ** 2 + (landing_trajectory.iris_position_y - 50) ** 2 + landing_trajectory.iris_position_z ** 2)
    landing_trajectory["planar_distance_estimate_to_apriltag_marker"] = numpy.sqrt(landing_trajectory.apriltag_position_x ** 2 + landing_trajectory.apriltag_position_y ** 2)
    # landing_trajectory["apriltag_estimate_position_x"] = landing_trajectory.iris_position_x + numpy.sin(landing_trajectory.iris_yaw) * landing_trajectory.planar_distance_estimate_to_apriltag_marker# + numpy.sin(landing_trajectory.apriltag_position_y)
    # landing_trajectory["apriltag_estimate_position_y"] = landing_trajectory.iris_position_y + numpy.cos(landing_trajectory.iris_yaw) * landing_trajectory.planar_distance_estimate_to_apriltag_marker
    landing_trajectory["apriltag_estimate_position_x"] = landing_trajectory.iris_position_x + landing_trajectory.apriltag_position_x
    landing_trajectory["apriltag_estimate_position_y"] = landing_trajectory.iris_position_y + landing_trajectory.apriltag_position_y
    landing_trajectory["apriltag_estimate_position_z"] = landing_trajectory.iris_position_z + landing_trajectory.apriltag_position_z
    landing_trajectory["apriltag_estimate_error"] = numpy.sqrt(landing_trajectory.apriltag_estimate_position_x ** 2 + (landing_trajectory.apriltag_estimate_position_y - 50) ** 2 + landing_trajectory.apriltag_estimate_position_z ** 2)

    # remove extreme outliers
    # landing_trajectory = landing_trajectory.where(numpy.abs(landing_trajectory["apriltag_estimate_position_x"] - 0) < 5)
    # landing_trajectory = landing_trajectory.where(numpy.abs(landing_trajectory["apriltag_estimate_position_y"] - 50) < 5)
    # landing_trajectory = landing_trajectory.where(numpy.abs(landing_trajectory["apriltag_estimate_position_z"] - 0) < 5)

    landing_trajectory["iris_angular_velocity_magnitude"] = numpy.sqrt(landing_trajectory.iris_angular_velocity_x ** 2 + landing_trajectory.iris_angular_velocity_y ** 2 + landing_trajectory.iris_angular_velocity_z ** 2)
    landing_trajectory["camera_angular_velocity_magnitude"] = numpy.sqrt(landing_trajectory.camera_angular_velocity_x ** 2 + landing_trajectory.camera_angular_velocity_y ** 2 + landing_trajectory.camera_angular_velocity_z ** 2)
    # print(type(landing_trajectory["camera_angular_velocity_magnitude"].to_frame().diff(axis=1)))
    landing_trajectory["camera_angular_acceleration_magnitude"] = numpy.abs(landing_trajectory["camera_angular_velocity_magnitude"].to_frame().diff())
    landing_trajectory["camera_angular_jerk_magnitude"] = numpy.abs(landing_trajectory["camera_angular_acceleration_magnitude"].to_frame().diff())

    apriltag_pose_estimate_axes.scatter(landing_trajectory["apriltag_estimate_position_x"], landing_trajectory["apriltag_estimate_position_y"], landing_trajectory["apriltag_estimate_position_z"])
    apriltag_estimate_error_axes.scatter(landing_trajectory["planar_distance_estimate_to_apriltag_marker"], landing_trajectory["apriltag_estimate_error"])
    apriltag_error_vs_angular_velocity_axes.scatter(landing_trajectory["iris_angular_velocity_magnitude"], landing_trajectory["apriltag_estimate_error"])
    apriltag_error_vs_camera_angular_velocity_axes.scatter(landing_trajectory["camera_angular_velocity_magnitude"], landing_trajectory["apriltag_estimate_error"])
    apriltag_error_vs_camera_angular_acceleration_axes.scatter(landing_trajectory["camera_angular_acceleration_magnitude"], landing_trajectory["apriltag_estimate_error"])
    # apriltag_error_vs_camera_jerk_acceleration_axes.scatter(landing_trajectory["camera_angular_jerk_magnitude"], landing_trajectory["apriltag_estimate_error"])
    temp_ax.scatter(landing_trajectory["iris_linear_velocity_x"], landing_trajectory["apriltag_estimate_error"])
    temp_ax.set_title("Iris linear velocity (E) vs apriltag pose estimate error")

    num_readings += len(landing_trajectory)
    all_data = pandas.concat([all_data, landing_trajectory])

print(all_data.head())

print("pose estimate average:\n%0.4f & %0.4f & %0.4f" % (numpy.nanmean(all_data["apriltag_estimate_position_x"]), numpy.nanmean(all_data["apriltag_estimate_position_y"]), numpy.nanmean(all_data["apriltag_estimate_position_z"])) )
print("pose estimate var:\n%0.4f & %0.4f & %0.4f" % (numpy.nanvar(all_data["apriltag_estimate_position_x"]), numpy.nanvar(all_data["apriltag_estimate_position_y"]), numpy.nanvar(all_data["apriltag_estimate_position_z"])) )
print("pose estimate std:\n%0.4f & %0.4f & %0.4f" % (numpy.sqrt(numpy.nanvar(all_data["apriltag_estimate_position_x"])), numpy.sqrt(numpy.nanvar(all_data["apriltag_estimate_position_y"])), numpy.sqrt(numpy.nanvar(all_data["apriltag_estimate_position_z"]))) )

print("number of pose estimates: %s" % len(all_data))
# print("number of outliers: %s" % len())

apriltag_pose_estimate_axes.set_title("April Tag Pose Estimation")
apriltag_pose_estimate_axes.set_xlabel("x (east)")
apriltag_pose_estimate_axes.set_ylabel("y (north)")
apriltag_pose_estimate_axes.set_zlabel("z (up)")
# apriltag_pose_estimate_axes.set_xlim(-10, 10)
# apriltag_pose_estimate_axes.set_ylim(40, 60)
# apriltag_pose_estimate_axes.set_zlim(-10, 10)

apriltag_estimate_error_axes.set_title("April Tag plane distance vs. pose estimate error magnitude")
apriltag_estimate_error_axes.set_xlabel("Relative distance (m)")
apriltag_estimate_error_axes.set_xlim(0, 20)
apriltag_estimate_error_axes.set_ylim(0, 3)
apriltag_estimate_error_axes.set_ylabel("Pose estimate error magnitude (m)")
apriltag_error_vs_angular_velocity_axes.set_title("Iris angular velocity vs April Tag pose estimate error")
apriltag_error_vs_camera_angular_velocity_axes.set_title("Camera angular velocity vs April Tag pose estimate error")
apriltag_error_vs_camera_angular_acceleration_axes.set_title("Camera angular acceleration vs April Tag pose estimate error")

plt.show()