import pandas
import matplotlib.pyplot as plt
import numpy
import glob
from squaternion import Quaternion

import statsmodels.formula.api as smf

from scipy.stats import linregress

from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * numpy.exp(b * x) + c

pandas.set_option('display.max_columns', None)

def get_yaw(data):
    # q = Quaternion(data["landing_pad_apriltag_global_pose_rotation_w"], data["landing_pad_apriltag_global_pose_rotation_x"], data["landing_pad_apriltag_global_pose_rotation_y"], data["landing_pad_apriltag_global_pose_rotation_z"])
    # eulers = q.to_euler()
    # yaw = eulers[2]

    data["yaw"] = 20

    # return yaw

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# flight_data_19_04_2020_16_08_42.csv
log_directory = "/home/joshua/Documents/gimbal_and_pose_estimation_testing/"# (orig with oscillation)/"
filenames = glob.glob(log_directory + "*.csv")

x = 0
y = 1
z = 2
error = 3

apriltag_3d = plt.figure().gca(projection="3d")

plots = []
for i in range(4):
    figure = plt.figure(figsize=(7,4))
    plots.append(figure.gca())

n_orig = 0
n = 0

all_data = pandas.DataFrame()

for filename in filenames:

    flight_data = pandas.read_csv(filename)

    # start_time = flight_data["time"].iloc[0]
    # flight_data["d_t"] = flight_data["time"] - start_time

    flight_data["landing_pad_apriltag_global_pose_position_x_diff"] = flight_data["landing_pad_apriltag_global_pose_position_x"].diff()

    flight_data = flight_data.query("apriltag_detected and (landing_phase > 0 or landing_phase < 5) and landing_pad_apriltag_global_pose_position_x_diff != 0")

    # flight_data.to_csv("/home/joshua/test.csv")

    # print(flight_data.tail())

    # plots[x].plot(flight_data["d_t"], flight_data["landing_pad_apriltag_global_pose_position_x"])
    # plots[y].plot(flight_data["d_t"], flight_data["landing_pad_apriltag_global_pose_position_y"])
    # plots[z].plot(flight_data["d_t"], flight_data["landing_pad_apriltag_global_pose_position_z"])

    flight_data["pose_estimation_error"] = numpy.sqrt( (flight_data["iris_position_x"] + flight_data["landing_pad_apriltag_global_pose_position_y"] - 30) ** 2 +
                                        (flight_data["iris_position_y"] + flight_data["landing_pad_apriltag_global_pose_position_x"]) ** 2 +
                                        (flight_data["iris_position_z"] + flight_data["landing_pad_apriltag_global_pose_position_z"]) ** 2)
    flight_data["true_plane_displacement"] = numpy.sqrt((flight_data["iris_position_x"] - flight_data["landing_pad_position_x"]) ** 2 +
                                                        (flight_data["iris_position_y"] - flight_data["landing_pad_position_y"]) ** 2)

    flight_data["gimbal_x_velocity"] = flight_data["gimbal_x_position"].diff()
    flight_data["gimbal_y_velocity"] = flight_data["gimbal_y_position"].diff()
    flight_data["gimbal_velocity"] = numpy.sqrt( flight_data["gimbal_x_velocity"] ** 2 + flight_data["gimbal_y_velocity"] ** 2 )
    flight_data["gimbal_acceleration"] = flight_data["gimbal_velocity"].diff()

    n_orig += len(flight_data)

    # flight_data = flight_data.query("pose_estimation_error < 10")


    # plots[error].scatter(flight_data["iris_position_z"], flight_data["pose_estimation_error"])
    # plots[error].scatter(flight_data["true_plane_displacement"], flight_data["pose_estimation_error"], s=5)


    # plots[error].scatter(flight_data["gimbal_y_velocity"], flight_data["pose_estimation_error"])

    n += len(flight_data)

    # plots[error].plot(flight_data["d_t"], flight_data["pose_estimation_error"])

    flight_data["true_displacement"] = numpy.sqrt(
        (flight_data["iris_position_x"] - flight_data["landing_pad_position_x"]) ** 2 +
        (flight_data["iris_position_y"] - flight_data["landing_pad_position_y"]) ** 2 +
        (flight_data["iris_position_z"] + flight_data["landing_pad_position_z"]) ** 2)

    flight_data["apriltag_estimated_position_x"] = flight_data["iris_position_x"] + flight_data["landing_pad_apriltag_global_pose_position_y"]
    flight_data["apriltag_estimated_position_y"] = flight_data["iris_position_y"] - flight_data["landing_pad_apriltag_global_pose_position_x"]
    flight_data["apriltag_estimated_position_z"] = flight_data["iris_position_z"] + flight_data["landing_pad_apriltag_global_pose_position_z"]

    # apriltag_3d.scatter(flight_data["iris_position_x"] + flight_data["landing_pad_apriltag_global_pose_position_y"],
    #                flight_data["iris_position_y"] - flight_data["landing_pad_apriltag_global_pose_position_x"],
    #                flight_data["iris_position_z"] + flight_data["landing_pad_apriltag_global_pose_position_z"],
    #                   s = 4)

    apriltag_3d.scatter(flight_data["apriltag_estimated_position_x"], flight_data["apriltag_estimated_position_y"], flight_data["apriltag_estimated_position_z"], s = 4)

    # plots[error].scatter(flight_data["true_displacement"], flight_data["pose_estimation_error"], s=5)
    plots[error].scatter(flight_data["true_displacement"], flight_data["pose_estimation_error"], s=5)

    # coefficients = numpy.polyfit(flight_data["true_displacement"], numpy.log(flight_data["pose_estimation_error"]), 1)
    # x_predicted = numpy.arange(0, 20, 0.1)
    # y_predicted = numpy.exp(coefficients[1]) * numpy.exp(coefficients[0] * x_predicted)
    # plots[error].plot(x_predicted, y_predicted)

    # print(len(flight_data["true_displacement"]))
    # print(len(flight_data["pose_estimation_error"]))

    all_data = all_data.append(flight_data, ignore_index=True)

    # print("x: %s, %s" % (numpy.mean(flight_data["apriltag_estimated_position_x"]), numpy.std(flight_data["apriltag_estimated_position_x"])))
    # print("y: %s, %s" % (numpy.mean(flight_data["apriltag_estimated_position_y"]), numpy.std(flight_data["apriltag_estimated_position_y"])))
    # print("z: %s, %s" % (numpy.mean(flight_data["apriltag_estimated_position_z"]), numpy.std(flight_data["apriltag_estimated_position_z"])))
    # break

apriltag_3d.set_xlabel("North")
apriltag_3d.set_ylabel("East")
apriltag_3d.set_zlabel("Up")
# apriltag_3d.set_xlim(20, 40)
# apriltag_3d.set_ylim(-10, 10)
# apriltag_3d.set_zlim(-10, 10)
apriltag_3d.set_xlim(10, 50)
apriltag_3d.set_ylim(-20, 20)
apriltag_3d.set_zlim(-20, 20)

plots[error].set_xlabel("Distance (m)")
plots[error].set_ylabel("Pose Estimation Error (m)")
plots[error].set_xlim(0, 12)
plots[error].set_ylim(0, 15)

coefficients = numpy.polyfit(all_data["true_displacement"], numpy.log(all_data["pose_estimation_error"]), 1)
x_predicted = numpy.arange(0, 20, 0.1)
y_predicted = numpy.exp(coefficients[1]) * numpy.exp(coefficients[0] * x_predicted)
# plots[error].plot(x_predicted, y_predicted)

# print("n_orig: %s" % len())
print("n: %s" % len(all_data))

print("x: %0.3f & %0.3f" % (numpy.nanmean(all_data["apriltag_estimated_position_x"]), numpy.nanstd(all_data["apriltag_estimated_position_x"])))
print("y: %0.3f & %0.3f" % (numpy.nanmean(all_data["apriltag_estimated_position_y"]), numpy.nanstd(all_data["apriltag_estimated_position_y"])))
print("z: %0.3f & %0.3f" % (numpy.nanmean(all_data["apriltag_estimated_position_z"]), numpy.nanstd(all_data["apriltag_estimated_position_z"])))

# popt, pcov = curve_fit(func, all_data["true_displacement"], numpy.log(all_data["pose_estimation_error"]), maxfev=1000, bounds=(0, [3., 1., 0.5]))
# plots[error].plot(all_data["true_displacement"], func(all_data["true_displacement"], *popt))

yaws = []
abs_yaws = []
for i in range(len(all_data)):
    yaw = Quaternion(all_data.iloc[i]["landing_pad_apriltag_global_pose_rotation_w"], all_data.iloc[i]["landing_pad_apriltag_global_pose_rotation_x"], all_data.iloc[i]["landing_pad_apriltag_global_pose_rotation_y"], all_data.iloc[i]["landing_pad_apriltag_global_pose_rotation_z"]).to_euler()[2]

    abs_yaws.append(yaw)

    if( yaw < 0 ):
        yaw += 2 * numpy.pi

    yaws.append(yaw)

# yaws = numpy.abs(yaws)

true_displacements = all_data["true_displacement"].to_numpy()
# print(true_displacements)
# print(yaws)

# print(len(true_displacements), len(yaws))
# print(len(yaws))
# print(len(all_data))

# all_data["yaw"] = "hi"
# all_data = all_data.apply(get_yaw, axis=1)
#
# print(all_data["yaw"])
yaw_figure = plt.figure(figsize=(7,4))
yaw_figure.gca().scatter(true_displacements, yaws, s=5)
yaw_figure.gca().set_xlim(0, 12)
yaw_figure.gca().set_ylim(numpy.pi - 0.5, numpy.pi + 0.5)
yaw_figure.gca().set_xlabel("Distance (m)")
yaw_figure.gca().set_ylabel("Estimated Yaw (rad)")

yaw_error_figure = plt.figure(figsize=(7,4)).gca()
yaw_errors = numpy.array(yaws) - numpy.pi
yaw_error_figure.scatter(true_displacements, numpy.abs(yaw_errors), s=2)
yaw_error_figure.set_xlim(0, 12)
yaw_error_figure.set_ylim(-0.5, 0.5)
yaw_error_figure.set_xlabel("Distance (m)")
yaw_error_figure.set_ylabel("Estimated Yaw Error (rad)")

# slope, intercept, r_value, p_value, std_err = linregress(true_displacements, numpy.abs(yaw_errors))

coeff, cov = numpy.polyfit(true_displacements, numpy.abs(yaw_errors), 1, cov=True)

print("cov: ", numpy.sqrt(numpy.diag(cov)))

x_predicted = numpy.arange(0, 20, 0.1)
y_predicted = numpy.polyval(coeff, x_predicted)
# y_predicted = intercept + slope * x_predicted

# print(x_predicted)
# print(y_predicted)


# print(coeff)
# yaw_error_figure.plot(x_predicted, y_predicted, linestyle="--", color="red", label=r"y=$%0.5f \cdot x + %0.3f$" % (coeff[0], coeff[1]) )
# print("std_err: %0.9f" % std_err)
# plt.legend()

# abs_yaw_figure

dataframe = pandas.DataFrame(data={"true_displacement" : true_displacements, "abs_yaw_error" : numpy.abs(yaw_errors)})

degree = 3
weights = numpy.polyfit(dataframe["true_displacement"], dataframe["abs_yaw_error"], degree)
model = numpy.poly1d(weights)
# print(model.coeffs)
results = smf.ols(formula="abs_yaw_error ~ model(true_displacement)", data=dataframe).fit()

str_label = r"$"
for i in range(len(model.coefficients)):

    coefficient = model.coeffs[i]
    exponent = len(model.coeffs) - i - 1

    if( i > 0 ):
        if( coefficient > 0 ):
            str_label += r"+"
        else:
            str_label += r"-"
    elif( coefficient < 0 ):
        str_label += r"-"

    if( exponent > 1 ):
        str_label += r"%0.3fx^%s" % ( numpy.abs(coefficient), exponent)
    elif( exponent == 1 ):
        str_label += r"%0.3fx" % (numpy.abs(coefficient))
    elif( exponent == 0 ):
        str_label += r"%0.3f" % (numpy.abs(coefficient))
str_label += r", \sigma=%0.4f" % (results.bse["model(true_displacement)"])
str_label += r"$"
print(str_label)
x = numpy.arange(0, 12, 0.1)
yaw_error_figure.plot(x, model(x), color="red", label="")
yaw_error_figure.set_title("Distance to April Tag Marker Versus Yaw Estimation Error")

print("n: %s" % len(yaws))
print(" %0.3f +- %0.3f" % (numpy.mean(yaws), numpy.std(yaws)))

plt.show()