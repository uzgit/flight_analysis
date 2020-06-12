import pandas
import matplotlib.pyplot as plt
import numpy
import glob
import pyquaternion
import statsmodels.formula.api as smf

from scipy.stats import linregress

def func(x, a, b, c):
    return a * numpy.exp(b * x) + c

pandas.set_option('display.max_columns', None)

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

# slope, intercept, r_value, p_value, std_err = linregress(all_data["true_displacement"], numpy.log(all_data["pose_estimation_error"]))

# x_predicted = numpy.arange(0, 20, 0.1)
# y_predicted = numpy.exp(x_predicted * slope + intercept)
# regression = plots[error].plot(x_predicted, y_predicted, linestyle="--", label=r"$y = e^{%0.3fx - %0.3f}$" % ( slope, numpy.abs(intercept) ) )
# plt.legend()

# coefficients = numpy.polyfit(all_data["true_displacement"], numpy.log(all_data["pose_estimation_error"]), 1)
# x_predicted = numpy.arange(0, 20, 0.1)
# y_predicted = numpy.exp(coefficients[1]) * numpy.exp(coefficients[0] * x_predicted)
# plots[error].plot(x_predicted, y_predicted)

# dataframe = pandas.DataFrame(columns=['y', 'x'])
# dataframe["x"] =

degree = 3
weights = numpy.polyfit(all_data["true_displacement"], all_data["pose_estimation_error"], degree)
model = numpy.poly1d(weights)
# print(model.coeffs)
results = smf.ols(formula="pose_estimation_error ~ model(true_displacement)", data=all_data[["true_displacement", "pose_estimation_error"]]).fit()

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

x = numpy.arange(0, 20, 0.1)
plots[error].plot(x, model(x), label="")
plots[error].set_title("Distance to April Tag Marker Versus Pose Estimation Error")
# plots[error].legend()

print(str_label)

# print(dir(results))
# print(type(results.bse))
# print(results.bse["model(true_displacement)"])
print(results.summary())

# print("n_orig: %s" % len())
print("n: %s" % len(all_data))

# print("x: %0.3f & %0.3f" % (numpy.nanmean(all_data["apriltag_estimated_position_x"]), numpy.nanstd(all_data["apriltag_estimated_position_x"])))
# print("y: %0.3f & %0.3f" % (numpy.nanmean(all_data["apriltag_estimated_position_y"]), numpy.nanstd(all_data["apriltag_estimated_position_y"])))
# print("z: %0.3f & %0.3f" % (numpy.nanmean(all_data["apriltag_estimated_position_z"]), numpy.nanstd(all_data["apriltag_estimated_position_z"])))
# print("e^(x*%s + %s" % ( slope, intercept ))
# print("r^2: %s" % (r_value ** 2))
# print("std_err: %s" % ( std_err ) )
# popt, pcov = curve_fit(func, all_data["true_displacement"], numpy.log(all_data["pose_estimation_error"]), maxfev=1000, bounds=(0, [3., 1., 0.5]))
# plots[error].plot(all_data["true_displacement"], func(all_data["true_displacement"], *popt))

plt.show()