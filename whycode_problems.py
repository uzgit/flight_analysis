import pandas
import matplotlib.pyplot as plt
import glob
import numpy

# for plotting whycon.png
from pylab import *
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png

filename = "/home/joshua/programming/whycode_flight_data/data_135432000000.csv"
data = pandas.read_csv(filename)

# positional x/y figure
figure = plt.figure()
axes = figure.gca()
plt.title("Pose estimate sequence number vs. WhyCode positional displacement.")
plt.xlabel("Pose estimate number")
plt.ylabel("Displacement in camera frame (m)")
axes.plot(data["position_x"], label="Yaw displacement")
axes.plot(data["position_y"], label="Pitch displacement")
axes.legend()

# positional x/y figure constrained
figure = plt.figure()
axes = figure.gca()
plt.title("Pose estimate sequence number vs. WhyCode positional displacement.")
plt.xlabel("Pose estimate number")
plt.ylabel("Displacement in camera frame (m)")
axes.plot(data["position_x"].iloc[500:600], label="Yaw displacement")
axes.plot(data["position_y"].iloc[500:600], label="Pitch displacement")
axes.legend()

# orientation
figure = plt.figure()
axes = figure.gca()
plt.title("Pose estimate sequence number vs. WhyCode orientation.")
plt.xlabel("Pose estimate number")
plt.ylabel("Orientation quaternion in camera frame")
axes.plot(data["orientation_x"], label="X orientation")
axes.plot(data["orientation_y"], label="Y orientation")
axes.plot(data["orientation_z"], label="Z orientation")
axes.plot(data["orientation_w"], label="W orientation")
axes.legend()

# orientation constrained
figure = plt.figure()
axes = figure.gca()
plt.title("Pose estimate sequence number vs. WhyCode orientation.")
plt.xlabel("Pose estimate number")
plt.ylabel("Orientation quaternion in camera frame")
axes.plot(data["orientation_x"].iloc[500:600], label="X orientation")
axes.plot(data["orientation_y"].iloc[500:600], label="Y orientation")
axes.plot(data["orientation_z"].iloc[500:600], label="Z orientation")
axes.plot(data["orientation_w"].iloc[500:600], label="W orientation")
axes.legend()

# orientation constrained
figure = plt.figure()
axes = figure.gca()
plt.title("Pose estimate sequence number vs. WhyCode orientation.")
plt.xlabel("Pose estimate number")
plt.ylabel("Orientation quaternion in camera frame")
axes.plot(data["orientation_x"].iloc[1775:1875], label="X orientation")
axes.plot(data["orientation_y"].iloc[1775:1875], label="Y orientation")
axes.plot(data["orientation_z"].iloc[1775:1875], label="Z orientation")
axes.plot(data["orientation_w"].iloc[1775:1875], label="W orientation")
axes.legend()

plt.show()