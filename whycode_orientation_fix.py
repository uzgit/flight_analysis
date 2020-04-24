import pandas
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

filename = "/home/joshua/programming/whycode_flight_data/data_135432000000.csv"
data = pandas.read_csv(filename)

data["inverse_orientation_w"] = ""
data["inverse_orientation_x"] = ""
data["inverse_orientation_y"] = ""
data["inverse_orientation_z"] = ""
data["is_flipped"] = ""

# set default orientation to be normal
previous_orientation = Quaternion(1, 0, 0, 0)
for index, row in data.iterrows():

    # create objects for orientation and inverse orientation
    orientation = Quaternion(row["orientation_w"], row["orientation_x"], row["orientation_y"], row["orientation_z"])
    inverse_orientation = orientation.inverse

    data.at[index, "inverse_orientation_w"] = inverse_orientation.w
    data.at[index, "inverse_orientation_x"] = inverse_orientation.x
    data.at[index, "inverse_orientation_y"] = inverse_orientation.y
    data.at[index, "inverse_orientation_z"] = inverse_orientation.z

    # determine if the orientation is assumed to be flipped
    data.at[index, "is_flipped"] = Quaternion.distance(inverse_orientation, previous_orientation) < 0.05

    # set iteration values
    if(data.at[index, "is_flipped"]):
        orientation = inverse_orientation
    previous_orientation = orientation

    data.at[index, "corrected_orientation_w"] = orientation.w
    data.at[index, "corrected_orientation_x"] = orientation.x
    data.at[index, "corrected_orientation_y"] = orientation.y
    data.at[index, "corrected_orientation_z"] = orientation.z

filename_elements = filename.split(".")
new_filename = filename_elements[0] + "_fixed.csv"
data.to_csv(new_filename)

print("Saved to: %s" % new_filename)

# orientation constrained
figure = plt.figure()
axes = figure.gca()
plt.title("Pose estimate sequence number vs. WhyCode orientation.")
plt.xlabel("Pose estimate number")
plt.ylabel("Orientation quaternion in camera frame")
axes.plot(data["orientation_x"].iloc[500:600], label="X")
axes.plot(data["orientation_y"].iloc[500:600], label="Y")
axes.plot(data["orientation_z"].iloc[500:600], label="Z")
axes.plot(data["orientation_w"].iloc[500:600], label="W")
axes.legend()

# orientation constrained
figure = plt.figure()
axes = figure.gca()
plt.title("Pose estimate sequence number vs. corrected WhyCode orientation.")
plt.xlabel("Pose estimate number")
plt.ylabel("Orientation quaternion in camera frame")
axes.plot(data["corrected_orientation_x"].iloc[500:600], label="X")
axes.plot(data["corrected_orientation_y"].iloc[500:600], label="Y")
axes.plot(data["corrected_orientation_z"].iloc[500:600], label="Z")
axes.plot(data["corrected_orientation_w"].iloc[500:600], label="W")
axes.legend()

plt.show()