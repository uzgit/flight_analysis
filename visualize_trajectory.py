import pandas
import matplotlib.pyplot as plt

# import flight data
filename = "/home/joshua/Documents/flight_data/flight_data_17_04_2020_23_20_59.csv"
flight_data = pandas.read_csv(filename)

landing_trajectory = flight_data.where(flight_data["landing_pad_detected"] == 1)

print(landing_trajectory)

# create 3d plot
figure = plt.figure()
axes = figure.gca(projection="3d")

# plot 3d trajectory
axes.plot(landing_trajectory["iris_position_x"], landing_trajectory["iris_position_y"], landing_trajectory["iris_position_z"])

# p = Circle((5, 5), 3)
# axes.add_patch(p)

plt.show()