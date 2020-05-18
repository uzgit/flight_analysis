import pandas
import matplotlib.pyplot as plt
import glob
import numpy
import matplotlib as mpl
from operator import itemgetter

# for plotting whycon.png
from pylab import *

simple_latex = {
    "text.latex.unicode" : True,
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": [],
    "font.sans-serif": ["Arial"],
    "font.monospace": [],
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.axisbelow": True,
    "axes.facecolor": "white",
    "axes.linewidth": 1.25,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "mathtext.fontset" : "stixsans",
    "mathtext.default": "regular",
    "pgf.preamble": [
        r"\usepackage{mathspec}",
        r"\setmathsfont(Digits,Latin,Greek){Arial}"
        ]
    }
mpl.rcParams.update(simple_latex)

# get all logs
log_directory = "/home/joshua/Documents/pid_u_testing/*"
filenames = glob.glob(log_directory)

filename_format = "test_-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f_%s_True.csv"

single_times = []
single_energies = []

plots = {}
times = {}
energies = {}

for filename in filenames:

    all_items = filename.split("_")
    key = all_items[3]
    print(all_items)
    gains = all_items[3].split(",")

    print(gains)

    success = eval(all_items[-1].split(".")[0])

    data = pandas.read_csv(filename)
    data = data.query("landing_phase != 0")

    start_time = data["time"].iloc[0]
    data["d_t"] = data["time"] - start_time
    start_energy = data["energy_consumed"].iloc[0]
    data["d_e"] = data["energy_consumed"] - start_energy

    if( key not in plots.keys() ):
        plots[key] = plt.figure(figsize=(7,4))
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        for i in range(len(gains)):
            gains[i] = float(gains[i])
        # text = "$kp_3=%0.1f, ki_3=%0.1f, kd_3=%0.1f$\n$kp_2=%0.1f, ki_2=%0.1f, kd_2=%0.1f$\n$kp_1=%0.1f, ki_1=%0.1f, kd_1=%0.1f$" % (gains[0], gains[1], gains[2], gains[3], gains[4], gains[5], gains[6], gains[7], gains[8])

        # text = "$kp_3=%0.1f, ki_3=%0.1f, kd_3=%0.1f$\n$kp_2=%0.1f, ki_2=%0.1f, kd_2=%0.1f$\n$kp_1=%0.1f, ki_1=%0.1f, kd_1=%0.1f$" % (
        # gains[0], gains[1], gains[2], gains[3], gains[4], gains[5], gains[6], gains[7], gains[8])

        text2 = r'''\begin{tabular}{|c|c|c|}\hline $k_p$ & $k_i$ & $k_d$ \\\hline %0.1f & %0.1f & %0.1f \\\hline\end{tabular}''' % (gains[0], gains[1], gains[2])
        plt.text(8, -0.3, text2, size=10)

        # text = r"\begin{tabular}{ c | c | c } 3 & 1 & 2 \\\hline\end{tabular}"
        # plt.figtext(0.3, 0.65, text, wrap=False, horizontalalignment='center', fontsize=10)
        # plt.table(cellText=gains)
        plt.xlabel("Time before Landing (s)")
        plt.ylabel("Vertical Velocity (m)")
        # plt.legend()

        times[key] = 0
        energies[key] = 0

    # plots[key].gca().plot(data["d_t"], data["iris_position_z"].iloc[::-1])
    plots[key].gca().plot(data["d_t"], data["iris_linear_velocity_z"].iloc[::-1])
    times[key] += data["d_t"].iloc[-1]
    energies[key] += data["d_e"].iloc[-1]

    single_time = float(data["d_t"].iloc[-1])
    single_energy = float(data["d_e"].iloc[-1])
    if( single_time < 50 and single_energy < 120 ):
        single_times.append(single_time)
        single_energies.append(single_energy)
    # break

# plt.show()

# for key in plots:
#     print(key)

for key in plots:
    print("Saving " + key)
    plots[key].savefig("/home/joshua/Pictures/figures/pid_test_figures_u_velocity/" + key + ".png", bbox_inches="tight", pad_inches=0.05, transparent=False)

for key in times:
    times[key] /= 5.0
    energies[key] /= 5.0

sorted_times = sorted(times.items(), key=itemgetter(1))
sorted_energies = sorted(energies.items(), key=itemgetter(1))

print(r"\begin{tabular}{|c|c|c|c|c|}")
print("\t\hline $k_{p}$ & $k_{i}$ & $k_{d}$ & $t_{ave}$ (s) & $e_{ave}$ (hJ) \\\\\hline")
for i in range(6):
    key = sorted_times[i][0]
    # print(key)
    gains = key.split(",")
    for ii in range(len(gains)):
        gains[ii] = float(gains[ii])
    print("\t%0.1f & %0.1f & %0.1f & %0.1f & %0.1f \\\\\hline" % (gains[0], gains[1], gains[2], sorted_times[i][1], energies[key]))
print("\end{tabular}")

print(sorted_times)
print(sorted_energies)

print("Times (n=%s): %0.2f +- %0.2f" % ( len(single_times), numpy.average(single_times), numpy.std(single_times) ))
print("Energies (n=%s): %0.2f +- %0.2f" % ( len(single_energies), numpy.average(single_energies), numpy.std(single_energies) ))