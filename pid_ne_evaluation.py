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
log_directory = "/home/joshua/Documents/pid_testing/*"
filenames = glob.glob(log_directory)

filename_format = "test_-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f,-%0.2f_%s_True.csv"

plots = {}
times = {}
energies = {}

for filename in filenames:

    all_items = filename.split("_")
    gains = all_items[2].split(",")

    success = eval(all_items[-1].split(".")[0])

    data = pandas.read_csv(filename)

    start_time = data["time"].iloc[0]
    data["d_t"] = data["time"] - start_time
    start_energy = data["energy_consumed"].iloc[0]
    data["d_e"] = data["energy_consumed"] - start_energy

    key = all_items[2]
    if( key not in plots.keys() ):
        plots[key] = plt.figure(figsize=(7,4))
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        for i in range(len(gains)):
            gains[i] = float(gains[i])
        # text = "$kp_3=%0.1f, ki_3=%0.1f, kd_3=%0.1f$\n$kp_2=%0.1f, ki_2=%0.1f, kd_2=%0.1f$\n$kp_1=%0.1f, ki_1=%0.1f, kd_1=%0.1f$" % (gains[0], gains[1], gains[2], gains[3], gains[4], gains[5], gains[6], gains[7], gains[8])

        # text = "$kp_3=%0.1f, ki_3=%0.1f, kd_3=%0.1f$\n$kp_2=%0.1f, ki_2=%0.1f, kd_2=%0.1f$\n$kp_1=%0.1f, ki_1=%0.1f, kd_1=%0.1f$" % (
        # gains[0], gains[1], gains[2], gains[3], gains[4], gains[5], gains[6], gains[7], gains[8])

        text2 = r'''\begin{tabular}{|c|c|c|c|}\hline Phase & $k_p$ & $k_i$ & $k_d$ \\\hline %s & %0.1f & %0.1f & %0.1f \\\hline %s & %0.1f & %0.1f & %0.1f \\\hline %s & %0.1f & %0.1f & %0.1f \\\hline\end{tabular}''' % (1, gains[0], gains[1], gains[2], 2, gains[3], gains[4], gains[5], 3, gains[6], gains[7], gains[8])
        plt.text(0, 13, text2, size=10)

        # text = r"\begin{tabular}{ c | c | c } 3 & 1 & 2 \\\hline\end{tabular}"
        # plt.figtext(0.3, 0.65, text, wrap=False, horizontalalignment='center', fontsize=10)
        # plt.table(cellText=gains)
        plt.xlabel("Time before Approach End (s)")
        plt.ylabel("Planar Distance (m)")

        times[key] = 0
        energies[key] = 0

    plots[key].gca().plot(data["d_t"], data["plane_displacement"].iloc[::-1])
    times[key] += data["d_t"].iloc[-1]
    energies[key] += data["d_e"].iloc[-1]
    # break
#
# plt.show()

# for key in plots:
#     print(key)

for key in plots:
    plots[key].savefig("/home/joshua/Pictures/figures/pid_test_figures/" + key + ".png", bbox_inches="tight", pad_inches=0.05, transparent=False)

for key in times:
    times[key] /= 5.0
    energies[key] /= 5.0

sorted_times = sorted(times.items(), key=itemgetter(1))
sorted_energies = sorted(energies.items(), key=itemgetter(1))

print(r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}")
print("\t\hline $k_{p1}$ & $k_{i1}$ & $k_{d1}$ & $k_{p2}$ & $k_{i2}$ & $k_{d2}$ & $k_{p3}$ & $k_{i3}$ & $k_{d3}$ & $t_{ave}$ (s) & $e_{ave}$ (hJ) \\\\\hline")
for i in range(5):
    key = sorted_times[i][0]
    print(key)
    gains = key.split(",")
    for ii in range(len(gains)):
        gains[ii] = float(gains[ii])
    print("\t%0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f \\\\\hline" % (gains[0], gains[1], gains[2], gains[3], gains[4], gains[5], gains[6], gains[7], gains[8], sorted_times[i][1], energies[key]))
print("\end{tabular}")

# print(r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}\hline $k_{p1}$ & $k_{i1}$ & $k_{d1}$ & $k_{p2}$ & $k_{i2}$ & $k_{d2}$ & $k_{p3}$ & $k_{i3}$ & $k_{d3}$ & t_{ave} & e_{ave} \\\hline")
# for i in range(5):
#     gains = sorted_energies[i][0].split(",")
#     for i in range(len(gains)):
#         gains[i] = float(gains[i])
#     print("%0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f & %0.1f \\\\\hline" % (gains[0], gains[1], gains[2], gains[3], gains[4], gains[5], gains[6], gains[7], gains[8], sorted_energies[i][1]))
# print("\end{tabular}")

# text2 = r'''\begin{tabular}{|c|c|c|c|}\hline Phase & $k_p$ & $k_i$ & $k_d$ \\\hline %s & %0.1f & %0.1f & %0.1f \\\hline %s & %0.1f & %0.1f & %0.1f \\\hline %s & %0.1f & %0.1f & %0.1f \\\hline\end{tabular}''' % (1, gains[0], gains[1], gains[2], 2, gains[3], gains[4], gains[5], 3, gains[6], gains[7], gains[8])


# plt.show()

# kp3 = 0.7
# kd3 = 0.4
# ki3 = 0.0
# ki2 = 0.0
# ki1 = 0.0
# for kp2 in [0.3, 0.4, 0.5]:
#     for kp1 in [0.2, 0.3, 0.4]:
#         for kd2 in [0.6, 0.7, 0.8]:
#             for kd1 in [0.8, 0.9, 1.0]:
#
#                 fig = plt.figure()
#                 axes = fig.gca()
#
#                 for i in range(5):
