import matplotlib.pyplot as plt
import numpy

figure = plt.figure(figsize=(15, 11))
axes = figure.gca(projection="3d")

def descent_radius(x, y):

    return numpy.log( numpy.sqrt(x**2 + y**2) / 0.15 ) / 0.4

# x = numpy.arange(0.00001, 20, 0.1)
# y = numpy.arange(0.00001, 20, 0.1)
#
# X, Y = numpy.meshgrid(x, y)
#
# z = descent_radius(x, y)
#
# axes.contour3D(X, Y, z)

def f(x, y):
    return numpy.sin(numpy.sqrt(x ** 2 + y ** 2))

x = numpy.linspace(-6, 6, 30)
y = numpy.linspace(-6, 6, 30)

X, Y = numpy.meshgrid(x, y)
Z = descent_radius(X, Y)

axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

axes.set_title("Descent Region")
axes.set_xlabel("North")
axes.set_ylabel("East")
axes.set_zlabel("Up")
axes.view_init(elev=30, azim=20)

plt.savefig("/home/joshua/Pictures/figures/descent_region.png", bbox_inches="tight", pad_inches=-0.2, transparent=True)
# plt.show()