"""
.. module: FSRStools.profilometer
   :platform: Windows
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu>

Load, display and convert SDF files from ADE profilometers. Supports extraction of film height using histogram analysis.

..
   This file is part of the FSRStools python module.

   The FSRStools python module is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The FSRStools python module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the FSRStools python module. If not, see <http://www.gnu.org/licenses/>.

   Copyright 2015 Daniel Dietze <daniel.dietze@berkeley.edu>.
"""
import numpy as np
import scipy.optimize as sp
import pylab as pl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import glob
import os


# get value associated with key from txt
# support function for load_sdf
def _get_value(txt, key):
    i1 = txt.find(key)
    if(i1 == -1):
        return 0.0

    i2 = txt.find("=", i1)
    i3 = txt.find("\n", i2)

    try:
        val = float(txt[i2+1:i3].strip())
        return val
    except:
        print("conversion error")
        return 0.0

# #########################################################################################################
# load / save data
def load_sdf(filename):
    """Loads an SDF file.

    :param str filename: Filename of the file to load.
    :returns: - X coordinate in meters (2d array)
              - Y coordinate in meters (2d array)
              - Z coordinate in meters (2d array)
    """
    # read entire file into string object
    print("reading file ", filename)
    fp = open(filename, "r")
    txt = fp.read()
    fp.close()
    print(" ..done")

    # get dimensions and scaling
    print("get image dimensions")
    x = int(_get_value(txt, "NumPoints"))
    y = int(_get_value(txt, "NumProfiles"))
    dx = _get_value(txt, "Xscale")
    dy = _get_value(txt, "Yscale")
    dz = _get_value(txt, "Zscale")
    print(" ..done: x = %d, y = %d" % (x, y))

    # get height data
    print("extract image data")
    i1 = txt.find("*")
    i2 = txt.find("*", i1+1)
    data = txt[i1+1:i2].replace('\n', '   ').strip().split()
    data = np.array(data, dtype=str).astype(dtype=float)
    data = data.reshape((y, x))
    data = data * dz
    print(" ..done")

    X = np.arange(x) * dx
    Y = np.arange(y) * dy
    X, Y = np.meshgrid(X, Y)

    # when returning cut off the first and last row along x and y as the
    # profilometer seems to store some bogus data there
    return Y[1:-1, 1:-1], X[1:-1, 1:-1], data[1:-1, 1:-1]

def save_xyz(filename, x, y, z):
    """Save height profile as simple XYZ ascii file.

    :param str filename: Destination file name.
    :param 2d_array x: X coordinates.
    :param 2d_array y: Y coordinates.
    :param 2d_array z: Z coordinates.
    """
    xo = x.ravel()
    yo = y.ravel()
    zo = z.ravel()
    np.savetxt(filename, np.array([xo, yo, zo]).transpose())

def convert(filename):
    """Convert files from sdf to xyz format.
    Saves the converted file using the same filename and ".xyz" as file ending.

    :param str filename: Filename to convert. Uses :py:mod:`glob` to support wildcards and regular expressions.
    """
    files = glob.glob(filename)
    for f in files:
        o = os.path.splitext(f)[0] + ".xyz"
        x, y, data = load_sdf(f)
        save_xyz(o, x, y, data)

# #########################################################################################################
# Thin film thickness extraction
def get_film_thickness(z, plotHist = False):
    """Extract film thickness from height profile using histogram analysis and two Gaussians.

    :param 2d_array z: Z coordinates / height profile.
    :param bool plotHist: If *True*, the histogram and fit results are plotted in a new figure (optional).
    :returns: Extracted height of the thin film along with its uncertainty in meters (*float*, *float*).
    """
    # get histogram and bin centers
    hist, edges = np.histogram(z, bins = 100, density=False)
    hist = hist[:-1] / np.amax(hist)        # remove invalid data
    edges = edges[:-1] * 1e6
    centers = ((edges + np.roll(edges, -1)) * 0.5)[:-1]

    # assume two dominant peaks corresponding to the lower and upper surface (substrate and film)
    # split data in two parts, look for maxima and perform fitting with two gaussians
    g = lambda x, A1, x01, dx1, A2, x02, dx2: A1 * np.exp(-(x - x01)**2 / (2*dx1**2)) + A2 * np.exp(-(x - x02)**2 / (2*dx2**2))

    # find approximate center between the two peaks using the second derivative and its center of mass
    tmp = np.absolute(np.diff(hist, 2))
    hp = int(np.sum(tmp * np.arange(len(tmp))) / np.sum(tmp))

    # fit with two gaussians
    popt = [np.amax(hist[:hp]), centers[np.argmax(hist[:hp])], (centers[-1]-centers[0])/50.0, np.amax(hist[hp:]), centers[hp + np.argmax(hist[hp:])], (centers[-1]-centers[0])/50.0]
    popt, _ = sp.curve_fit(g, centers, hist, popt, maxfev=10000)

    # get thickness and stddev
    d = abs(popt[1] - popt[4]) * 1e-6
    dd = np.sqrt( popt[2]**2 + popt[5]**2 ) * 1e-6

    # if we were not able to find two distinct peaks in the first attempt, use a different method to find those peaks
    # based on the original histogram and its center of mass
    if d / 2 <= dd:
        hp = int(np.sum(hist * np.arange(len(hist))) / np.sum(hist))

        popt = [np.amax(hist[:hp]), centers[np.argmax(hist[:hp])], (centers[-1]-centers[0])/50.0, np.amax(hist[hp:]), centers[hp + np.argmax(hist[hp:])], (centers[-1]-centers[0])/50.0]
        popt, _ = sp.curve_fit(g, centers, hist, popt, maxfev=10000)

        d = abs(popt[1] - popt[4]) * 1e-6
        dd = np.sqrt( popt[2]**2 + popt[5]**2 ) * 1e-6

    # if we want plotting, plot..
    if plotHist:
        pl.figure()
        x = np.linspace(edges[0], edges[-2], 256)
        pl.bar(edges[:-1], hist, width=(edges[1]-edges[0]), color="r")
        pl.plot(x, g(x, *popt), "b-")
        pl.xlabel("Z Position ($\mu$m)")
        pl.ylabel("Histogram (norm)")

    return d, dd

# #########################################################################################################
# Tilt correction
def _a_plane(XY, z0, dzx, dzy):
    """Helper function for :py:func:`tilt_correction`.

    :param (array, array) XY: Tuple of x and y coordinates generated by :py:func:`np.meshgrid`.
    :param float z0: Height offset.
    :param float dzx: Slope along x-direction.
    :param float dzy: Slope along y-direction.
    :returns: Z coordinates / height profile of a plane.
    """
    x, y = XY
    return z0 + x * dzx + y * dzy

def _errorfunction(p, XY, zdata, type = 'mean', threshold = 0):
    """Helper function for :py:func:`tilt_correction`.

    :param (array) p: Set of parameters (z0, dzx, dzy) for plane construction.
    :param (array, array) XY: Tuple of x and y coordinates generated by :py:func:`np.meshgrid`.
    :param (2d_array) zdata: Experimental height profile.
    :param str type: Filtering of data:

                                        - *lower*: choose only height values below *threshold*.
                                        - *upper*: choose only height values above *threshold*.
                                        - *mean*: use all height values.
    :returns: Difference between calculated plane and experimental data.
    """
    out = (zdata - _a_plane(XY, *p)).ravel()
    if type == "lower":
        return np.compress(zdata.ravel() < threshold, out)
    if type == "upper":
        return np.compress(zdata.ravel() > threshold, out)
    return out

def tilt_correction(z, type = "lower"):
    """Correct tilt by subtracting a plane from the 2d height profile.

    :param 2d_array z: Z coordinates / height profile.
    :param str type: Type of subtraction:

                     - *lower*: apply correction to lower plane data
                     - *upper*: apply correction to upper plane data
                     - *mean*: apply correction to all data
    :returns: corrected height profile.
    """
    # generate x and y coordinate arrays
    x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))

    # determine threshold value
    # find approximate center between the two peaks using the second derivative and its center of mass
    if type == 'upper' or type == 'lower':
        hist, edges = np.histogram(z, bins = 256, density=False)
        tmp = np.absolute(np.diff(hist, 2))
        hp = int(np.sum(tmp * np.arange(len(tmp))) / np.sum(tmp))
        threshold = edges[hp]
    else:
        threshold = 0.0

    # fit with a plane
    popt = sp.leastsq(_errorfunction, [0, 1, 1], args=((x, y), z, type, threshold))[0]

    # subtract plane and return
    return z - _a_plane((x, y), *popt)

# #########################################################################################################
# Fancy plotting of 3d surfaces.
def plot_profile(x, y, z, filename = None, *args):
    """Create a 3d surface plot.

    :param 2d_array x: Array of x-coordinates.
    :param 2d_array y: Array of y-coordinates.
    :param 2d_array z: Array of z-coordinates / height profile.
    :param str filename: Save figure to this filename if not None.
    :param mixed args: List of arguments that should be passed to the plotting function.
    """
    # create a 3d figure
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot
    ax.plot_surface(x * 1e6, y * 1e6, z * 1e9, cmap=cm.gist_heat, antialias=True, *args)

    # labels
    ax.set_xlabel("X-Position ($\mu$m)")
    ax.set_ylabel("Y-Position ($\mu$m)")
    ax.set_zlabel("Height (nm)")

    if filename is not None:
        pl.savefig(filename)
