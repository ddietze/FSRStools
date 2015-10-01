"""
.. module: FSRStools.yeh_tm
   :platform: Windows
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu>

Python implementation of Yeh's 4x4 transfer matrix formalism for arbitrary birefringent layers and arbitrary angles of incidence. The code follows Yeh's two seminal papers: Yeh, *J. Opt. Soc. Am.* **69**, 742 (1979) and Yeh, *Surf. Sci.* **96**, 41 (1980).

Layers are represented by the :py:class:`Layer` class that holds all parameters describing the optical properties of a single layer. The optical system is assembled using the :py:class:`System` class.

Example
-------

Reflection spectrum of an 8-layer high-reflector designed for 800nm::

    import numpy as np
    import pylab as pl
    import FSRStools.yeh_tm as tm
    import FSRStools.refind as rf

    # create wavelength and frequency axes in nm and 1/nm
    l = np.linspace(400, 1300, 256)
    w = 2 * np.pi / l

    # use the refrative index data from the FSRStools.refind module
    e1 = lambda x: rf.n_to_eps(rf.n_glass(2e-3 * np.pi / x, type = "SiO2"))
    e2 = lambda x: rf.n_to_eps(rf.n_coatings(2e-3 * np.pi / x, type = "MgF2_o"))
    e3 = lambda x: rf.n_to_eps(rf.n_coatings(2e-3 * np.pi / x, type = "ZnSe"))

    # calculate optimal thickness of coating layers for 800nm
    d2 = 800 / rf.n_coatings(0.8, type="MgF2_o") / 4.0
    d3 = 800 / rf.n_coatings(0.8, type="ZnSe") / 4.0

    # setup the optical system with vacuum as superstrate and substrate
    s = tm.System()

    # create a thick glass slide as support
    s.add_layer(tm.Layer(epsilon1 = e1, thick=True)) # substrate

    # calculate and plot the reflectance of the bare glass
    C = s.get_intensity_coefficients(w)[0] # rxx, rxy, ryx, ryy, txx, txy, tyx, tyy
    pl.plot(l, C)

    # add the HR coating: (HL)^n H glass
    s.add_layer(tm.Layer(thickness = d3, epsilon1 = e3)) # coating 1
    for i in range(8):
        s.add_layer(tm.Layer(thickness = d2, epsilon1 = e2)) # coating 2
        s.add_layer(tm.Layer(thickness = d3, epsilon1 = e3)) # coating 1

    # calculate and plot the reflectance of the HR coated glass
    C = s.get_intensity_coefficients(w)[0]
    pl.plot(l, C)

    # calculate and plot the reflectance of the HR coated glass for 45deg angle of incidence (p-pol)
    s.set_theta(45.0)
    C = s.get_intensity_coefficients(w)[0]
    pl.plot(l, C)

    pl.show()

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
import copy

# ************************************************************************************************************** #
# some helper functions


# compute the null space of matrix A which is the solution set x to the homogeneous equation Ax = 0
# see Wikipedia and http://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
# eps is relative distance to maximum entry
def null_space(A, eps=1e-4):
    """Compute the null space of matrix A which is the solution set x to the homogeneous equation Ax = 0.

    :param matrix A: Matrix.
    :param float eps: Maximum size of selected singular value relative to maximum.
    :returns: Null space (list of vectors) and associated singular components.

    .. seealso:: Wikipedia and http://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy.
    """
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps * np.amax(s), vh, axis=0)
    return null_space, np.compress(s <= eps * np.amax(s), s, axis=0)


# evanescent root - see Orfanidis book, Ch 7
def eroot(a):
    """Returns the evanescent root of a number, where the imaginary part has the physically correct sign.

    :param complex a: A number.
    :returns: Square root of a.
    """
    return np.where(np.real(a) < 0 and np.imag(a) == 0, -1j * np.sqrt(np.absolute(a)), np.lib.scimath.sqrt(a))


# normalize a to its length
def norm(a):
    """Normalize a vector to its length.

    :param vector a: A vector.
    :returns: Unit vector with same direction as a.
    """
    return a / np.sqrt(np.dot(a, np.conj(a)))     # use standard sqrt as argument is real and positive


# vacuum dielectric constant
def evacuum(x):
    """Returns the dielectric function of vacuum (1).

    :param array x: Wavelength / frequency axis.
    :returns: Array of ones with same shape as x.
    """
    try:
        return np.ones(len(x))
    except:
        return 1.0


# ************************************************************************************************** #
# layer class
class Layer:
    """Construct a layer class instance, which controls the physical properties of a single layer.

    :param float thickness: Thickness of layer in same units as wavelength.
    :param function epsilon1: Function that takes the frequency w as argument and returns a value for the dielectric constant along this axis. If no function if provided, defaults to vacuum.
    :param function epsilon2: Function that takes the frequency w as argument and returns a value for the dielectric constant along this axis. If no function if provided, defaults to epsilon1.
    :param function epsilon3: Function that takes the frequency w as argument and returns a value for the dielectric constant along this axis. If no function if provided, defaults to epsilon1.
    :param float theta: Euler angle :math:`\\theta`.
    :param float phi: Euler angle :math:`\\phi`.
    :param float psi: Euler angle :math:`\\psi`.
    :param bool thick: If True, use incoherent propagation by discarding the phase information across this layer.
    """

    # speed of light and magnetic permeability
    c = 1   #: speed of light in vacuum
    mu = 1  #: permeability of free space

    # constructor
    # thickness = thickness of layer in same units as wavelength / wavenumber
    # epsilon1..3 = functions that take the frequency w and return the value of the dielectric constant
    # along this axis; if no function is given, the value for epsilon1 is copied;
    # if no value is given for epsilon1, it is set to vacuum
    # theta, phi, psi = three Euler angles
    def __init__(self, thickness=1, epsilon1=None, epsilon2=None, epsilon3=None, theta=0, phi=0, psi=0, thick=False):

        # layer parameters
        self.epsilon = np.identity(3, dtype=np.complex128)
        self.g = np.zeros(4, dtype=np.complex128)
        self.p = np.zeros((4, 3), dtype=np.complex128)
        self.q = np.zeros((4, 3), dtype=np.complex128)
        self.D = np.zeros((4, 4), dtype=np.complex128)
        self.Di = np.zeros((4, 4), dtype=np.complex128)
        self.P = np.zeros((4, 4), dtype=np.complex128)
        self.T = np.zeros((4, 4), dtype=np.complex128)

        # layer thickness
        self.d = thickness

        # dielectric tensor in crystal frame
        if(epsilon1 != None):
            self.fepsilon1 = epsilon1
        else:
            self.fepsilon1 = evacuum
        self.fepsilon2 = epsilon2
        self.fepsilon3 = epsilon3
        if(self.fepsilon2 == None):
            self.fepsilon2 = self.fepsilon1
        if(self.fepsilon3 == None):
            self.fepsilon3 = self.fepsilon1

        # this variable tells us whether propagation through this layer should be handled coherently or not
        # thick: False = coherent; True = incoherent
        # the actual effect of this parameter is to change the calculation in the system class; it has no effect in the layer class
        self.thick = thick

        # euler matrix for rotation of dielectric tensor
        self.euler = np.identity(3, dtype=np.complex128)
        self.euler[0,0] = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi)
        self.euler[0,1] = -np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi)
        self.euler[0,2] = np.sin(theta) * np.sin(phi)
        self.euler[1,0] = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
        self.euler[1,1] = -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi)
        self.euler[1,2] = -np.sin(theta) * np.cos(phi)
        self.euler[2,0] = np.sin(theta) * np.sin(psi)
        self.euler[2,1] = np.sin(theta) * np.cos(psi)
        self.euler[2,2] = np.cos(theta)

    def calculate_epsilon(self, w):
        """Calculate the dielectric tensor for given frequency w. Stores the result in `self.epsilon`, which can be used for repeated access.

        :param float w: Frequency value.
        :returns: Dielectric tensor (4x4).
        """
        # get epsilon for given frequency in crystal frame
        if(self.fepsilon1 == None):
            epsilon0 = np.identity(3, dtype=np.complex128)
        else:
            epsilon0 = np.zeros((3,3), dtype=np.complex128)
            epsilon0[0,0] = self.fepsilon1(w)
            epsilon0[1,1] = self.fepsilon2(w)
            epsilon0[2,2] = self.fepsilon3(w)

        # rotate dielectric tensor to lab frame
        self.epsilon = np.dot(np.linalg.pinv(self.euler), np.dot(epsilon0, self.euler))
        return self.epsilon.copy()

    def calculate_g(self, w, a, b):
        """Calculate propagation constants g along z-axis for current layer.

        :param float w: Frequency value.
        :param complex a: In-plane propagation constant along x-axis.
        :param complex b: In-plane propagation constant along y-axis.
        :returns: Propagation constants along z-axis. The entries are sorted such that sign( Re(g) ) has the order `+,-,+,-`. However, the polarization (p or s) may not yet be correctly assigned. This is done using :py:func:`calculate_p_q`.
        """
        # set up the coefficients for the fourth-order polynomial that yields the z-propagation constant as the four roots
        # these terms are taken from my Maple calculation of the determinant of the matrix in Eq. 4 of Yeh's paper
        p = np.zeros(5, dtype=np.complex128)
        p[0] = w**2 * self.epsilon[2,2]
        p[1] = w**2 * self.epsilon[2,0] * a + b * w**2 * self.epsilon[2,1] + a * w**2 * self.epsilon[0,2] + w**2 * self.epsilon[1,2] * b
        p[2] = w**2 * self.epsilon[0,0] * a**2 + w**2 * self.epsilon[1,0] * b * a - w**4 * self.epsilon[0,0] * self.epsilon[2,2] + w**2 * self.epsilon[1,1] * b**2 + w**4 * self.epsilon[1,2] * self.epsilon[2,1] + b**2 * w**2 * self.epsilon[2,2] + w**4 * self.epsilon[2,0] * self.epsilon[0,2] + a**2 * w**2 * self.epsilon[2,2] + a * w**2 * self.epsilon[0,1] * b - w**4 * self.epsilon[1,1] * self.epsilon[2,2]
        p[3] =  - w**4 * self.epsilon[0,0] * self.epsilon[1,2] * b + w**2 * self.epsilon[2,0] * a * b**2 - w**4 * self.epsilon[0,0] * b * self.epsilon[2,1] - a * w**4 * self.epsilon[1,1] * self.epsilon[0,2] + w**4 * self.epsilon[1,0] * b * self.epsilon[0,2] + a**2 * b * w**2 * self.epsilon[1,2] + a**3 * w**2 * self.epsilon[0,2] + w**4 * self.epsilon[1,0] * self.epsilon[2,1] * a + b**3 * w**2 * self.epsilon[2,1] + a * b**2 * w**2 * self.epsilon[0,2] - w**4 * self.epsilon[2,0] * self.epsilon[1,1] * a + b**3 * w**2 * self.epsilon[1,2] + w**4 * self.epsilon[2,0] * self.epsilon[0,1] * b + w**2 * self.epsilon[2,0] * a**3 + a**2 * b * w**2 * self.epsilon[2,1] + a * w**4 * self.epsilon[0,1] * self.epsilon[1,2]
        p[4] = w**6 * self.epsilon[2,0] * self.epsilon[0,1] * self.epsilon[1,2] - w**6 * self.epsilon[2,0] * self.epsilon[1,1] * self.epsilon[0,2] + w**4 * self.epsilon[2,0] * a**2 * self.epsilon[0,2] + w**6 * self.epsilon[0,0] * self.epsilon[1,1] * self.epsilon[2,2] - w**4 * self.epsilon[0,0] * self.epsilon[1,1] * a**2 - w**4 * self.epsilon[0,0] * self.epsilon[1,1] * b**2 - w**4 * self.epsilon[0,0] * a**2 * self.epsilon[2,2] + w**2 * self.epsilon[0,0] * a**2 * b**2 - w**6 * self.epsilon[0,0] * self.epsilon[1,2] * self.epsilon[2,1] - b**2 * w**4 * self.epsilon[1,1] * self.epsilon[2,2] + b**2 * w**2 * self.epsilon[1,1] * a**2 + b**2 * w**4 * self.epsilon[1,2] * self.epsilon[2,1] + w**6 * self.epsilon[1,0] * self.epsilon[2,1] * self.epsilon[0,2] - w**6 * self.epsilon[1,0] * self.epsilon[0,1] * self.epsilon[2,2] + w**4 * self.epsilon[1,0] * self.epsilon[0,1] * a**2 + w**4 * self.epsilon[1,0] * self.epsilon[0,1] * b**2 + w**2 * self.epsilon[1,0] * a**3 * b + w**2 * self.epsilon[1,0] * a * b**3 + a**3 * b * w**2 * self.epsilon[0,1] + a * b**3 * w**2 * self.epsilon[0,1] - w**4 * self.epsilon[1,0] * a * b * self.epsilon[2,2] + a * b * w**4 * self.epsilon[2,1] * self.epsilon[0,2] - a * b * w**4 * self.epsilon[0,1] * self.epsilon[2,2] + w**4 * self.epsilon[2,0] * a * b * self.epsilon[1,2] + w**2 * self.epsilon[0,0] * a**4 + b**4 * w**2 * self.epsilon[1,1]

        # the four solutions for the g's are obtained by numerically solving the polynomial equation
        # these four solutions are not yet in the right order!!
        self.g = np.roots(p)

        # force all roots to have the same sign for real and imaginary parts
        self.g = np.where( np.sign(np.real(self.g)) != np.sign(np.imag(self.g)), np.conj(self.g), self.g )

        # some cleaning
        self.g = np.where( np.absolute(np.imag(self.g)) > 1e-6, self.g, np.real(self.g) )

        # sort the solution in two steps:
        # first, sort according to the sign of Re(g) +,-,+,-
        # second, sort for polarization; this is done in the next step (calculate_p_q)
        for i in range(4):
            if( np.sign(np.real(self.g[i])) == np.power(-1, i+1) ): # wrong sign
                for j in range(i+1, 4):
                    if(np.sign(np.real(self.g[i])) != np.sign(np.real(self.g[j]))):
                        self.g[i], self.g[j] = self.g[j], self.g[i]         # swap values
                        break                                               # break j-for loop

        return self.g.copy()

    def calculate_p_q(self, w, a, b):
        """Calculate the electric and magnetic polarization vectors p and q for the four solutions of `self.g`.

        :param float w: Frequency value.
        :param complex a: In-plane propagation constant along x-axis.
        :param complex b: In-plane propagation constant along y-axis.
        :returns: Electric and magnetic polarization vectors p and q sorted according to (x+, x-, y+, y-).

        .. note:: This function also sorts the in-plane propagation constants according to their polarizations (x+, x-, y+, y-).

        .. important:: Requires prior execution of :py:func:`calculate_g`.
        """

        has_to_sort = False

        # calculate normal vector on plane of incidence
        nPOI = np.cross(np.array([a, b, 0]), np.array([0,0,1]))

        # iterate over the four solutions of the z-propagation constant self.g
        for i in range(4):
            # this idea is partly based on Reider's book, as the explanation in the papers is misleading

            # define the matrix for getting the co-factors
            # use the complex conjugate to get the phases right!!
            M = np.conj(np.array([[w**2 * self.mu * self.epsilon[0,0] - b**2 - self.g[i]**2, w**2 * self.mu * self.epsilon[0,1] + a*b, w**2 * self.mu * self.epsilon[0,2] + a*self.g[i]], [w**2 * self.mu * self.epsilon[0,1] + a*b, w**2 * self.mu * self.epsilon[1,1] - a**2 - self.g[i]**2, w**2 * self.mu * self.epsilon[1,2] + b*self.g[i]], [w**2 * self.mu * self.epsilon[0,2] + a*self.g[i], w**2 * self.mu * self.epsilon[1,2] + b*self.g[i], w**2 * self.mu * self.epsilon[2,2] - b**2 - a**2]], dtype=np.complex128))

            # get null space to find out which polarization is associated with g[i]
            P, s = null_space(M)

            # directions have to be calculated according to plane of incidence ( defined by (a, b, 0) and (0, 0, 1) )
            # or normal to that ( defined by (a, b, 0) x (0, 0, 1) )
            if(len(s) == 1):    # single component
                has_to_sort = True
                self.p[i] = norm(P[0])
            else:

                if(i < 2):  # should be p pol
                #   print("looking for p:", np.absolute(np.dot(nPOI, P[0])))
                    if( np.absolute(np.dot(nPOI, P[0])) < 1e-3 ):
                        # polarization lies in plane of incidence made up by vectors ax + by and z
                        # => P[0] is p pol
                        self.p[i] = norm(P[0])
                    #   print("\t-> 0")
                    else:
                        # => P[1] has to be p pol
                        self.p[i] = norm(P[1])
                    #   print("\t-> 1")
                else:       # should be s pol
                #   print("looking for s:", np.absolute(np.dot(nPOI, P[0])))
                    if( np.absolute(np.dot(nPOI, P[0])) < 1e-3 ):
                        # polarization lies in plane of incidence made up by vectors ax + by and z
                        # => P[1] is s pol
                        self.p[i] = norm(P[1])
                    #   print("\t-> 1")
                    else:
                        # => P[0] has to be s pol
                        self.p[i] = norm(P[0])
                    #   print("\t-> 0")


        # if solutions were unique, sort the polarization vectors according to p and s polarization
        # the sign of Re(g) has been taken care of already
        if(has_to_sort == True):
            for i in range(2):
                if( np.absolute(np.dot(nPOI, self.p[i])) > 1e-3 ):
                    self.g[i], self.g[i+2] = self.g[i+2], self.g[i]
                    self.p[[i, i+2]] = self.p[[i+2, i]]                 # IMPORTANT! standard python swapping does not work for 2d numpy arrays; use advanced indexing instead

        for i in range(4):
            # select right orientation or p vector - see Born, Wolf, pp 39
            if((i == 0 and np.real(self.p[i][0]) > 0.0) or (i == 1 and np.real(self.p[i][0]) < 0.0) or (i >= 2 and np.real(self.p[i][1]) < 0.0)):
                self.p[i] *= -1.0
            #self.p[i][2] = np.conj(self.p[i][2])
            # calculate the corresponding q-vectors by taking the cross product between the normalized propagation constant and p[i]
            K = np.array([a, b, self.g[i]], dtype=np.complex128)
            self.q[i] = np.cross(K, self.p[i]) * self.c / (w * self.mu)

        return [self.p.copy(), self.q.copy()]

    # calculate the dynamic matrix and its inverse
    def calculate_D(self):
        """Calculate the dynamic matrix and its inverse using the previously calculated values for p and q.

        returns: :math:`D`, :math`D^{-1}`

         .. important:: Requires prior execution of :py:func:`calculate_p_q`.
        """
        self.D[0,0] = self.p[0,0]
        self.D[0,1] = self.p[1,0]
        self.D[0,2] = self.p[2,0]
        self.D[0,3] = self.p[3,0]
        self.D[1,0] = self.q[0,1]
        self.D[1,1] = self.q[1,1]
        self.D[1,2] = self.q[2,1]
        self.D[1,3] = self.q[3,1]
        self.D[2,0] = self.p[0,1]
        self.D[2,1] = self.p[1,1]
        self.D[2,2] = self.p[2,1]
        self.D[2,3] = self.p[3,1]
        self.D[3,0] = self.q[0,0]
        self.D[3,1] = self.q[1,0]
        self.D[3,2] = self.q[2,0]
        self.D[3,3] = self.q[3,0]

        self.Di = np.linalg.pinv(self.D)#, rcond=1e-20)

        return [self.D.copy(), self.Di.copy()]

    # calculate the propagation matrix
    def calculate_P(self):
        """Calculate the propagation matrix using the previously calculated values for g.

        returns: :math:`P`

        .. important:: Requires prior execution of :py:func:`calculate_g`.
        """
        self.P = np.diag(np.exp(-1j * self.g * self.d))
        return self.P.copy()

    # calculate the layer transfer matrix
    def calculate_T(self, calculateDP = True):
        """Calculate the layer transfer matrix. If D and P have not yet been calculated, these are calculated automatically.

        :returns: :math:`T = D P D^{-1}`.

        .. note:: This formulation of the layer transfer matrix is slightly different from the one used by Yeh (:math:`T_{n-1,n} =  D_{n-1}^{-1} D_n P_n`).

        .. important:: Requires prior execution of :py:func:`calculate_g` and :py:func:`calculate_p_q`.
        """
        if(calculateDP == True):
            self.calculate_D()
            self.calculate_P()

        self.T = np.dot(self.D, np.dot(self.P, self.Di))
        return self.T.copy()

    # shortcut that can be used when the wavelength is changed; can be called directly after __init__()
    # uses the frequency w, and in-plane propagation constants a and b to recalculate all layer properties
    def update(self, w, a, b):
        """Shortcut to recalculate all layer properties. Can be used, for instance, when only the wavelength is changed.

        :param float w: Frequency value.
        :param complex a: In-plane propagation constant along x-axis.
        :param complex b: In-plane propagation constant along y-axis.
        :returns: Dynamic matrix D and its inverse, propagation matrix P, and layer transfer matrix T.
        """
        self.calculate_epsilon(w)
        self.calculate_g(w, a, b)
        self.calculate_p_q(w, a, b)
        self.calculate_D()
        self.calculate_P()
        self.calculate_T()
        return [self.D.copy(), self.Di.copy(), self.P.copy(), self.T.copy()]

# ******************************************************************************************************************************* #
# system class
# this is the actual interface for calculations
class System:
    """Construct an instance of the System class, which manages the optical system.

    An optical system consists of the first (semi-infinite) layer (substrate), the intermediate layers and the last (semi-infinite) layer (superstrate). The *superstrate* is assumed to be **isotropic** and only the xx component of the dielectric tensor is used for the calculation of the in-plane (x, y) propagation constants.

    :param float theta: Angle of incidence versus the surface normal / z-axis (deg, default = 0). The plane of incidence is the x-z-plane.
    :param float phi: Angle of rotation of the system around the z-axis (deg, default = 0).
    :param layer substrate: Layer definition of the substrate. If None, use vacuum.
    :param layer superstrate: Layer definition of the superstrate. If None, use vacuum.
    :param list layers: A list of layer definitions for the intermediate layers sorted from substrate to superstrate.

    .. important:: The frequency w which is used for the calculations is actually the wavenumber in vacuum, i.e. :math:`w = \\frac{2\\pi}{\\lambda}`.
    """
    c = 1 #: speed of light in vacuum

    # these are the first (semiinfinite) layer, central layers and last (semiinfinite) layer
    # the superstrate is assumed to be isotropic (!) and only the x,x component of the dielectric tensor
    # is used for calculation of the in-plane propagation constants
    # if no values are given for superstrate / substrate, vacuum is assumed
    # theta and phi are the angles (in deg) of incidence relative to the z-axis and the rotation of the system relative to
    # the x-axis that are used to calculate the in-plane propagation constants a, b
    # the frequency w is actually wavevector in vacuum, i.e. w = 2 Pi / lambda
    def __init__(self, theta = 0.0, phi = 0.0, substrate = None, superstrate = None, layers = []):

        self.layers = []
        if(len(layers) > 0):
            self.layers = layers

        self.a = 0.0
        self.b = 0.0
        self.T = np.zeros((4,4), dtype=np.complex128)

        # in-plane propagation constants are initialized with zero for normal incidence; use interface functions to change that
        self.theta = theta * np.pi / 180.0
        self.phi = phi * np.pi / 180.0

        if(substrate != None):
            self.substrate = substrate
        else:
            self.substrate = Layer()
        if(superstrate != None):
            self.superstrate = superstrate
        else:
            self.superstrate = Layer()

    def set_substrate(self, S):
        """Set the substrate to S.
        """
        self.substrate = np.copy(S)

    def set_superstrate(self, S):
        """Set the superstrate to S.
        """
        self.superstrate = np.copy(S)

    def get_substrate(self):
        """Returns the substrate layer class.
        """
        return self.substrate

    def get_superstrate(self):
        """Returns the superstrate layer class.
        """
        return self.superstrate

    def set_theta(self, theta):
        """Set angle of incidence to theta (in deg).
        """
        self.theta = theta * np.pi / 180.0

    def set_phi(self, phi):
        """Set rotation of sample around the z-axis (in deg).
        """
        self.phi = phi * np.pi / 180.0

    # layers have to be added from substrate to superstrate!!; the beam is assumed incident from the superstrate!!
    def add_layer(self, L):
        """Add a layer L to the system. Layers are added from substrate to superstrate. The beam is assumed incident from the superstrate side.

        .. note:: This function adds a reference to L to the list. So if you are adding the same layer several times, be aware that if you change something for one of them, it changes all of them.
        """
        self.layers.append(L)

    def get_layers(self):
        """Returns the list of layers.
        """
        return self.layers

    def get_num_layers(self):
        """Returns the number of layers in the stack.
        """
        return len(self.layers)

    def get_layer(self, pos):
        """Returns the layer specified by position.

        :param int pos: Position of the layer in the stack (0 closest to substrate).
        """
        if pos >= 0 and pos < len(self.layers):
            return self.layers[pos]
        else:
            return None

    def insert_layer(self, L, pos):
        """Insert a layer at a given position.

        :param Layer L: Layer class instance to add to the stack.
        :param int pos: Position in the stack (0 to N, where 0 is closest to substrate, and N is number of layers).
        """
        self.layers.insert(max(min(len(self.layers), pos), 0), L)

    def remove_layer(self, pos):
        """Removes the layer at a given position.

        :param int pos: Position of the layer to be removed (0 to N-1). Does nothing if pos is invalid.
        """
        if pos >= 0 and pos < len(self.layers):
            self.layers.pop(pos)

    # calculate system transfer matrix for given frequency w
    def calculate_T(self, w):
        """Calculate the system transfer matrix :math:`T_{0,S}` for a given frequency.

        :param float w: Frequency value.
        :returns: System transfer matrix.
        """
        # calculate dielectric tensor of superstrate
        e = self.superstrate.calculate_epsilon(w)

        # use only xx component assuming it's an isotropic layer
        k0 = w / self.c * np.lib.scimath.sqrt(e[0,0])

        # the in-plane propagation constants are the projection of k0 onto the x and y axes
        self.a = k0 * np.sin(self.theta) * np.cos(self.phi)
        self.b = k0 * np.sin(self.theta) * np.sin(self.phi)

        # use a list for the transfer matrix; by doing this, we can separate coherent and incoherent parts of the system
        Ttot = []

        # start with the dynamic matrix substrate
        T, _, _, _ = self.substrate.update(w, self.a, self.b)

        # now iterate through all layers and multiply the matrices
        for i in range(len(self.layers)):
            # calculate dynamic and propagation matrices
            D, Di, P, Ttmp = self.layers[i].update(w, self.a, self.b)

            if(self.layers[i].thick == True):
                # if the layer is thick, i.e. incoherent propagation should be used, finish the existing T matrix by multiplying
                # with Di and add as new matrix in Ttot
                # then add P as new matrix in Ttot
                # then start a new T matrix with D
                T = np.dot(Di, T)
                Ttot.append(T)
                Ttot.append(P)
                T = np.copy(D)
            else:
                # if propagation is coherent, proceed as usual
                T = np.dot(Ttmp, T)

        # finally add the superstrate inverse dynamic matrix
        _, Ttmp, _, _ = self.superstrate.update(w, self.a, self.b)
        T = np.dot(Ttmp, T)

        # append last T matrix to Ttot
        Ttot.append(T)

        # depending on the number of entries in Ttot, assemble the total T matrix coherently or incoherently
        # assemble the system transfer matrix
        if(len(Ttot) == 1):         # coherent
            self.T = Ttot[0]
        else:
            # for incoherent propagation, multiply the absolute value of the T matrices
            # thereby, the calculation of the coefficients still remains valid
            # of course the field coefficients will lack the phase information
            self.T = np.identity(4, dtype=np.complex128)
            for i in range(len(Ttot)):
                self.T = np.dot(np.absolute(Ttot[i])**2, self.T)
            self.T = np.sqrt(self.T)

        return self.T.copy()

    # reflection and transmission coefficients
    def rxx(self):
        """Returns the x,x component of the field reflection coefficient (incident x, reflected x).

        .. important:: Requires prior execution of :py:func:`calculate_T`.
        """
        return np.nan_to_num((self.T[1,0] * self.T[2,2] - self.T[1,2] * self.T[2,0])/(self.T[0,0] * self.T[2,2] - self.T[0,2] * self.T[2,0]))

    def rxy(self):
        """Returns the x,y component of the field reflection coefficient (incident x, reflected y).

        .. important:: Requires prior execution of :py:func:`calculate_T`.
        """
        return np.nan_to_num((self.T[3,0] * self.T[2,2] - self.T[3,2] * self.T[2,0])/(self.T[0,0] * self.T[2,2] - self.T[0,2] * self.T[2,0]))

    def ryx(self):
        """Returns the y,x component of the field reflection coefficient (incident y, reflected x).

        .. important:: Requires prior execution of :py:func:`calculate_T`.
        """
        return np.nan_to_num((self.T[0,0] * self.T[1,2] - self.T[1,0] * self.T[0,2])/(self.T[0,0] * self.T[2,2] - self.T[0,2] * self.T[2,0]))

    def ryy(self):
        """Returns the y,y component of the field reflection coefficient (incident y, reflected y).

        .. important:: Requires prior execution of :py:func:`calculate_T`.
        """
        return np.nan_to_num((self.T[0,0] * self.T[3,2] - self.T[3,0] * self.T[0,2])/(self.T[0,0] * self.T[2,2] - self.T[0,2] * self.T[2,0]))

    def txx(self):
        """Returns the x,x component of the field transmission coefficient (incident x, transmitted x).

        .. important:: Requires prior execution of :py:func:`calculate_T`.
        """
        return np.nan_to_num(self.T[2,2]/(self.T[0,0] * self.T[2,2] - self.T[0,2] * self.T[2,0]))

    def txy(self):
        """Returns the x,y component of the field transmission coefficient (incident x, transmitted y).

        .. important:: Requires prior execution of :py:func:`calculate_T`.
        """
        return np.nan_to_num(-self.T[2,0]/(self.T[0,0] * self.T[2,2] - self.T[0,2] * self.T[2,0]))

    def tyx(self):
        """Returns the y,x component of the field transmission coefficient (incident y, transmitted x).

        .. important:: Requires prior execution of :py:func:`calculate_T`.
        """
        return np.nan_to_num(-self.T[0,2]/(self.T[0,0] * self.T[2,2] - self.T[0,2] * self.T[2,0]))

    def tyy(self):
        """Returns the y,y component of the field transmission coefficient (incident y, transmitted x).

        .. important:: Requires prior execution of :py:func:`calculate_T`.
        """
        return np.nan_to_num(self.T[0,0]/(self.T[0,0] * self.T[2,2] - self.T[0,2] * self.T[2,0]))

    def get_field_coefficients(self, w):
        """Shortcut to calculate all reflection / transmission coefficients as function of frequency w.

        :param array w: Frequency axis.
        :returns: rxx, rxy, ryx, ryy, txx, txy, tyx, tyy (with same shape as w).
        """
        rxx = np.zeros(len(w), dtype=np.complex128)
        rxy = np.zeros(len(w), dtype=np.complex128)
        ryx = np.zeros(len(w), dtype=np.complex128)
        ryy = np.zeros(len(w), dtype=np.complex128)
        txx = np.zeros(len(w), dtype=np.complex128)
        txy = np.zeros(len(w), dtype=np.complex128)
        tyx = np.zeros(len(w), dtype=np.complex128)
        tyy = np.zeros(len(w), dtype=np.complex128)

        for i in range(len(w)):
            self.calculate_T(w[i])
            rxx[i] = self.rxx()
            rxy[i] = self.rxy()
            ryx[i] = self.ryx()
            ryy[i] = self.ryy()
            txx[i] = self.txx()
            txy[i] = self.txy()
            tyx[i] = self.tyx()
            tyy[i] = self.tyy()

        return [rxx, rxy, ryx, ryy, txx, txy, tyx, tyy]

    # shortcut to calculate all transmittance / reflectance terms as function of frequency w - this time a numpy.array
    def get_intensity_coefficients(self, w):
        """Shortcut to calculate all intensity reflectance / transmittance terms as function of frequency w.
        The transmittance terms are angle of incidence and refractive index corrected.

        :param array w: Frequency axis.
        :returns: Rxx, Rxy, Ryx, Ryy, Txx, Txy, Tyx, Tyy (with same shape as w).
        """
        rxx = np.zeros(len(w), dtype=np.float64)
        rxy = np.zeros(len(w), dtype=np.float64)
        ryx = np.zeros(len(w), dtype=np.float64)
        ryy = np.zeros(len(w), dtype=np.float64)
        txx = np.zeros(len(w), dtype=np.float64)
        txy = np.zeros(len(w), dtype=np.float64)
        tyx = np.zeros(len(w), dtype=np.float64)
        tyy = np.zeros(len(w), dtype=np.float64)

        for i in range(len(w)):
            self.calculate_T(w[i])
            # get coefficients
            rxx[i] = np.absolute(self.rxx())**2
            rxy[i] = np.absolute(self.rxy())**2
            ryx[i] = np.absolute(self.ryx())**2
            ryy[i] = np.absolute(self.ryy())**2
            txx[i] = np.real(self.substrate.g[0] / self.superstrate.g[0]) * np.absolute(self.txx())**2
            txy[i] = np.real(self.substrate.g[0] / self.superstrate.g[2]) * np.absolute(self.txy())**2
            tyx[i] = np.real(self.substrate.g[2] / self.superstrate.g[0]) * np.absolute(self.tyx())**2
            tyy[i] = np.real(self.substrate.g[2] / self.superstrate.g[2]) * np.absolute(self.tyy())**2

        return [rxx, rxy, ryx, ryy, txx, txy, tyx, tyy]

    # shortcut to calculate all reflection / transmission coefficients as function of incidence angle AOI for given
    # frequency w; AOI is a numpy array
    def get_field_coefficients_AOI(self, w, AOI):
        """Shortcut to calculate all reflection / transmission coefficients as function of angle of incidence.

        :param float w: Frequency value.
        :param array AOI: List of incidence angles (deg).
        :returns: rxx, rxy, ryx, ryy, txx, txy, tyx, tyy (with same shape as AOI).
        """
        rxx = np.zeros(len(AOI), dtype=np.complex128)
        rxy = np.zeros(len(AOI), dtype=np.complex128)
        ryx = np.zeros(len(AOI), dtype=np.complex128)
        ryy = np.zeros(len(AOI), dtype=np.complex128)
        txx = np.zeros(len(AOI), dtype=np.complex128)
        txy = np.zeros(len(AOI), dtype=np.complex128)
        tyx = np.zeros(len(AOI), dtype=np.complex128)
        tyy = np.zeros(len(AOI), dtype=np.complex128)

        for i in range(len(AOI)):
            # set angle
            self.theta = AOI[i] * np.pi / 180.0
            # get transfer matrix
            self.calculate_T(w)
            # get coefficients
            rxx[i] = self.rxx()
            rxy[i] = self.rxy()
            ryx[i] = self.ryx()
            ryy[i] = self.ryy()
            txx[i] = self.txx()
            txy[i] = self.txy()
            tyx[i] = self.tyx()
            tyy[i] = self.tyy()

        return [rxx, rxy, ryx, ryy, txx, txy, tyx, tyy]

    # same for intensities
    def get_intensity_coefficients_AOI(self, w, AOI):
        """Shortcut to calculate all intensity reflectance / transmittance terms as function of angle of incidence.

        :param float w: Frequency value.
        :param array AOI: List of incidence angles (deg).
        :returns: Rxx, Rxy, Ryx, Ryy, Txx, Txy, Tyx, Tyy (with same shape as AOI).
        """
        rxx = np.zeros(len(AOI), dtype=np.float64)
        rxy = np.zeros(len(AOI), dtype=np.float64)
        ryx = np.zeros(len(AOI), dtype=np.float64)
        ryy = np.zeros(len(AOI), dtype=np.float64)
        txx = np.zeros(len(AOI), dtype=np.float64)
        txy = np.zeros(len(AOI), dtype=np.float64)
        tyx = np.zeros(len(AOI), dtype=np.float64)
        tyy = np.zeros(len(AOI), dtype=np.float64)

        for i in range(len(AOI)):
            # set angle
            self.theta = AOI[i] * np.pi / 180.0
            # get transfer matrix
            self.calculate_T(w)
            # get coefficients
            rxx[i] = np.absolute(self.rxx())**2
            rxy[i] = np.absolute(self.rxy())**2
            ryx[i] = np.absolute(self.ryx())**2
            ryy[i] = np.absolute(self.ryy())**2
            txx[i] = np.real(self.substrate.g[0] / self.superstrate.g[0]) * np.absolute(self.txx())**2
            txy[i] = np.real(self.substrate.g[0] / self.superstrate.g[2]) * np.absolute(self.txy())**2
            tyx[i] = np.real(self.substrate.g[2] / self.superstrate.g[0]) * np.absolute(self.tyx())**2
            tyy[i] = np.real(self.substrate.g[2] / self.superstrate.g[2]) * np.absolute(self.tyy())**2

        return [rxx, rxy, ryx, ryy, txx, txy, tyx, tyy]

    def get_electric_field(self, w, dz, pol=(1,0), x = 0.0, y = 0.0):
        """Calculate the electric field profile for a given frequency.

        The z-axis range is taken according to the total thickness of the layer stack plus the thicknesses of substrate and superstrate.

        :param float w: Frequency value.
        :param float dz: Step size along the z-axis.
        :param tuple pol: (Complex) Ratio of x- and y-amplitudes of incident radiation (default = x-pol).
        :param float x: x-coordinate.
        :param float y: y-coordinate.
        :returns: z-axis, complex electric field vector (3d) along z-axis ([z][Ex, Ey, Ez]), and list of z-coordinates of interfaces (zn).

        .. important:: This function ignores the `thick` parameter, i.e. propagation is always coherent.
        """
        # first calculate the overall system response
        self.calculate_T(w)

        # now iterate through all layers and get the necessary parameters for each layer
        N = len(self.layers)

        # z-coordinate of substrate sided boundary
        zn = np.zeros(N+2)

        # amplitude vector
        An = np.zeros((N+2, 4), dtype=np.complex128)

        # get incident polarization and calculate the correct outgoing amplitudes
        Ax = pol[0] / np.sqrt(pol[0]**2 + pol[1]**2)
        Ay = pol[1] / np.sqrt(pol[0]**2 + pol[1]**2)

        # start with substrate
        zn[0] = 0.0#self.substrate.d   # set the substrate layer z to zero to get the correct amplitudes
        An[0] = np.array([self.txx() * Ax + self.tyx() * Ay, 0.0, self.tyy() * Ay + self.txy() * Ax, 0.0])

        if N > 0:
            # calculate first layer manually
            zn[1] = 0.0
            T = np.dot(self.layers[0].Di, self.substrate.D)
            # ignore the substrate P-matrix as the amplitudes are for the last interface
            An[1] = np.dot(T, An[0])

            # calculate intermediate layers in loop
            for i in range(2, N+1):
                zn[i] = zn[i-1] - self.layers[i-2].d

                T = np.dot(self.layers[i-1].Di, np.dot(self.layers[i-2].D, self.layers[i-2].P))
                An[i] = np.dot(T, An[i-1])

            # calculate last layer / superstrate manually again
            zn[-1] = zn[-2] - self.layers[-1].d

            T = np.dot(self.superstrate.Di, np.dot(self.layers[-1].D, self.layers[-1].P))
            An[-1] = np.dot(T, An[-2])
        else:
            zn[1] = 0.0
            #T = np.dot(self.superstrate.Di, np.dot(self.substrate.D, self.substrate.P))
            T = np.dot(self.superstrate.Di, self.substrate.D)
            An[1] = np.dot(T, An[0])

        # shift z-coordinates such that the top surface is at zero
        zn = zn - zn[-1]

        # generate z axis and field vector array
        z = np.arange(-self.superstrate.d, zn[0] + self.substrate.d, dz)
        E = np.zeros((len(z), 3), dtype=np.complex128)

        current_layer = N+1

        # calculate the field distribution
        for i in range(len(z)):

            # check for current layer
            if z[i] > zn[current_layer] and current_layer > 0:
                current_layer -= 1

            if current_layer == 0:
                L = self.substrate
            elif current_layer == N+1:
                L = self.superstrate
            else:
                L = self.layers[current_layer - 1]

            # calculate field
            E[i] = np.dot(An[current_layer] * np.exp(1j * self.a * x + 1j * self.b * y + 1j * L.g * (z[i] - zn[current_layer])), L.p)

        # return result
        return z, E, zn


# ******************************************************************************************************************************* #
# Needle Algorithm
# See Appl. Opt. 35, 5484 (1996) for details.

def get_merit(S, w, Q, dQ=0.01, Qid=4):
    """Calculate the merit function of the current layer stack.

    :param System S: Optical system containing the layer stack.
    :param array w: List of frequencies where a target is specified.
    :param array Q: List of target values at the given frequencies (same shape as w).
    :param float dQ: Allowed design tolerance of the target (in absolute units).
    :param int Qid: Identifier of the quantity that should be used for calculation of the merit function. (0 = Rxx, ..., 7 = Tyy).
    :returns: Value of merit function.
    """
    QS = S.get_intensity_coefficients(w)[Qid]
    return np.sqrt(np.mean(((QS - Q) / dQ)**2))


def optimize_thicknesses(S, w, Q, dQ=0.01, Qid=4, keep_first_fixed=True):
    """Optimize the thicknesses of each layer to minimize the merit function.
    The algorithm uses a simple Newton optimization.

    :param System S: Optical system containing the layer stack.
    :param array w: List of frequencies where a target is specified.
    :param array Q: List of target values at the given frequencies (same shape as w).
    :param float dQ: Allowed design tolerance of the target (in absolute units).
    :param int Qid: Identifier of the quantity that should be used for calculation of the merit function. (0 = Rxx, ..., 7 = Tyy).
    :param bool keep_first_fixed: Set to True if first layer should not be modified.
    :returns: Value of merit function.
    """
    SO = copy.deepcopy(S)


    # minimization function, x is list of layer thicknesses
    def minfunc(x):
        for i, d in enumerate(x):
            SO.get_layer(i + int(keep_first_fixed)).d = d
        return get_merit(SO, w, Q, dQ, Qid)

    # get initial thicknesses as list
    x0 = [L.d for i, L in enumerate(SO.get_layers()[int(keep_first_fixed):])]
    bnds = [(0, None) for _ in x0]
    # minimize the merit function by changing layer thicknesses
    x = sp.minimize(minfunc, x0, bounds=bnds).x
    # apply new thicknesses
    print "========================================="
    print "New thicknesses:"
    for i, d in enumerate(x):
        SO.get_layer(i + int(keep_first_fixed)).d = d
        print "Layer %d: thickness = %f" % (i + int(keep_first_fixed), d)
    print "========================================="
    return SO


def insert_needle(S, pos, M1, M2, d=1e-2):
    """Insert a needle, i.e., a very thin layer with different refractive index into the center of the layer at a given position
    and return the resulting System class instance. The function creates a NEW System class using deepcopy and modifies this one. The old System instance remains unchanged.

    :param System S: Optical system.
    :param int pos: Position / index of layer into which the needle is inserted (0..N-1).
    :param func M1: Dielectric function 1. The needle will use either M1 or M2 depending on the dielectric function of the host layer.
    :param func M2: Dielectric function 2.
    :param float d: Thickness of the needle layer. Usually around 10nm (default 0.01 assumes units are um).
    :returns: New System class.
    """
    SN = copy.deepcopy(S)
    L0 = SN.get_layer(pos)
    if L0 is not None:
        if L0.fepsilon1 is M1:
            neps = M2
        else:
            neps = M1
        L0.d = L0.d / 2.0

        SN.insert_layer(copy.deepcopy(L0), pos + 1)
        SN.insert_layer(Layer(epsilon1=neps, thickness=d), pos + 1)
    return SN


def remove_needle(S, pos):
    """Remove a layer from the layer stack in the given System from the given position. If adjacent layers are equal,
    combine into one thicker layer.

    :param System S: System class.
    :param int pos: Position of the layer to be removed (0..N-1).
    :returns: S without the specified layer.
    """
    SN = copy.deepcopy(S)
    SN.remove_layer(pos)
    # mergin of adjacent layers is only relevant for intermediate layers
    if pos > 0 and pos < S.get_num_layers() - 1:
        if SN.get_layer(pos - 1).fepsilon1 is SN.get_layer(pos).fepsilon1:
            SN.get_layer(pos - 1).d = SN.get_layer(pos - 1).d + SN.get_layer(pos).d
            SN.remove_layer(pos)
    return SN


def remove_all_needles(S, dmin=1e-4):
    """Iteratively removes all layers from the System class whose thickness is below dmin.

    :param System S: System class.
    :param float dmin: Minimum thickness for each layer (default 1e-3, i.e., 1nm assuming units are 1um).
    :returns: S without thin layers.
    """
    found_needle = -1
    for i, L in enumerate(S.get_layers()):
        if L.d < dmin:
            found_needle = i
            break
    if found_needle > -1:
        S = remove_needle(S, found_needle)
        S = remove_all_needles(S, dmin)
    return S


def optimized_insert_needle(S, w, Q, M1, M2, d=1e-2, dQ=0.01, Qid=4, keep_first_fixed=True):
    """Insert a needle in the Layer stack at the optimum position by comparing the merit functions of all combinations.

    :param System S: Optical system containing the layer stack.
    :param array w: List of frequencies where a target is specified.
    :param array Q: List of target values at the given frequencies (same shape as w).
    :param func M1: Dielectric function 1. The needle will use either M1 or M2 depending on the dielectric function of the host layer.
    :param func M2: Dielectric function 2.
    :param float d: Thickness of the needle layer. Usually around 10nm (default 0.01 assumes units are um).
    :param float dQ: Allowed design tolerance of the target (in absolute units).
    :param int Qid: Identifier of the quantity that should be used for calculation of the merit function. (0 = Rxx, ..., 7 = Tyy).
    :param bool keep_first_fixed: Set to True if first layer should not be modified.
    :returns: New System class instance.
    """
    # get list of possible positions for inserting the needle
    positions = range(S.get_num_layers())[int(keep_first_fixed):]

    # list of merit function values
    M = np.zeros(len(positions))
    M0 = get_merit(S, w, Q, dQ, Qid)

    # get merit values for all layers
    for i, pos in enumerate(positions):
        SN = insert_needle(S, pos, M1, M2, d)
        M[i] = get_merit(SN, w, Q, dQ, Qid)

    # retain the one with the lowest merit function if its value is below the initial one
    i1 = np.argmin(M)
#    if M[i1] < M0:
    return insert_needle(S, positions[i1], M1, M2, d)
#    return S


def needle(S, w, Q, M1, M2, d=1e-2, dQ=0.01, Qid=4, keep_first_fixed=True, max_num_layers=100, max_num_iterations=100, min_layer_thickness=1e-3, min_merit=1e-4):
    """Simplified implementation of the needle algorithm for thin film design. See Appl. Opt. 35, 5484 (1996) for details.
    This is the only function to be called directly by the user. Given an initial layer stack and some target values, the algorithm tries to replicate the target by insertion of additional layers and thickness optimization.

    :param System S: Initial optical system containing the layer stack.
    :param array w: List of frequencies where a target is specified.
    :param array Q: List of target values at the given frequencies (same shape as w).
    :param func M1: Dielectric function 1. The needle will use either M1 or M2 depending on the dielectric function of the host layer.
    :param func M2: Dielectric function 2.
    :param float d: Thickness of the needle layer. Usually around 10nm (default 0.01 assumes units are um).
    :param float dQ: Allowed design tolerance of the target (in absolute units).
    :param int Qid: Identifier of the quantity that should be used for calculation of the merit function. (0 = Rxx, ..., 7 = Tyy).
    :param bool keep_first_fixed: Set to True if first layer should not be modified.
    :param int max_num_layers: Maximum number of layers permitted (default 100).
    :param int max_num_iterations: Maximum number of iterations (default 100).
    :param float min_layer_thickness: Minimum thickness of a layer. If, after thickness optimization, a layer turns out thinner, it is discarded and the resulting stack simplified (default 1e-3, i.e. 1nm assuming units are um).
    :param float min_merit: Threshold value for merit function to be reached for convergence (default 1e-4).
    :returns: System class instance with optimized design.
    """
    SO = optimize_thicknesses(S, w, Q, dQ, Qid, keep_first_fixed)
    ME0 = get_merit(SO, w, Q, dQ, Qid)
    print "Initial merit = %f" % ME0

    for i in range(max_num_iterations):
        print "iteration %d" % i

        # insert needle
        print " .. insert needle"
        SN = optimized_insert_needle(SO, w, Q, M1, M2, d, dQ, Qid, keep_first_fixed)

        # optimize thicknesses first
        print " .. optimize thicknesses"
        SN = optimize_thicknesses(SN, w, Q, dQ, Qid, keep_first_fixed)

        # remove too thin layers
        SN = remove_all_needles(SN, min_layer_thickness)

        # if no improvement of merit function was achieved, break
        ME1 = get_merit(SN, w, Q, dQ, Qid)
        print " .. new merit = %f" % ME1
        if ME1 >= ME0 and i > 3:
            break

        # assign new System class
        SO = SN
        ME0 = ME1

        # if current merit function value is below threshold, break
        if ME0 <= min_merit:
            break

        # check number of layers before next iteration
        if SO.get_num_layers() >= max_num_layers - 2:
            break

    return SO
