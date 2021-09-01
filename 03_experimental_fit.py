import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from numpy.linalg import inv
from numpy import pi, sin, cos, sqrt
from scipy.interpolate import griddata
from scipy.constants import physical_constants as pc
from uncertainties import ufloat
from uncertainties import umath as um
# math functions are getting a bit messy, but cleaning it up isn't worth the time

# New data
# Starting with background measurements, avg of before and after
# and convert it to mag field from potential
Bx = (-2.241176477 + -2.24198550783517) * 9500 / \
    2
Bz = (-5.29412940998822 + -5.27280325707483) * 9500 / 2
# load in data from file
filename = 'data.csv'
data = np.genfromtxt(filename, delimiter=',')
# store columns as vectors
latit = data[:, 0]
longit = data[:, 1] % 360
X = data[:, 2]-Bx
Z = data[:, 3]-Bz
# we can roughly approximate standard deviation as range/4
# then square it to get the variance to weight least squares
xVar = (np.array(data[:, 4]) / 4) ** 2
zVar = (np.array(data[:, 5]) / 4) ** 2

# amount of points measured for one coordinate
N = np.shape(X)[0]

# plot the data on a worldmap, according to the original code.
# err, except this doesn't plot it onto a worldmap?
# I've dropped it, deeming it unecessary given the later plots.
# plt.subplot(211)
# plt.contourf(Xi.T, 32, extent=(0, 360, -90, 90))
# plt.scatter(longit, latit, marker='.')
# plt.subplot(212)
# plt.contourf(Zi.T, 32, extent=(0, 360, -90, 90))
# plt.scatter(longit, latit, marker='.')
# plt.show()


# create the linear system
kernel = np.zeros((2*N, 8))  # initalise kernel
measurements = np.append(X, Z)  # create data vector

# convert lat/long into standard spherical coordinates
theta = (pi/180)*(90-latit)
phi = (pi/180)*longit

# The need for normalised polynomials is unclear in the lab script.
# format is generic coefficents * associated legendre polynomials
kernel[:N, 0] = -sin(theta)  # g_1^0
kernel[:N, 1] = cos(phi) * cos(theta)  # g_1^1
kernel[:N, 2] = sin(phi) * cos(theta)  # h_1^1
kernel[:N, 3] = -3*sin(theta)*cos(theta)  # g_2^0
kernel[:N, 4] = cos(phi) * sqrt(3)*(cos(theta)**2 - sin(theta)**2)  # g_2^1
kernel[:N, 5] = sin(phi) * sqrt(3)*(cos(theta)**2 - sin(theta)**2)  # h_2^1
kernel[:N, 6] = cos(2*phi) * sqrt(3)*sin(theta)*cos(theta)  # g_2^2
kernel[:N, 7] = sin(2*phi) * sqrt(3)*sin(theta)*cos(theta)  # h_2^2

kernel[N:, 0] = -2 * cos(theta)  # g_1^0
kernel[N:, 1] = -2*cos(phi) * sin(theta)  # g_1^1
kernel[N:, 2] = -2*sin(phi) * sin(theta)  # h_1^1
kernel[N:, 3] = -3 * 1/2*(3*cos(theta)**2 - 1)  # g_2^0
kernel[N:, 4] = -3*cos(phi) * sqrt(3)*cos(theta)*sin(theta)  # g_2^1
kernel[N:, 5] = -3*sin(phi) * sqrt(3)*cos(theta)*sin(theta)  # h_2^1
kernel[N:, 6] = -3*cos(2*phi) * sqrt(3)/2*sin(theta)**2  # g_2^2
kernel[N:, 7] = -3*sin(2*phi) * sqrt(3)/2*sin(theta)**2  # h_2^2

# generate weighting matrix
# weighting by inverse of variances causes more certain values have greater influence on the fit
weights = np.diag(1 / np.append(xVar, zVar))

# Now solve this with weighted least squares
# See Geophysical Data Analysis: Discrete Inverse Theory (by Menke), eq. 3.47 of 4th ed.
# The sample code uses unweighted least squares. This change allows the code to partially account for errors in measurements.
# First we'll construct a generalised inverse
genInverse = inv(kernel.T @ weights @
                 kernel) @ kernel.T @ weights
# Then find parameters
parameters = genInverse @ measurements
# Total fit error. Least squares seeks to minimise this function.
error = kernel @ parameters - measurements  # error vector
error = error.T @ weights @ error  # length of vector
# Now for the standard deviation of the parameter estimates
# data covariance matrix, assuming uncorrelated error
dataCovar = np.diag(np.append(xVar, zVar))
paramCovar = genInverse @ dataCovar @ genInverse.T  # parameter covariance matrix
paramStd = sqrt(np.diag(paramCovar))  # convert to vector

# Let's store the parameters with meaningful names
keys = ['g10', 'g11', 'h11', 'g20', 'g21', 'h21',
        'g22', 'h22']  # key format glm = g_l^m
gauss = dict(zip(keys, [ufloat(e[0], e[1])
             for e in zip(parameters, paramStd)]))

# And print them for our use
print(gauss)
print(f'Total fit error: {error}')

# We'll need these in a moment
R = 0.15  # globe radius in metres
# pull in value from scipy's CODATA2018 database
mu0 = pc['vacuum mag. permeability'][0]

# Now we can start calculating derived values
# Using the same variable naming as the lab script where I can
derived = {}  # we'll store them in another dict
derived['m0'] = 4*pi/mu0*R**3*gauss['g10'] * \
    1e-9  # dipole moment of axial dipole in Am^2
derived['m1'] = 4*pi/mu0*R**3 * \
    um.sqrt((gauss['g11']*1e-9)**2 + (gauss['h11']*1e-9)
            ** 2)  # equatorial dipole moment in Am^2
derived['m'] = 4*pi/mu0*R**3*um.sqrt((gauss['g10']*1e-9)**2 + (gauss['g11']*1e-9)
                                     ** 2 + (gauss['h11']*1e-9)**2)  # total dipole moment in Am^2

# angle of tot. dipole inclination w.r.t rotation axis in radians
derived['I'] = um.atan(
    um.sqrt(gauss['g11']**2 + gauss['h11']**2) / gauss['g10'])
derived['Ie'] = um.atan(gauss['h11'] / gauss['g11']
                        )  # alignment of eq. dipole in radians
print(derived)  # and print it, of course


# And finally we'll plot everything.

"""
Takes true data and model data as 1D numpy arrays, plotting both as contour field overlays on a world map.
The optional label parameter is appended directly onto the titles. Be aware a space at the start of the string may be needed.
Shows then returns the figure.
"""


def plotOnMap(trueData, modelData, label=''):
    # First generate a bunch of points across globe and turn them into a grid
    tlat = np.array(range(-96, 96, 3))
    tlon = np.array(range(0, 361, 15))
    lati, loni = np.meshgrid(tlat, tlon)

    # Interpolate data to get values at all points in the grid we just generated
    trueDataGrid = griddata(data[:, :2], trueData,
                            (lati, loni), method='linear')
    modelDataGrid = griddata(
        data[:, :2], modelData, (lati, loni), method='linear')

    # Now for the actual plot creation
    # First generate the figure with two subplots
    fig, axes = plt.subplots(
        nrows=2, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})  # setting central_longitude to 180 to get the tapered bit of the field off to one end
    gl = []  # this will hold the gridline object we create for each axe
    # prevent title overlapping with x ticks of previous plot
    plt.subplots_adjust(hspace=0.3)
    data_crs = ccrs.PlateCarree()  # We use this in a couple of places

    axes[0].set_global()  # show whole projection
    axes[0].coastlines()  # with coastlines
    axes[0].set_title(f'Actual data{label}')
    # add gridlines -> this causes axes ticks (via draw_labels) and gridlines to display
    gl.append(axes[0].gridlines(draw_labels=True))
    # then hide the gridlines because we actually only wanted ticks and they're frankly ugly
    gl[0].xlines = False
    gl[0].ylines = False
    # hide x-ticks along top for cleanliness and to save space
    gl[0].top_labels = False

    # Repeat that all for the model data. Copy-paste is a bit inelegant, but a simple solution.
    axes[1].set_global()
    axes[1].coastlines()
    axes[1].set_title(f'Model data{label}')
    gl.append(axes[1].gridlines(draw_labels=True))
    gl[1].xlines = False
    gl[1].ylines = False
    gl[1].top_labels = False

    # find min and maxes for plots so we can apply the colour map identically
    combined = np.append(trueData, modelData)
    dataMin = np.min(combined)
    dataMax = np.max(combined)

    # Plot! Notice the vmin/vmax so their colour maps are scaled over the same range, and the transform so that our input coordinates are converted to match the projection
    axes[0].contourf(loni, lati, trueDataGrid, 20,
                     vmin=dataMin, vmax=dataMax, transform=data_crs)
    axes[1].contourf(loni, lati, modelDataGrid, 20,
                     vmin=dataMin, vmax=dataMax, transform=data_crs)
    axes[0].scatter(longit, latit, marker='.')

    # add a colourbar in space stolen from both subplots
    fig.colorbar(ScalarMappable(norm=Normalize(dataMin, dataMax)), ax=axes)

    # display the figure
    plt.show()

    return fig


# Use the parameters to get a model field.
modelData = kernel @ parameters
modelXData = modelData[:N]
modelZData = modelData[N:]

plotOnMap(X, modelXData, label=" for $B_x$")
plotOnMap(Z, modelZData, label=" for $B_z$")

# Create subplot for 3d quiver plot of model
ax = plt.figure().add_subplot(projection='3d')

# Meshgrid in spherical coordinates
r, theta, phi = np.meshgrid(np.arange(0.05, 0.15, 0.04),
                            # don't start at 0 to avoid division by 0 error
                            np.arange(0.01, pi, 0.2),
                            np.arange(0, 2*pi, 0.2))

# Three function for finding components of B with l <= 2


def Bx(r, theta, phi, g10, g11, h11, g20, g21, h21, g22, h22):
    s1 = R**3/r**2 * g10 * -sin(theta)
    s2 = R**3/r**2 * (g11*cos(phi) + h11*sin(phi)) * cos(theta)
    s3 = R**4/r**3 * g20 * -3*sin(theta)*cos(theta)
    s4 = R**4/r**3 * (g21*cos(phi) + h21*sin(phi)) * \
        sqrt(3)*(cos(theta)**2 - sin(theta)**2)
    s5 = R**4/r**3 * (g22*cos(2*phi) + h22*sin(2*phi)) * \
        sqrt(3)*sin(theta)*cos(theta)

    return ((s1 + s2 + s3 + s4 + s5)/r)


def By(r, theta, phi, g10, g11, h11, g20, g21, h21, g22, h22):
    s1 = R**3/r**2 * g10 * cos(theta)
    s2 = R**3/r**2 * (g11 * -sin(phi) + h11*cos(phi)) * sin(theta)
    s3 = R**4/r**3 * g20 * (1/2)*(3*cos(theta)**2 - 1)
    s4 = R**4/r**3 * (g21 * -sin(phi) + h21*cos(phi)) * \
        sqrt(3)*cos(theta)*sin(theta)
    s5 = R**4/r**3 * (g22*2 * -sin(2*phi) + h22*2 *
                      cos(2*phi)) * sqrt(3)/2*sin(theta)**2

    return -(s1 + s2 + s3 + s4 + s5)/(r*sin(theta))


def Bz(r, theta, phi, g10, g11, h11, g20, g21, h21, g22, h22):
    s1 = -2*R**3/r**3 * g10 * cos(theta)
    s2 = -2*R**3/r**3 * (g11*cos(phi) + h11*sin(phi)) * sin(theta)
    s3 = -3*R**4/r**4 * g20 * (1/2)*(3*cos(theta)**2 - 1)
    s4 = -3*R**4/r**4 * (g21*cos(phi) + h21*sin(phi)) * \
        sqrt(3)*cos(theta)*sin(theta)
    s5 = -3*R**4/r**4 * (g22*cos(2*phi) + h22*sin(2*phi)
                         ) * sqrt(3)/2*sin(theta)**2

    return (s1 + s2 + s3 + s4 + s5)


# Convert grid to Cartesian coordiantes
x = r*cos(phi)*sin(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(theta)

# Find components of vectors at point in grid
dx = Bx(r, theta, phi, *parameters)
dy = By(r, theta, phi, *parameters)
dz = Bz(r, theta, phi, *parameters)

# Plot
ax.quiver(x, y, z, dx, dy, dz, length=0.01, normalize=True)
plt.show()

# Print rounded off derived values
print(derived['m0'])
print(derived['m1'])
print(derived['m'])
print(derived['I'])
print(derived['Ie'])

# Convert some to lat/long


def latitude(theta):
    return 90 + theta*180/pi # flipped sign here because theta is negative


def longitude(phi):
    # I is negative so it points to the opposite side of the globe than we'd expect.
    # To componsate I am rotating longitudes by 180 from what you'd expect
    return phi*180/pi + 180



print('Intersect surface at {} deg lat and {} deg long'.format(
    latitude(derived['I']), longitude(derived['Ie'])))

print('Intersect surface at {} deg lat and {} deg long'.format(
    -latitude(derived['I']), longitude(derived['Ie']-pi)))

print('Offset: {}'.format(R*gauss['g20']/2/gauss['g10']))