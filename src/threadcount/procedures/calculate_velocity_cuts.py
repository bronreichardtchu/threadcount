"""
NAME:
	calculate_velocity_cuts.py

FUNCTIONS INCLUDED:


"""
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy import units
from astropy.constants import G
from astropy.cosmology import WMAP9 as cosmo

from threadcount import fit
from threadcount import models
from threadcount import lines
from threadcount.procedures import calculate_star_formation_rate as calc_sfr

import importlib
importlib.reload(fit)



#-------------------------------------------------------------------------------
# READ IN DATA
#-------------------------------------------------------------------------------

def fits_read_in(filename):
    """
    Reads in the data fits file, assuming that the data and variance are in the
    same file with indices 0 and 1 respectively

    Parameters
    ----------
    filename : str
        the filename and location of the fits file

    Returns
    -------
    :class:`mpdaf.obj.Cube`
        A data cube.
    """
    cube = fit.open_fits_cube(filename, data_hdu_index=0, var_filename=filename, var_hdu_index=1)

    return cube

def get_wave_vector(cube, z=None):
    """
    Gets the wavelength vector from the mpdaf cube object and deredshifts it

    Parameters
    ----------
    cube : :class:`mpdaf.obj.Cube`
        the data cube
    z : float or None, optional
        the redshift (Default is None)

    Returns
    -------
    :class:`mpdaf.obj.Cube`
        the de-redshifted cube
    """
    wave = cube.wave.coord()

    if z is not None:
        #wave = wave/(1+z)
        cube.wave.set_crval(cube.wave.get_crval()/(1+z))
        cube.wave.set_step(cube.wave.get_step()/(1+z))

    return cube



def read_in_threadcount_dict(filename):
    """
    Reads in the threadcount output as a dictionary

    Parameters
    ----------
    filename : str
        the file name and location of the threadcount results saved in a dictionary

    Returns
    -------
    gal_dict : dict
        the threadcount results in a dictionary
    wcs_step : list [float, float]
        the step in the WCS from the threadcount results header
    z : float
        the redshift from the threadcount results header
    """
    #read in the dictionary
    gal_dict = fit.ResultDict.loadtxt(filename)

    #get the comment lines
    comment_lines = gal_dict.comment.split('\n')

    #get the WCS from the comment lines
    wcs_step = extract_from_comments(comment_lines, 'wcs_step:')

    #get the redshift from the comment lines
    z = extract_from_comments(comment_lines, 'z_set:')

    return gal_dict, wcs_step, z


def extract_from_comments(comment_lines, search_string):
    """
    Extracts info from the comments in the threadcount output text file
    Copied from a extract_wcs() in analyze_outflow_extent.py

    Parameters
    ----------
    comment_lines : str
        the commented header lines from the threadcount results header
    search_string : str
        the variable to search for (e.g. "wcs_step:", "z_set:")

    Returns
    -------
    The value or list indicated in the threadcount output file by the
    search_string
    """
    #search_string = "wcs_step:"
    wcs_line = [x for x in comment_lines if x.startswith(search_string)][0]
    return eval(wcs_line[len(search_string) :].strip().replace(" ", ","))

#-------------------------------------------------------------------------------
# SUBTRACT BASELINE
#-------------------------------------------------------------------------------

def create_subcube(cube, center_wavelength=lines.Hb4861, wavelength_range=(-150,150)):
    """
    Creates a subcube centred on the emission line

    Parameters
    ----------
    cube : :class:`mpdaf.obj.Cube`
        A datacube containing the wavelength range set in these parameters
    center_wavelength : float, optional
        The center wavelength of the emission line to fit, by default :const:`threadcount.lines.Hb4861`
    wavelength_range : array-like [float, float], optional
        The wavelength range to fit, in Angstroms. These are defined as a change
        from the `center_wavelength`, by default (-150, 150)

    Returns
    -------
    subcube : :class:`mpdaf.obj.Cube`
        A subset of the input datacube centred on the input emission line
    """
    subcube = cube.select_lambda(
        center_wavelength + wavelength_range[0],
        center_wavelength + wavelength_range[1])

    return subcube


def subtract_baseline(spec, this_baseline_range, baseline_fit_type):
    """
    Fits and subtracts the baseline from the spectrum

    Parameters
    ----------
    spec : `mpdaf.obj.spectrum.Spectrum`
        An mpdaf spectrum with the data to subtract the baseline from
    this_baseline_range : list
        A list of [[left_begin, left_end],[right_begin, right_end]]
        Describes the wavelength range of data to use in fitting the baseline
    baseline_fit_type : str or None
        Options: None, "linear", "quadratic"

    Returns
    -------
    baseline_fit : `lmfit.model.ModelResult`
        The lmfit fitted model class for the baseline
    new_spec : `numpy.ma.core.MaskedArray`
        A numpy masked array of the data minus the baseline fit
    """
    #create the fit
    baseline_fit = fit.fit_baseline(
        spec,
        this_baseline_range=this_baseline_range,
        baseline_fit_type=baseline_fit_type)

    #subtract the best fit from the data
    new_spec = spec.data - baseline_fit.best_fit

    return baseline_fit.best_fit, new_spec




#-------------------------------------------------------------------------------
# SUBTRACT CENTRAL LINE
#-------------------------------------------------------------------------------

def subtract_gaussian(wave, spec, height, center, sigma, const=None):
    """
    Subtracts the fitted Gaussian from the data

    Parameters
    ----------
    wave : :obj:'~numpy.ndarray'
        Array with the wavelength vector
    spec : `mpdaf.obj.spectrum.Spectrum`
        An mpdaf spectrum with the data to subtract the Gaussian from
    height : float
        The height of the Gaussian
    center : float
        The central wavelength of the Gaussian in Angstroms
    sigma : float
        The dispersion of the Gaussian in Angstroms
    const : float, optional
        A constant to add to the Gaussian in case there's a constant continuum
        level to take care of

    Returns
    -------
    residuals : :class: `mpdaf.obj.spectrum.Spectrum`
        The residual of the Gaussian subtracted from the data
    """
    #get the gaussian
    gauss = models.gaussianH(wave, height=height, center=center, sigma=sigma)

    #add the constant
    if const is not None:
        gauss = gauss + const

    #subtract from the data
    residuals = spec - gauss

    return residuals




#-------------------------------------------------------------------------------
# CONVERT TO VELOCITY SPACE
#-------------------------------------------------------------------------------

def wave_to_vel(wave, center):
    """
    Converts the wavelength to the velocity

    Parameters
    ----------
    wave : :obj:'~numpy.ndarray'
        Vector of wavelengths
    center : float
        The central value fit for the narrow galaxy gaussian

    Returns
    -------
    vel_vector : :obj:'~numpy.ndarray'
        Vector of velocities
    """
    #minus the central wavelength off the wavelength vector
    wave = wave - center

    #do c*wave/center
    c = 299792.458 * (units.km/units.s)
    vel_vector = c * wave/center

    return vel_vector

#-------------------------------------------------------------------------------
# VELOCITY BANDS
#-------------------------------------------------------------------------------

def get_velocity_bands(vel_vec, residuals, gal_center, gal_sigma, v_esc):
    """
    Converts the sigma from the fits to the velocity dispersion

    Parameters
    ----------
    gal_sigma : float or :obj:'~numpy.ndarray'
        Vector of sigmas
    gal_center : float or :obj:'~numpy.ndarray'
        The central value fit for the narrow galaxy gaussian

    Returns
    -------
    vel_disp: float or :obj:'~numpy.ndarray'
        Vector of velocity dispersions
    """
    #convert the galaxy sigma to velocity space
    #do c*wave/center
    c = 299792.458 * (units.km/units.s)
    gal_sigma_vel = c * gal_sigma/gal_center

    return gal_sigma_vel

#-------------------------------------------------------------------------------
# VELOCITY BANDS
#-------------------------------------------------------------------------------

def get_velocity_bands(vel_vec, wave, residuals, gal_sigma_vel, v_esc, v_end):
    """
    Gets the flux in each velocity band (disk, fountain and escaping) from a
    single emission line

    Parameters
    ----------
    vel_vec : :obj:'~numpy.ndarray'
        Vector of velocities
    wave : :obj:'~numpy.ndarray'
        Vector of wavelengths
    residuals : :obj:'~numpy.ndarray'
        Vector of residuals (make sure this is the numpy array, not the
        :class:`mpdaf.obj.spectrum.Spectrum` object)
    gal_sigma_vel : float
        The average velocity dispersion of the galaxy disk
    v_esc : float
        The escape velocity
    v_end : float
        The velocity where the flux of the emission line disappears into the noise

    Returns
    -------
    disk_turb_flux : float
        Emission line flux from gas which is likely remaining within the galaxy
        disk
    fountain_flux : float
        Flux from gas which is likely above the plane of the galaxy, but not
        reaching high enough velocities to escape
    escape_flux : float
        Flux from gas which is likely reaching velocities that enable it to escape
        the galaxy
    """
    #Disk Turbulence
    #add up everything between vel=0 and vel=gal_sigma
    disk_turb_mask = (vel_vec > -gal_sigma_vel) & (vel_vec < gal_sigma_vel)
    residuals_masked = ma.masked_where(~disk_turb_mask, residuals)
    try:
        dlam = (wave[disk_turb_mask][-1] - wave[disk_turb_mask][0])
    except IndexError:
        dlam = 0*units.Angstrom
    disk_turb_flux = np.nansum(residuals_masked, axis=0)*dlam

    #if the flux disappears into the noise after the escape velocity
    if v_end > v_esc:
        #Fountain Gas
        #add up everything between vel=gal_sigma and vel=escape vel
        fountain_mask = (vel_vec < -gal_sigma_vel) & (vel_vec > -v_esc)
        residuals_masked = ma.masked_where(~fountain_mask, residuals)
        try:
            dlam = (wave[fountain_mask][-1] - wave[fountain_mask][0])
        except IndexError:
            dlam = 0*units.Angstrom
        fountain_flux = np.nansum(residuals_masked, axis=0)*dlam

        #Escaping gas
        #add up everything between the escape velocity and where the flux reaches
        #the noise level
        escape_mask = (vel_vec < -v_esc) & (vel_vec > -v_end)
        residuals_masked = ma.masked_where(~escape_mask, residuals)
        try:
            dlam = (wave[escape_mask][-1] - wave[escape_mask][0])
        except IndexError:
            dlam = 0
        escape_flux = np.nansum(residuals_masked, axis=0)*dlam

    #otherwise, we need to truncate the fountain gas and set the escape flux
    #to zero
    else:
        #Fountain Gas
        #add up everything between vel=gal_sigma and vel=end vel
        fountain_mask = (vel_vec < -gal_sigma_vel) & (vel_vec > -v_end)
        residuals_masked = ma.masked_where(~fountain_mask, residuals)
        try:
            dlam = (wave[fountain_mask][-1] - wave[fountain_mask][0])
        except IndexError:
            dlam = 0
        fountain_flux = np.nansum(residuals_masked, axis=0)*dlam

        #Escaping gas
        #add up everything between the escape velocity and where the flux reaches the standard deviation
        escape_flux = 0.0

    return disk_turb_flux, fountain_flux, escape_flux


def determine_v_end(vel_vec, spec, residuals):
    """
    Finds where the residuals reach the same level as the noise
    Returns the velocity at which this happens

    Parameters
    ----------
    vel_vec : :obj:'~numpy.ndarray'
        Vector of wavelengths
    spec : `mpdaf.obj.spectrum.Spectrum`
        An mpdaf spectrum with the data
        This should be from BEFORE any extra baseline subtraction has taken place
        since it is used to find the standard deviation of the continuum, and
        the baseline subtraction alters the shape of the continuum away from the
        emission line.
    residuals : :obj:'~numpy.ndarray'
        The residual array

    Return
    ------
    v_end : float
        the velocity at which the flux residuals reach the same level as the noise
    """
    #check if the residuals are a fully masked array
    if residuals.mask.all() == True:
        v_end = 0.0 * (units.km/units.s)

    else:
        #calculate the standard deviation of the continuum
        threshold = np.nanstd(spec.subspec(lmin=4700, lmax=4800).data)

        #find where the residual falls below this threshold
        below_threshold = residuals.data < threshold

        #find for which velocities this is true on the blue side
        vels_below_threshold = vel_vec[(vel_vec<0) & (below_threshold)]

        #find the maximum
        try:
            v_end = np.nanmax(vels_below_threshold, axis=0) * (units.km/units.s)
        except ValueError:
            v_end = 0.0 * (units.km/units.s)
        #v_end = np.nanmax(vels_below_threshold, axis=0) * (units.km/units.s)

    return v_end

def calculate_escape_velocity(radius, mass, z):
    """
    Calculates the escape velocity given a radius and mass
        v_esc = sqrt(2 G M / r)

    Parameters
    ----------
    radius : float
        The radius of the galaxy in arcseconds (but without units)
        Usually put in the effective radius, but could use any radius
    mass : float
        The mass of the galaxy in solar masses (without units)
        Usually use the stellar mass
    z : float
        The redshift (used to convert the radius from arcseconds to kpc)

    Returns
    -------
    v_esc : astropy.units.quantity.Quantity
        The escape velocity in km/s (includes the units)
    """
    #give mass units
    mass = mass * units.solMass

    #give the radius units
    radius = radius * units.arcsec

    #convert to kpc
    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(units.kpc/units.arcsec)
    radius = radius * proper_dist
    print("radius for escape velocity:", radius)

    #calculate the escape velocity
    v_esc = np.sqrt(2*G*mass/radius)

    #convert to km/s
    v_esc = v_esc.to('km/s')

    return v_esc
#-------------------------------------------------------------------------------
# PLOTS
#-------------------------------------------------------------------------------

def plot_data_minus_gal(wave, spec, residuals, gal_dict, i, j):
    """
    Plot of the two gaussian fit, and the residuals for spaxel (i,j)

    Parameters
    ----------
    wave : :obj:'~numpy.ndarray'
        wavelength vector
    spec : :obj:'~numpy.ndarray'
        the spectrum (same length as wave)
    residuals : :obj:'~numpy.ndarray'
        the residuals from spec - Gaussian (same length as wave)
    gal_dict : dict
        the threadcount results dictionary
    i : int
        the x-index of the spaxel to plot
    j : int
        the y-index of the spaxel to plot

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        a plot with the double Gaussian fit from threadcount, with the residuals
        from the data - Gaussian also plotted
    """
    #get the centre values
    gal_center, gal_center_err, flow_center, flow_center_err = calc_sfr.get_arrays(gal_dict, var_string='center')

    #get the height values
    gal_height, gal_height_err, flow_height, flow_height_err = calc_sfr.get_arrays(gal_dict, var_string='height')

    #get the sigma values
    gal_sigma, gal_sigma_err, flow_sigma, flow_sigma_err = calc_sfr.get_arrays(gal_dict, var_string='sigma')

    #get the constant values
    const, const_err = gal_dict['avg_c'], gal_dict['avg_c_err']

    #create the gaussians
    gal_gauss = models.gaussianH(wave, height=gal_height[i,j], center=gal_center[i,j], sigma=gal_sigma[i,j])

    flow_gauss = models.gaussianH(wave, height=flow_height[i,j], center=flow_center[i,j], sigma=flow_sigma[i,j])

    #create the interpolated model
    model_x = np.linspace(gal_center[i,j]-15, gal_center[i,j]+15,500)
    model_mask = (wave>gal_center[i,j]-15) & (wave<gal_center[i,j]+15)
    model_interp = np.interp(model_x, wave[model_mask], gal_gauss[model_mask]+flow_gauss[model_mask]+const[i,j])

    #plot the things
    plt.figure()

    plt.step(wave, cube.data[:,i,j], where='mid', c='k', label='data')

    plt.step(wave, gal_gauss+const[i,j], where='mid', c='g', ls='--', label='galaxy gaussian')
    plt.step(wave, flow_gauss+const[i,j], where='mid', c='b', ls='--', label='outflow gaussian')

    #plt.plot(wave, gal_gauss+flow_gauss+const[i,j], c='grey', ls=':', label='model fit')
    plt.plot(model_x, model_interp, c='grey', ls=':', label='total model fit')

    plt.step(wave, residuals[:,i,j], where='mid', c='r', label='data - galaxy')

    plt.xlim(gal_center[i,j]-10, gal_center[i,j]+10)

    plt.title('Data and Galaxy Gaussian-subtracted residual ('+str(i)+', '+str(j)+')')

    plt.legend()

    plt.show()






#-------------------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------------------

def main(data_filename, tc_filename, this_baseline_range=[], baseline_fit_type=None, v_esc=300*units.km/units.s, disk_sigma=60*units.km/units.s, line=lines.L_Hb4861):
    """
    Runs the whole thing

    Parameters
    ----------
    data_filename : str
        The file name and location of the data fits file with data in extension
        0 and variance in extension 1
    tc_filename : str
        The file name and location of where the threadcount dictionary results
        were saved
    this_baseline_range : list [[float, float], [float, float]], optional
        The range of wavelengths to use in the baseline subtraction.  A list of
        [[left_begin, left_end], [right_begin, right_end]]
        If baseline_fit_type is not None, this NEEDS TO BE INCLUDED.
    baseline_fit_type : str, optional
        The type of fit to do to the baseline to subtract leftover continuum.
        Options are "quadratic", "linear" or None
        If this is not None, MUST include this_baseline_range parameter.
    v_esc : float, :obj: `astropy.units.quantity.Quantity`
        The escape velocity in km/s (includes the units)
        Default is 300 km/s
    disk_sigma : float, :obj: `astropy.units.quantity.Quantity`
        The average disk velocity dispersion in km/s (includes the units)
        Default is 60 km/s
    line : :obj: `threadcount.lines.Line`
        A threadcount emission line object with the information about line centre,
        line name, etc.

    Returns
    -------
    residuals : :obj: `mpdaf.obj.cube.Cube`
        An mpdaf Cube with the results of the emission line minus Gaussian
    disk_turb_flux : :obj:'~numpy.ndarray' `astropy.units.quantity.Quantity`
        Array of emission line flux from gas which is likely remaining within
        the galaxy disk
    fountain_flux : :obj:'~numpy.ndarray' `astropy.units.quantity.Quantity`
        Array of flux from gas which is likely above the plane of the galaxy,
        but not reaching high enough velocities to escape
    escape_flux : :obj:'~numpy.ndarray' `astropy.units.quantity.Quantity`
        Array of flux from gas which is likely reaching velocities that enable
        it to escape the galaxy
    """
    #read in the data file
    cube = fits_read_in(data_filename)

    #read in the threadcount results
    gal_dict, wcs_step, z = read_in_threadcount_dict(tc_filename)

    #deredshift the wavelength array
    cube = get_wave_vector(cube, z=z)

    #create an array to put the gaussian-subtracted data in
    #and the velocity vectors
    residuals = np.zeros_like(cube.data)
    vel_vecs = np.zeros((cube.data.shape[0], cube.data.shape[1], cube.data.shape[2]))

    #get the centre values
    gal_center, gal_center_err, flow_center, flow_center_err = calc_sfr.get_arrays(gal_dict, var_string='center')

    #get the height values
    gal_height, gal_height_err, flow_height, flow_height_err = calc_sfr.get_arrays(gal_dict, var_string='height')

    #get the sigma values
    gal_sigma, gal_sigma_err, flow_sigma, flow_sigma_err = calc_sfr.get_arrays(gal_dict, var_string='sigma')

    #get the constant values
    const, const_err = gal_dict['avg_c'], gal_dict['avg_c_err']

    #create a subcube with a shorter wavelength range
    subcube = create_subcube(cube)

    #iterating through the data array
    for i in np.arange(sub_cube.data.shape[1]):
        for j in np.arange(sub_cube.data.shape[2]):
            this_spec = sub_cube[:,i,j]

            #subtract the baseline from the data
            if baseline_fit_type is not None:
                baseline_fit, new_spec = subtract_baseline(this_spec, this_baseline_range, baseline_fit_type)
                this_spec = new_spec

            #subtract the gaussian from the data
            residuals[:,i,j] = subtract_gaussian(cube.wave.coord(), cube.data[:,i,j], gal_height[i,j], gal_center[i,j], gal_sigma[i,j], const=const[i,j])

            #transform from wavelength to velocity space
            vel_vecs[:,i,j] = wave_to_vel(cube.wave.coord(), gal_center[i,j])

    #do the flux calculation
    disk_turb_flux, fountain_flux = get_velocity_bands(vel_vecs, residuals, gal_center, gal_sigma, v_esc=300)


    #return residuals, vel_vecs
    return residuals, disk_turb_flux, fountain_flux