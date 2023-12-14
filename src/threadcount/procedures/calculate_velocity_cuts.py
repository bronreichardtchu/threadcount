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
    search_line = [x for x in comment_lines if x.startswith(search_string)][0]
    return eval(search_line[len(search_string) :].strip().replace(" ", ","))

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
        center_wavelength + wavelength_range[1],
        unit_wave=units.angstrom
        )

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
    try:
        new_spec = spec.data - baseline_fit.best_fit
    except:
        new_spec = spec.data

    return baseline_fit, new_spec




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
    #gauss = models.gaussianH(wave, height=height, center=center, sigma=sigma)
    gmodel = models.GaussianModelH()
    params = gmodel.make_params(height=height, center=center, sigma=sigma)
    gauss = gmodel.eval(params, x=wave)


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

def sigma_to_vel_disp(gal_sigma, gal_center):
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

def get_velocity_band_flux(index, vel_vec, wave, residuals, vel_start, vel_end):
    """
    Gets the flux in a velocity band using the residuals from a single emission
    line, from vel_start to vel_end
    e.g. for the "Fountain gas" this will be from gal_sigma_vel to v_esc

    Parameters
    ----------
    index : list of int
        the index location of the emission line in the data cube [i,j]
    vel_vec : :obj:'~numpy.ndarray'
        Vector of velocities
    wave : :obj:'~numpy.ndarray'
        Vector of wavelengths
    residuals : :class:`mpdaf.obj.spectrum.Spectrum` object
        The object containing the residuals after the gaussian was subtracted,
        with the data and variance arrays as properties
    vel_start : float
        The start velocity of the band
    vel_end : float
        The end velocity of the band

    Returns
    -------
    vel_band_flux : float
        Emission line flux from gas within the velocity band
    vel_band_sn : float
        The signal to noise of the velocity band
    """
    #mask flux not in the velocity band
    vel_band_mask = (vel_vec < -vel_start) & (vel_vec > -vel_end)
    residuals_masked = ma.masked_where(~vel_band_mask, residuals.data[:,index[0],index[1]])
    residuals_var_masked = ma.masked_where(~vel_band_mask, residuals.var[:,index[0],index[1]])

    #get the wavelength range we're integrating over
    try:
        dlam = (wave[vel_band_mask][-1] - wave[vel_band_mask][0])
    except IndexError:
        dlam = 0*units.Angstrom

    #integrate up everything between vel=vel_start and vel=vel_end
    #this is in 10^-16 erg/cm^2/s/A, so multiplying by dlam puts it in units
    #of 10^-16 erg/cm^2/s
    vel_band_flux = np.nansum(residuals_masked, axis=0)*dlam

    #calculate the variance
    vel_band_var = np.nansum(residuals_var_masked, axis=0) * dlam**2

    #get the median value and divide by noise to get S/N
    #vel_band_sn = np.nanmedian(residuals_masked, axis=0)/average_noise
    #vel_band_sn = np.nanmedian(residuals_masked, axis=0)/np.nanmedian(np.sqrt(np.abs(residuals_var_masked)), axis=0)
    vel_band_sn = vel_band_flux/np.sqrt(np.abs(vel_band_var))

    return vel_band_flux, vel_band_var, vel_band_sn

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


def determine_v_end(vel_vec, spec, residuals, gal_sigma_vel=60, noise_level=3.0):
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
    gal_sigma_vel : float
        The average velocity dispersion of the galaxy disk
    noise_level : float
        The multiple of the noise level to create the threshold

    Return
    ------
    v_end : float
        the velocity at which the flux residuals reach the same level as the noise
    """
    #check if the residuals are a fully masked array
    if residuals.mask.all() == True:
        average_noise = np.nanstd(spec.subspec(lmin=4700, lmax=4800).data)
        v_end = 0.0 * (units.km/units.s)

    else:
        #calculate the standard deviation of the continuum
        average_noise = np.nanstd(spec.subspec(lmin=4700, lmax=4800).data)
        threshold = noise_level * average_noise

        #find where the residual falls below this threshold
        below_threshold = residuals.data < threshold

        #find for which velocities this is true on the blue side
        #vels_below_threshold = vel_vec[(vel_vec<0) & (below_threshold)]
        vels_below_threshold = vel_vec[(vel_vec < -gal_sigma_vel) & (below_threshold)]

        #find the maximum
        try:
            v_end = np.nanmax(vels_below_threshold, axis=0) * (units.km/units.s)
        except ValueError:
            v_end = 0.0 * (units.km/units.s)

        #if the v_end is larger than 1000km/s, it's almost certainly noise
        #so make that the upper limit
        if v_end.value < -1000:
            v_end = -1000 * (units.km/units.s)

    return v_end, average_noise

def calculate_escape_velocity_pointsource(radius, mass, z):
    """
    Calculates the escape velocity given a radius and mass (assuming point source)
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

def calculate_escape_velocity(mass):
    """
    Calculates the escape velocity given stellar mass
        v_esc = 3 v_circ

    Parameters
    ----------
    mass : float
        The stellar mass of the galaxy in solar masses (without units)

    Returns
    -------
    v_esc : astropy.units.quantity.Quantity
        The escape velocity in km/s (includes the units)
    """
    #give mass units
    #mass = mass * units.solMass

    #for rotation dominated galaxies in SAMI from Tiley+18
    log_vcirc = 2 + (np.log10(mass)-9.66)/4
    vcirc = 10**log_vcirc

    #calculate escape velocity
    v_esc = 3 * vcirc

    #give the escape velocity units
    v_esc = v_esc * units.km/units.s

    return v_esc

def calculate_high_velocity(mass):
    """
    Calculates the high velocity cutoff given a stellar mass
        v_esc = 3 v_circ

    Parameters
    ----------
    mass : float
        The stellar mass of the galaxy in solar masses (without units)

    Returns
    -------
    v_high : astropy.units.quantity.Quantity
        The high velocity cutoff in km/s (includes the units)
    """
    #give mass units
    #mass = mass * units.solMass

    #for rotation dominated galaxies in SAMI from Tiley+18
    log_vcirc = 2 + (np.log10(mass)-9.66)/4
    vcirc = 10**log_vcirc

    #calculate escape velocity
    v_high = 1.5 * vcirc

    #give the escape velocity units
    v_high = v_high * units.km/units.s

    return v_high
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
    #gal_gauss = models.gaussianH(wave, height=gal_height[i,j], center=gal_center[i,j], sigma=gal_sigma[i,j])
    gmodel = models.GaussianModelH()
    params = gmodel.make_params(height=gal_height[i,j], center=gal_center[i,j], sigma=gal_sigma[i,j])
    gal_gauss = gmodel.eval(params, x=wave)

    #flow_gauss = models.gaussianH(wave, height=flow_height[i,j], center=flow_center[i,j], sigma=flow_sigma[i,j])
    gmodel = models.GaussianModelH()
    params = gmodel.make_params(height=flow_height[i,j], center=flow_center[i,j], sigma=flow_sigma[i,j])
    flow_gauss = gmodel.eval(params, x=wave)

    #create the interpolated model
    try:
        model_x = np.linspace(gal_center[i,j]-15, gal_center[i,j]+15,500)
        model_mask = (wave>gal_center[i,j]-15) & (wave<gal_center[i,j]+15)
        model_interp = np.interp(model_x, wave[model_mask], gal_gauss[model_mask]+flow_gauss[model_mask]+const[i,j])
    except ValueError:
        model_x = np.linspace(gal_center[i,j]-15, gal_center[i,j]+15,500)
        model_interp = np.zeros_like(model_x)

    #plot the things
    fig = plt.figure()

    try:
        plt.step(wave, spec, where='mid', c='k', label='data')

        plt.step(wave, gal_gauss+const[i,j], where='mid', c='g', ls='--', label='galaxy gaussian')
        plt.step(wave, flow_gauss+const[i,j], where='mid', c='b', ls='--', label='outflow gaussian')

        #plt.plot(wave, gal_gauss+flow_gauss+const[i,j], c='grey', ls=':', label='model fit')
        plt.plot(model_x, model_interp, c='grey', ls=':', label='total model fit')

        #plt.step(wave, residuals, where='mid', c='r', label='data - galaxy')

        plt.xlim(gal_center[i,j]-10, gal_center[i,j]+10)

        plt.xlabel('Wavelength [$\AA$]')

    except ValueError:
        plt.text(0.5, 0.5, "No Fits", horizontalalignment="center", verticalalignment="center", transform=plt.gca().transAxes, fontsize=14)

    plt.title('Data and Galaxy Gaussian-subtracted residual ('+str(i)+', '+str(j)+')')

    plt.legend(fontsize='small', frameon=False)

    #plt.show()

    return fig


def plot_residuals_vel_space(vel_vec, spec, residuals, gal_sigma_vel, v_esc=300, v_end=600):
    """
    Plots the data and the residuals in velocity space with the three velocity
    bands highlighted

    Parameters
    ----------
    wave : :obj:'~numpy.ndarray'
        wavelength vector
    spec : :obj:'~numpy.ndarray'
        the spectrum (same length as wave)
    residuals : :obj:'~numpy.ndarray'
        the residuals from spec - Gaussian (same length as wave)
    gal_sigma_vel : float
        the average disk velocity dispersion for the galaxy
    v_esc : float, optional
        the escape velocity of the galaxy (Default = 300)
    v_end : float, optional
        the velocity where the emission line flux disappears into the continuum
        noise (Default=600)

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        a plot of the residuals from the data - Gaussian in velocity space with
        the three velocity bands highlighted
    """
    #calculate the standard deviation of the continuum
    threshold = 3*np.nanstd(spec.subspec(lmin=4700, lmax=4800).data)

    #make the plot
    fig = plt.figure()

    plt.axhspan(-threshold, threshold, color='grey', alpha=0.3, label='Noise Level')

    plt.step(vel_vec, spec.data, where='mid', c='k', label='data')

    plt.step(vel_vec, residuals.data, where='mid', c='r', label='residuals')

    #plot where the escape velocity and sigma values are
    plt.axvspan(-gal_sigma_vel, gal_sigma_vel, color='blue', alpha=0.3, label='Within the disk')

    if v_end > v_esc:
        plt.axvspan(-v_esc, -gal_sigma_vel, color='green', alpha=0.3, label='Fountain Gas')
        plt.axvspan(-v_end, -v_esc, color='yellow', alpha=0.3, label=r'Above $v_{esc}=$'+f' {v_esc:.2f}km/s')
    else:
        plt.axvspan(-v_end, -gal_sigma_vel, color='green', alpha=0.3, label='Truncated Mid Velocity')

    plt.axvline(0, c='grey', ls=':')

    print(gal_sigma_vel)

    plt.axvline(-gal_sigma_vel, c='grey', ls=':')

    plt.axvline(-v_esc, c='grey', ls=':')

    plt.xlim(-1000, 1000)

    plt.xlabel('Velocity [km/s]')

    plt.title('Data and Galaxy Gaussian-subtracted residual in Velocity Space')

    plt.legend(fontsize='small', frameon=False)

    return fig




#-------------------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------------------

def main(cube, tc_filename, baseline_fit_range=[], baseline_fit_type=None, v_esc=300*units.km/units.s, disk_sigma=60*units.km/units.s, total_vel_start=0.0*units.km/units.s, line=lines.L_Hb4861, monitor_pixels=[], plot_fits=False):
    """
    Runs the whole thing

    Parameters
    ----------
    cube : :class:`mpdaf.obj.Cube`
        the data cube
    tc_filename : str
        The file name and location of where the threadcount dictionary results
        were saved
    baseline_fit_range : list [[[float, float], [float, float]]], optional
        The range of wavelengths to use in the baseline subtraction.  A list of
        [[[left_begin, left_end], [right_begin, right_end]]]
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
    total_vel_start : float, :obj: `astropy.units.quantity.Quantity`
        The velocity at which to start counting flux for the total residual flux,
        including units
        Default is 0.0 km/s.  Could also use the disk_sigma.
    line : :obj: `threadcount.lines.Line`
        A threadcount emission line object with the information about line centre,
        line name, etc.
    monitor_pixels : list
        A list of pixels to monitor - if this is empty, then the code iterates
        through the entire cube.  If there are pixels listed in the form
        [(x1,y1),(x2,y2)] then only the results for these pixels are calculated.
        Default is an empty list.
    plot_fits : boolean
        Whether to plot all the different fits.  This will plot everything, and
        you need to close the figures to run the next iteration, so make this
        False if doing the entire cube.  Default is False.

    Returns
    -------
    residuals : :obj: `mpdaf.obj.cube.Cube`
        An mpdaf Cube with the results of the emission line minus Gaussian
    vel_cuts_dict : dictionary
        A dictionary of the results, containing numpy arrays of the:
        - total_residual_flux : total flux within the residual
        - total_residual_sn : signal-to-noise ratio of the residual
        - disk_flux : emission line flux from gas which is likely remaining
        within the galaxy disk
        - disk_residual_sn : signal-to-noise ratio of the flux within the disk
        - low_velocity_outflow : flux from gas which is likely above the plane
        of the galaxy, but not reaching high enough velocities to escape
        - low_velocity_sn : signal-to-noise ratio of the flux from the low
        velocity outflow gas
        - high_velocity_outflow : flux from gas which is likely reaching
        velocities that enable it to escape the galaxy
        - high_velocity_sn : signal-to-noise ratio of the flux from the high
        velocity outflow gas
    """
    #read in the threadcount results
    gal_dict, wcs_step, z = read_in_threadcount_dict(tc_filename)

    #create a subcube with a shorter wavelength range
    subcube = create_subcube(cube, center_wavelength=line.center)

    #create an array to put the gaussian-subtracted data in
    #residuals = np.zeros_like(subcube.data)
    #by cloning the subcube - creates a new object with the same shape and
    #coordinates as the subcube, but the .data and .var arrays are set to None
    #by default.
    residuals = subcube.clone(data_init=np.zeros, var_init=np.zeros)
    residuals.var[:,:,:] = subcube.var

    total_residual_flux = np.zeros_like(subcube[0,:,:].data)
    total_residual_var = np.zeros_like(subcube[0,:,:].data)
    total_residual_sn = np.zeros_like(subcube[0,:,:].data)

    disk_turb_flux = np.zeros_like(subcube[0,:,:].data)
    fountain_flux = np.zeros_like(subcube[0,:,:].data)
    escape_flux = np.zeros_like(subcube[0,:,:].data)

    disk_turb_var = np.zeros_like(subcube[0,:,:].data)
    fountain_var = np.zeros_like(subcube[0,:,:].data)
    escape_var = np.zeros_like(subcube[0,:,:].data)

    disk_turb_sn = np.zeros_like(subcube[0,:,:].data)
    fountain_sn = np.zeros_like(subcube[0,:,:].data)
    escape_sn = np.zeros_like(subcube[0,:,:].data)

    #get the centre values
    gal_center, gal_center_err, flow_center, flow_center_err = calc_sfr.get_arrays(gal_dict, var_string='center')

    #get the height values
    gal_height, gal_height_err, flow_height, flow_height_err = calc_sfr.get_arrays(gal_dict, var_string='height')

    #get the sigma values
    gal_sigma, gal_sigma_err, flow_sigma, flow_sigma_err = calc_sfr.get_arrays(gal_dict, var_string='sigma')

    #get the constant values
    const, const_err = gal_dict['avg_c'], gal_dict['avg_c_err']

    #if monitor_pixels is an empty list, add all of the pixels to the list
    if len(monitor_pixels) == 0:
        for x in np.arange(subcube.data.shape[1]):
            for y in np.arange(subcube.data.shape[2]):
                monitor_pixels.append((x,y))

    #iterating through the data array
    for i in np.arange(subcube.data.shape[1]):
        for j in np.arange(subcube.data.shape[2]):
            #only run on pixels in monitor_pixels
            if (i,j) in monitor_pixels:
                #get the individual spectrum
                this_spec = subcube[:,i,j]

                #subtract the baseline from the data
                if baseline_fit_type is not None:
                    this_baseline_range = baseline_fit_range[0]
                    baseline_fit, new_spec = subtract_baseline(this_spec, this_baseline_range, baseline_fit_type)
                    this_spec.data = new_spec

                    if plot_fits == True:
                        #plot the fit
                        fig1 = fit.plot_baseline(baseline_fit)
                        plt.gca().set_xlim(this_baseline_range[0][0]-5, this_baseline_range[1][1]+5)

                #subtract the gaussian from the data
                residuals.data[:,i,j] = subtract_gaussian(this_spec.wave.coord(), this_spec, gal_height[i,j], gal_center[i,j], gal_sigma[i,j], const=const[i,j])

                #transform from wavelength to velocity space
                vel_vec = wave_to_vel(this_spec.wave.coord(), gal_center[i,j])

                #calculate where the residual disappears into the noise
                #need to use pre-baseline subtracted data
                v_end, average_noise = determine_v_end(vel_vec, subcube[:,i,j], residuals.data[:,i,j], gal_sigma_vel=disk_sigma)
                v_end = abs(v_end)

                if plot_fits == True:
                    #plot the data minus galaxy
                    fig2 = plot_data_minus_gal(this_spec.wave.coord(), this_spec.data, residuals.data, gal_dict, i, j)

                    #plot the data and residuals in velocity space
                    fig3 = plot_residuals_vel_space(vel_vec, this_spec, residuals[:,i,j], disk_sigma.value, v_esc=v_esc.value, v_end=v_end.value)

                    plt.show()

                #do the flux cuts calculation
                #disk_turb_flux[i,j], fountain_flux[i,j], escape_flux[i,j] = get_velocity_bands(vel_vec.value, this_spec.wave.coord(), residuals.data[:,i,j], disk_sigma.value, v_esc=v_esc.value, v_end=v_end)

                disk_turb_flux.data[i,j], disk_turb_var.data[i,j], disk_turb_sn.data[i,j] = get_velocity_band_flux([i,j], vel_vec.value, this_spec.wave.coord(), residuals, disk_sigma.value, -disk_sigma.value)

                total_residual_flux.data[i,j], total_residual_var.data[i,j], total_residual_sn.data[i,j] = get_velocity_band_flux([i,j], vel_vec.value, this_spec.wave.coord(), residuals, total_vel_start.value, v_end.value)

                #if the flux disappears into the noise after v_esc
                if v_end.value > v_esc.value:
                    fountain_flux.data[i,j], fountain_var.data[i,j], fountain_sn.data[i,j] = get_velocity_band_flux([i,j], vel_vec.value, this_spec.wave.coord(), residuals, disk_sigma.value, v_esc.value)

                    escape_flux.data[i,j], escape_var.data[i,j], escape_sn.data[i,j] = get_velocity_band_flux([i,j], vel_vec.value, this_spec.wave.coord(), residuals, v_esc.value, v_end.value)

                else:
                    fountain_flux.data[i,j], fountain_var.data[i,j], fountain_sn.data[i,j] = get_velocity_band_flux([i,j], vel_vec.value, this_spec.wave.coord(), residuals, disk_sigma.value, v_end.value)

                    escape_flux.data[i,j], escape_var.data[i,j], escape_sn.data[i,j] = 0.0, 0.0, 0.0

    #save the results in a dictionary
    vel_cuts_dict = fit.ResultDict(data_dict={
                        "total_residual_flux" : total_residual_flux.data,
                        "total_residual_var" : total_residual_var.data,
                        "total_residual_sn" : total_residual_sn.data,
                        "disk_flux" : disk_turb_flux.data,
                        "disk_var" : disk_turb_var.data,
                        "disk_residual_sn" : disk_turb_sn.data,
                        "low_velocity_outflow" : fountain_flux.data,
                        "low_velocity_var" : fountain_var.data,
                        "low_velocity_sn" : fountain_sn.data,
                        "high_velocity_outflow" : escape_flux.data,
                        "high_velocity_var" : escape_var.data,
                        "high_velocity_sn" : escape_sn.data,
                        },
                        comment = f"Total residual measured from {total_vel_start:.0g} to v_end\n"+
                        "units: 10^-16 erg / (cm^2 s)\n"
                    )

    return residuals, vel_cuts_dict
