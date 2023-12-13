"""
NAME:
	analyse_face_on_galaxies.py

FUNCTIONS INCLUDED:


"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from types import SimpleNamespace

from mpdaf.obj import Image, gauss_image, WCS
from astropy import units
from astropy.constants import G
from astropy.cosmology import WMAP9 as cosmo

import cmasher as cmr

import threadcount as tc
from threadcount.procedures import calculate_star_formation_rate as calc_sfr
from threadcount.procedures import calculate_velocity_cuts as calc_vc
from threadcount.procedures import set_rcParams
from threadcount.procedures import open_cube_and_deredshift



#the inputs that will be used if there are no command line arguments
default_settings = {
    # need the continuum subtracted data fits file for the velocity cuts method
    "data_filename" : "ex_gal_fits_file.fits",
    "two_gauss_mc_input_file" : "ex_4861_mc_best_fit.txt",
    "gal_name" : "example_galaxy_name", #used for labelling plots
    "line_label" : r"H$\beta$", #used for labelling plots
    "z" : 0.03, # redshift
    "monitor_pixels" : [], # pixels to monitor - if this has length greater
    #than zero, then only these fits are created, and the plots are shown
    # now for some default fitting details
    "line" : tc.lines.L_Hb4861,
    "baseline_subtract" : None, # baseline can be "quadratic" or "linear" or None
    # the range of wavelengths to use in the baseline subtraction
    "baseline_fit_range" : [
                        ], # a list of: [[left_begin, left_end],[right_begin, right_end]], one for each line
    # also need the stellar mass of the galaxy
    "stellar_mass" : 10**11.21, # MUST BE INCLUDED
    # we need the escape velocity for the data
    # either this is a given parameter, or threadcount can work it out if
    # you give it data to use to calculate the effective radius
    "escape_velocity" : 456 * units.km/units.s, # put as None if you don't know
    # either give the effective radius, or the threadcount output will
    # be used to find an effective radius, which assumes that the entire
    # galaxy is within the field of view for the IFU data
    "effective_radius" : None, # in arcseconds, used to calculate the escape
    # velocity, so doesn't technically need to be v_50
    # alternatively, you can give another data image (e.g. PANSTARRs) to use
    # to find the effective radius
    "image_data_filename" : "ex_image_file.fits", #or put as None
    # escape_velocity MUST BE NONE IF YOU WANT TO CALCULATE IT FROM THE
    # IMAGE DATA FILE GIVEN ABOVE
    "Av_array_filename" : None, # the location of the file with the
    # extinction Av array - if None, will assume the data has been
    # extinction corrected already when calculating SFR
    "average_disk_sigma" : None, # if None will use the threadcount fits to find average_disk_sigma.
    # output options
    "output_base_name" : "ex_velocity_cuts_results", # saved files will begin with this
    "plot_results" : True, #boolean
    "crop_data" : None, # Use to define how much of the data goes into the maps in the
    # format: [axis1_begin, axis1_end, axis2_begin, axis2_end] or None
    # e.g. [2, -1, 3, -2] will map data[2:-1, 3:-2]
    "shiftx" : None, # hardcoded shift in the x direction for the coord arrays
    #(in arcseconds). If this is none, it finds the maximum point of the flux
    #from the threadcount fits and makes this the centre.  Default is None.
    "shifty" : None, # hardcoded shift in the y direction for the coord arrays
    #(in arcseconds). If this is none, it finds the maximum point of the flux
    #from the threadcount fits and makes this the centre.  Default is None.
}




def run(user_settings):
    #test if the cube has been opened.  If not, open cube and deredshift
    if "cube" not in user_settings.keys():
        user_settings = open_cube_and_deredshift.run(user_settings)
        set_rcParams.set_params({"image.aspect" : user_settings["image_aspect"]})
    # s for settings
    s = tc.fit.process_settings_dict(default_settings, user_settings)

    #print('data filename:', s.data_filename)
    print('tc data filename:', s.two_gauss_mc_input_file)

    # calculate the escape velocity if it wasn't given
    if s.escape_velocity is None:
        # check for a given effective radius
        if s.effective_radius is None:
            #need to calculate the effective radius
            if s.image_data_filename is None:
                #use the threadcount output to find the effective radius
                tc_data, _, _ = calc_vc.read_in_threadcount_dict(s.two_gauss_mc_input_file)

                _, _, rad = data_coords(tc_data, s.z_set, s.wcs_step)

                s.effective_radius = calc_effective_radius_tc(tc_data, rad, flux_percentage=50)

                #get rid of stuff we don't need out of memory
                del tc_data, rad

            else:
                #use the given image data to find the effective radius
                s.effective_radius = calc_effective_radius_fits(s.image_data_filename, fits_ext='COMPRESSED_IMAGE', flux_percentage=50)

        #now use that to calculate the escape velocity
        #this uses the assumed redshift from settings, or if you've used tc_data
        #to calculate the effective_radius it uses the z found in threadcount
        print("redshift:", s.z_set)
        #s.escape_velocity = calc_vc.calculate_escape_velocity(s.effective_radius, s.stellar_mass, s.z_set)
        s.escape_velocity = calc_vc.calculate_high_velocity(s.stellar_mass)

    if s.average_disk_sigma is None:
        #use the threadcount output to find the average disk sigma
        tc_data, _, _ = calc_vc.read_in_threadcount_dict(s.two_gauss_mc_input_file)

        #get the fitted galaxy sigma
        gal_sigma, _, _, _ = calc_sfr.get_arrays(tc_data, var_string='sigma')

        gal_center, _, _, _ = calc_sfr.get_arrays(tc_data, var_string='center')

        #the sigma is in Angstroms, need to convert to km/s
        gal_sigma_vel = calc_vc.sigma_to_vel_disp(gal_sigma, gal_center)

        #take the average
        avg_gal_sigma_vel = np.nanmean(gal_sigma_vel)

        s.average_disk_sigma = avg_gal_sigma_vel

        #get rid of stuff we don't need out of memory
        del tc_data, gal_sigma, gal_center, gal_sigma_vel



    if len(s.monitor_pixels) == 0:
        residuals, vel_cuts_dict = calc_vc.main(
            cube = s.cube,
            tc_filename = s.two_gauss_mc_input_file,
            baseline_fit_range = s.baseline_fit_range,
            baseline_fit_type = s.baseline_subtract,
            v_esc = s.escape_velocity, disk_sigma=s.average_disk_sigma, line=s.line,
            monitor_pixels=[],
            plot_fits=False)
    else:
        residuals, vel_cuts_dict = calc_vc.main(
            cube = s.cube,
            tc_filename = s.two_gauss_mc_input_file,
            baseline_fit_range = s.baseline_fit_range,
            baseline_fit_type = s.baseline_subtract,
            v_esc = s.escape_velocity, disk_sigma=s.average_disk_sigma, line=s.line,
            monitor_pixels=s.monitor_pixels,
            plot_fits=True)


    #print some stuff
    print('escape velocity:', s.escape_velocity)
    print('average disk sigma:', s.average_disk_sigma)
    #give the radius units
    radius = s.effective_radius * units.arcsec
    #convert to kpc
    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(s.z).to(units.kpc/units.arcsec)
    radius = radius * proper_dist
    print("radius for escape velocity:", radius)
    print('Residuals type', type(residuals))
    print('Residuals shape', residuals.shape)
    print('Fountain flux type', type(vel_cuts_dict['low_velocity_outflow']))
    print('Fountain flux shape', vel_cuts_dict['low_velocity_outflow'].shape)
    print('Escape flux shape', vel_cuts_dict['high_velocity_outflow'].shape)

    #save the results - OVERWRITES ANY EXISTING FILES
    residuals.write(s.output_base_name+'_'+str(s.line.label)+'_residuals.fits')

    vel_cuts_dict.savetxt(s.output_base_name+'_'+str(s.line.label)+'_vel_cuts_dict.txt')

    #run through the plotting scripts
    if s.plot_results == True:
        make_plots(s)
        



def make_plots(user_settings):
    #check that the input wasn't a SimpleNamespace
    if type(user_settings) == SimpleNamespace:
        user_settings = user_settings.__dict__

    # s for settings
    s = tc.fit.process_settings_dict(default_settings, user_settings)

    #read in velocity cuts results
    vel_cuts_dict = tc.fit.ResultDict.loadtxt(s.output_base_name+'_'+str(s.line.label)+'_vel_cuts_dict.txt')

    #read in the threadcount results
    tc_data, wcs_step, z = calc_vc.read_in_threadcount_dict(s.two_gauss_mc_input_file)

    s.tc_dict = tc_data
    s.wcs_step = wcs_step

    xx, yy, rad = data_coords(tc_data, z, wcs_step, shiftx=s.shiftx, shifty=s.shifty)

    s.radius_array = rad

    #read in the Av array for the extinction values
    if s.Av_array_filename != None:
        print('Loading Av array')
        Av_array = np.loadtxt(s.Av_array_filename)
    else:
        Av_array = None

    s.Av_array = Av_array

    #if we're not looking at the fits to Hbeta, read in Hbeta so we can calculate SFR
    if "Hb" not in s.line.label:
        tc_data_hbeta, wcs_step, z = calc_vc.read_in_threadcount_dict(s.output_filename+'_4861_mc_best_fit.txt')

    #print some stuff
    gal_flux, gal_flux_err, flow_flux, flow_flux_err = calc_sfr.get_arrays(tc_data, var_string='flux')

    print('=====================')
    print('Galaxy:', s.gal_name)
    print('=====================')
    flux_ratio = np.nansum(vel_cuts_dict['high_velocity_outflow'])/np.nansum(gal_flux)
    print('High velocity gas to galaxy flux ratio:',
                flux_ratio,
                '+/-',
                #calculate the uncertainty
                #flux ratio * sqrt( var/high_vel^2 + (gal_flux_err/gal_flux)^2 )
                (flux_ratio) * np.sqrt(np.nansum(vel_cuts_dict['high_velocity_var'])/np.nansum(vel_cuts_dict['high_velocity_outflow'])**2 + (np.nansum(gal_flux_err)/np.nansum(gal_flux))**2)
                )

    flux_ratio = np.nansum(vel_cuts_dict['low_velocity_outflow'])/np.nansum(gal_flux)
    print('Fountain velocity gas to galaxy flux ratio:',
                flux_ratio,
                '+/-',
                #calculate the uncertainty
                #flux ratio * sqrt( var/low_vel^2 + (gal_flux_err/gal_flux)^2 )
                (flux_ratio) * np.sqrt(np.nansum(vel_cuts_dict['low_velocity_var'])/np.nansum(vel_cuts_dict['low_velocity_outflow'])**2 + (np.nansum(gal_flux_err)/np.nansum(gal_flux))**2)
                )
    print('')

    flux_ratio = np.nansum(vel_cuts_dict['high_velocity_outflow'])/(np.nansum(vel_cuts_dict['low_velocity_outflow'])+np.nansum(vel_cuts_dict['high_velocity_outflow']))
    print('High velocity gas to total outflow flux ratio:',
                flux_ratio,
                '+/-',
                #calculate the uncertainty
                #flux ratio * sqrt( var/high_vel^2 + (var_low+var_high)/(low_vel+high_vel)^2 )
                (flux_ratio) * np.sqrt(np.nansum(vel_cuts_dict['high_velocity_var'])/np.nansum(vel_cuts_dict['high_velocity_outflow'])**2 + (np.nansum(vel_cuts_dict['low_velocity_var'])+np.nansum(vel_cuts_dict['high_velocity_var']))/(np.nansum(vel_cuts_dict['low_velocity_outflow'])+np.nansum(vel_cuts_dict['high_velocity_outflow']))**2)
                )

    flux_ratio = np.nansum(vel_cuts_dict['low_velocity_outflow'])/(np.nansum(vel_cuts_dict['low_velocity_outflow'])+np.nansum(vel_cuts_dict['high_velocity_outflow']))
    print('Fountain velocity gas to total outflow flux ratio:',
                flux_ratio,
                '+/-',
                #calculate the uncertainty
                #flux ratio * sqrt( var/low_vel^2 + (var_low+var_high)/(low_vel+high_vel)^2 )
                (flux_ratio) * np.sqrt(np.nansum(vel_cuts_dict['low_velocity_var'])/np.nansum(vel_cuts_dict['low_velocity_outflow'])**2 + (np.nansum(vel_cuts_dict['low_velocity_var'])+np.nansum(vel_cuts_dict['high_velocity_var']))/(np.nansum(vel_cuts_dict['low_velocity_outflow'])+np.nansum(vel_cuts_dict['high_velocity_outflow']))**2)
                )


    #run through the plotting scripts
    if "Hb" not in s.line.label:
        fig = plot_velocity_cut_maps(s, vel_cuts_dict['low_velocity_outflow'], vel_cuts_dict['high_velocity_outflow'], title=s.gal_name+' '+s.line_label, tc_data_hbeta=tc_data_hbeta)
    else:
        fig = plot_velocity_cut_maps(s, vel_cuts_dict['low_velocity_outflow'], vel_cuts_dict['high_velocity_outflow'], title=s.gal_name+' '+s.line_label, tc_data_hbeta=None)

    plt.show(block=False)



def data_coords(gal_dict, z, wcs_step, shiftx=None, shifty=None):
    """
    Takes the data cube and creates coordinate arrays that are centred on the
    galaxy.  The arrays can be shifted manually.  If this is not given to
    the function inputs, the function finds the centre using the maximum continuum
    value.

    Parameters
    ----------
    gal_dict : dictionary
        dictionary with the threadcount results

    z : float
        redshift

    wcs_step : list of floats
        the step for the wcs size of the spaxels in arcseconds

    shiftx : float or None
        the hardcoded shift in the x direction for the coord arrays (in arcseconds).
        If this is none, it finds the maximum point of the median across a section
        of continuum, and makes this the centre.  Default is None.

    shifty : float or None
        the hardcoded shift in the y direction for the coord arrays (in arcseconds).
        If this is none, it finds the maximum point of the median across a section
        of continuum, and makes this the centre.  Default is None.

    Returns
    -------
    xx : :obj:'~numpy.ndarray'
        2D x coordinate array

    yy : :obj:'~numpy.ndarray'
        2D y coordinate array

    rad : :obj:'~numpy.ndarray'
        2D radius array
    """
    #get the data shape
    s = gal_dict['choice'].shape

    #create x and y ranges
    x = np.arange(s[0]) #RA
    y = np.arange(s[1]) #DEC

    #multiply through by wcs_step values
    x = x*wcs_step[0]
    y = y*wcs_step[1]

    print("x shape, y shape:", x.shape, y.shape)

    #shift the x and y
    if None not in (shiftx, shifty):
        x = x + shiftx
        y = y + shifty

    #otherwise use the flux fits to find the centre of the galaxy
    else:
        flux_results, flux_error, outflow_flux, outflow_flux_err = calc_sfr.get_arrays(gal_dict, var_string='flux')

        #i, j = np.unravel_index(np.nanargmax(flux_results), flux_results.shape)

        #turn the flux results into an mpdaf image object
        wcs1 = WCS(crval=0, cd=np.array([[0.0, wcs_step[0]],[wcs_step[1],0.0]]), deg=True)
        flux_image = Image(data=flux_results, wcs=wcs1)

        #find the centre of the galaxy by fitting a 2D gaussian model
        gfit = flux_image.gauss_fit(plot=False, unit_center=None)
        i, j = gfit.center

        shiftx = i*wcs_step[0]
        shifty = j*wcs_step[1]

        print("shiftx, shifty:", shiftx, shifty)
        x = x - shiftx
        y = y - shifty

    #create x and y arrays
    xx, yy = np.meshgrid(x,y, indexing='ij')

    print("xx shape, yy shape", xx.shape, yy.shape)

    #create radius array
    rad = np.sqrt(xx**2+yy**2)

    return xx, yy, rad

def calc_effective_radius_tc(gal_dict, radius_array, flux_percentage=50):
    """
    Calculates the effective radius (default) of the galaxy using the fitted
    flux, but can also be used to calculate e.g. r_75 or r_90
    """
    #get the galaxy flux array
    gal_flux, gal_flux_err, flow_flux, flow_flux_err = calc_sfr.get_arrays(gal_dict, var_string='flux')

    #get the total flux
    total_flux = np.nansum(gal_flux)

    #get the half flux (or whatever percentage of the flux you wanted)
    effective_flux = total_flux * (flux_percentage/100)
    print('Looking for effective flux:', effective_flux)

    #get the unique radii from the radius array so we can iterate through them
    unique_rad = np.unique(radius_array)

    #iterate through the available radii and add up the flux
    for i, radius in enumerate(unique_rad):
        #calcualte the enclosed flux
        enclosed_flux = np.nansum(gal_flux[radius_array<=radius])

        #calculate the percentage of the total flux enclosed
        percentage_total_flux = (enclosed_flux/total_flux) * 100

        if percentage_total_flux < flux_percentage:
            continue

        elif percentage_total_flux > flux_percentage:
            print('radius:', radius, 'enclosed flux', enclosed_flux)
            print('percentage of total flux', percentage_total_flux)

            #the previous radius is the one we want, since this one gives too
            #much flux
            effective_radius = unique_rad[i-1]
            enclosed_flux = np.nansum(gal_flux[radius_array<=effective_radius])
            percentage_total_flux = (enclosed_flux/total_flux) * 100

            print('effective radius:', effective_radius, 'enclosed flux', enclosed_flux)
            print('final percentage of total flux', percentage_total_flux)

            return effective_radius


def calc_effective_radius_fits(fits_filename, fits_ext=None, flux_percentage=50):
    """
    Reads in an image fits file (e.g. PANSTARRs) and then calculates the
    effective radius (default) of the galaxy using the summed flux, but can
    also be used to calculate e.g. r_75 or r_90

    Parameters
    ----------
    fits_filename : str
        the data filename (.fits, .png or .bmp)
    fits_ext : int or(int,int) or string or (string,string)
        Number/name of the data extension or numers/names of the data and
        variance extensions. e.g. 'COMPRESSED_IMAGE', 'SCI'.  Default is None.
    flux_percentage : float
        the percentage of the flux included within the radius
    """
    #read in the fits file
    gal = Image(fits_filename, ext=fits_ext)

    #find the centre of the galaxy by fitting a 2D gaussian model
    gfit = gal.gauss_fit(plot=False, unit_center=None)
    gal_center = gfit.center

    #get the data shape
    s = gal.shape

    #create x and y ranges
    x = np.arange(s[0]) #RA
    y = np.arange(s[1]) #DEC

    #shift the x and y by the galaxy centre value
    x = x - gal_center[0]
    y = y - gal_center[1]

    #multiply through by wcs_step values
    x = x*gal.get_step(unit=units.arcsec)[0]
    y = y*gal.get_step(unit=units.arcsec)[1]

    print("x shape, y shape:", x.shape, y.shape)

    #create x and y arrays
    xx, yy = np.meshgrid(x,y, indexing='ij')

    print("xx shape, yy shape", xx.shape, yy.shape)

    #create radius array
    radius_array = np.sqrt(xx**2+yy**2)

    #turn all flux below the "continuum" to nan values
    gal.data[gal.data<gfit.cont] = np.nan

    #get the total flux
    total_flux = np.nansum(gal.data)
    #total_flux = gfit.flux

    #get the half flux (or whatever percentage of the flux you wanted)
    effective_flux = total_flux * (flux_percentage/100)
    print('Looking for effective flux:', effective_flux)

    #get the unique radii from the radius array so we can iterate through them
    unique_rad = np.unique(np.around(radius_array, decimals=2))

    #get the maximum radius
    #max_rad = np.nanmax(radius_array)

    #create an array of radii to iterate through
    #unique_rad = np.linspace(0, max_rad, 100)

    #iterate through the available radii and add up the flux
    for i, radius in enumerate(unique_rad):
        #calculate the enclosed flux
        enclosed_flux = np.nansum(gal.data[radius_array<=radius])

        #calculate the percentage of the total flux enclosed
        percentage_total_flux = (enclosed_flux/total_flux) * 100

        if percentage_total_flux < flux_percentage:
            continue

        elif percentage_total_flux > flux_percentage:
            print('radius:', radius, 'enclosed flux', enclosed_flux)
            print('percentage of total flux', percentage_total_flux)

            #the previous radius is the one we want, since this one gives too
            #much flux
            effective_radius = unique_rad[i-1]
            enclosed_flux = np.nansum(gal.data[radius_array<=effective_radius])
            percentage_total_flux = (enclosed_flux/total_flux) * 100

            print('effective radius:', effective_radius, 'enclosed flux', enclosed_flux)
            print('final percentage of total flux', percentage_total_flux)

            return effective_radius

def calc_stellar_mass_surface_density(stellar_mass, radius_eff, z):
    """
    Calculates the stellar mass surface density of a galaxy given a stellar mass
    and an effective radius
    """
    #give mass units
    stellar_mass = stellar_mass * units.solMass

    #give the radius units
    radius_eff = radius_eff * units.arcsec

    #convert to kpc
    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(units.kpc/units.arcsec)
    radius_eff = radius_eff * proper_dist

    #calcuate the stellar mass surface density
    sigma_star = stellar_mass/(4*np.pi*radius_eff**2)

    return sigma_star


def calc_average_disk_height(gal_dict, sigma_star):
    """
    Calculates the average disk height using the velocity dispersion from the
    threadcount fits
    """
    #give the stellar mass units
    sigma_star = sigma_star * units.solMass

    #get the fitted galaxy sigma
    gal_sigma, gal_sigma_err, flow_sigma, flow_sigma_err = calc_sfr.get_arrays(gal_dict, var_string='sigma')

    gal_center, gal_center_err, flow_center, flow_center_err = calc_sfr.get_arrays(gal_dict, var_string='center')

    #the sigma is in Angstroms, need to convert to km/s
    gal_sigma_vel = calc_vc.sigma_to_vel_disp(gal_sigma, gal_center)

    #take the average
    avg_gal_sigma_vel = np.nanmean(gal_sigma_vel)

    #calculate the average disk height
    avg_disk_height = avg_gal_sigma_vel**2/(np.pi*G*sigma_star)

    return avg_disk_height.to('pc')




#-------------------------------------------------------------------------------
# PLOTS
#-------------------------------------------------------------------------------

def _map_flux(ax, flux_array, wcs_step, cbar_label, contour_array=None, radius_array=None, rad_to_plot=None, vmin=None, vmax=None):
    """
    Maps the flux onto the axis
    """
    try:
        im = ax.imshow(np.log10(flux_array.T), origin='lower', aspect=wcs_step[1]/wcs_step[0], cmap=cmr.gem, vmin=vmin, vmax=vmax)
    except:
        im = ax.imshow(np.log10(flux_array.value).T, origin='lower', aspect=wcs_step[1]/wcs_step[0], cmap=cmr.gem, vmin=vmin, vmax=vmax)

    #make the colourbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(cbar_label, fontsize='small')

    #make the contours
    if contour_array == None:
        try:
            contour_array = np.log10(flux_array)
        except:
            contour_array = np.log10(flux_array.value)
    else:
        try:
            contour_array = np.log10(contour_array)
        except:
            contour_array = np.log10(contour_array.value)

    extent = (-1, flux_array.shape[0], -1, flux_array.shape[1])
    cs = ax.contour(contour_array.T, levels=1, origin='lower', extent=extent, colors='k', alpha=0.7)


    for c in cs.collections:
        c.set_linestyle('solid')

    #plot the radius
    #if rad_to_plot is given, it's just one contour
    if radius_array is not None:
        if rad_to_plot == None:
            cs = ax.contour(radius_array.T, origin='lower', extent=extent, colors='k', alpha=0.7)
        else:
            cs = ax.contour(radius_array.T, levels=[rad_to_plot], origin='lower', extent=extent, colors='k', alpha=0.7)

    #invert xaxis
    ax.invert_xaxis()

    #turn off tick labels
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)


def plot_velocity_cut_maps(settings_array, mid_velocity_array, high_velocity_array, title='Gal Name Line', tc_data_hbeta=None):
    """
    Makes maps of the velocity cuts

    Parameters
    ----------

    crop_data : list of ints or None
        Use to define how much of the data goes into the maps in the format:
        [axis1_begin, axis1_end, axis2_begin, axis2_end]
        e.g. [2, -1, 3, -2] will map data[2:-1, 3:-2]
    """
    #get the flux values
    #gal_flux, gal_flux_err, flow_flux, flow_flux_err = calc_sfr.get_arrays(gal_dict, var_string='flux')

    set_rcParams.set_params({"axes.facecolor": 'none'})

    #calculate the sfr surface density
    if "Hb" not in settings_array.line.label:
        sfr, sfr_err, total_sfr, sigma_sfr, sigma_sfr_err = calc_sfr.calc_sfr(tc_data_hbeta, settings_array.z, settings_array.wcs_step, include_outflow=False, Av=settings_array.Av_array)
    else:
        sfr, sfr_err, total_sfr, sigma_sfr, sigma_sfr_err = calc_sfr.calc_sfr(settings_array.tc_dict, settings_array.z, settings_array.wcs_step, include_outflow=False, Av=settings_array.Av_array)

    #low_vel_masked = ma.masked_where(low_velocity_array<0, low_velocity_array)
    try:
        mid_vel_masked = ma.masked_where(mid_velocity_array<0, mid_velocity_array.value)
    except AttributeError:
        mid_vel_masked = ma.masked_where(mid_velocity_array<0, mid_velocity_array)
    try:
        high_vel_masked = ma.masked_where(high_velocity_array<0, high_velocity_array.value)
    except AttributeError:
        high_vel_masked = ma.masked_where(high_velocity_array<0, high_velocity_array)

    #crop the data
    if settings_array.crop_data:
        #gal_flux = gal_flux[crop_data[0]:crop_data[1], crop_data[2]:crop_data[3]]
        #flow_flux = flow_flux[crop_data[0]:crop_data[1], crop_data[2]:crop_data[3]]
        mid_vel_masked = mid_vel_masked[settings_array.crop_data[0]:settings_array.crop_data[1], settings_array.crop_data[2]:settings_array.crop_data[3]]
        high_vel_masked = high_vel_masked[settings_array.crop_data[0]:settings_array.crop_data[1], settings_array.crop_data[2]:settings_array.crop_data[3]]
        sigma_sfr = sigma_sfr[settings_array.crop_data[0]:settings_array.crop_data[1], settings_array.crop_data[2]:settings_array.crop_data[3]]

    #create the figure
    fig_maps, ax_maps = plt.subplots(1, 3, sharex=True, sharey=True,
            figsize=(9,3), constrained_layout=True)


    #map the things
    _map_flux(ax_maps[0], sigma_sfr, settings_array.wcs_step, r'Log $\Sigma_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]', radius_array=None, rad_to_plot=settings_array.effective_radius, vmin=-2.7, vmax=0.5)#vmin=-2.5, vmax=0.5)#vmin=-2.0, vmax=0.7)# #-2.7, -0.5

    _map_flux(ax_maps[1], mid_vel_masked, settings_array.wcs_step, r'Log Flux$_{\rm mid vel}$ [10$^{-16}$ erg/(cm2 s)]', contour_array=sigma_sfr, radius_array=None, rad_to_plot=settings_array.effective_radius, vmin=-1.5, vmax=1.5)#vmin=-0.7, vmax=2.5)# #-0.8, 1.6

    _map_flux(ax_maps[2], high_vel_masked, settings_array.wcs_step, r'Log Flux$_{\rm high vel}$ [10$^{-16}$ erg/(cm2 s)]', contour_array=sigma_sfr, radius_array=None, rad_to_plot=settings_array.effective_radius, vmin=-1.5, vmax=1.5)#vmin=-0.7, vmax=2.5)


    plt.suptitle(title)

    return fig_maps
