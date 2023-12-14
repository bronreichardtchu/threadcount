"""
NAME:
    calculate_stellar_mass.py
"""
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy import units

from threadcount.procedures import match_data as md

#define Vega magnitudes as a unit
#NOTE: This will not allow you to then convert to another magnitude system,
#it is mainly so that the data will have a defined magnitude unit returned
VEGA = units.def_unit('VEGA')
VEGAMAG = units.mag(VEGA)

from mpdaf.obj import Image, Cube


#-------------------------------------------------------------------------------
# READ IN SPITZER DATA
#-------------------------------------------------------------------------------
def read_in_image_data(filename, **kwargs):
    """
    Read in the Spitzer or 2MASS data
    """
    image_data = Image(filename, **kwargs)

    return image_data

#-------------------------------------------------------------------------------
# BACKGROUND SUBTRACT DATA
#-------------------------------------------------------------------------------
def background_subtract_region(image, sky_region, save_to_file=False):
    """
    Uses the median value of the defined sky_region to subtract the sky from
    images

    Parameters:
    -----------
    image : mpdaf image object
        the image that needs to be sky subtracted
    sky_region : list of int
        a list describing the region of the data to use as the sky.  Should be
        in the format [x_begin, x_end, y_begin, y_end]
    save_to_file : boolean
        whether to save to file or not
    """
    #get the sky region
    sky_array = image.data[sky_region[0]:sky_region[1], sky_region[2]:sky_region[3]]

    #take the median and standard deviation
    sky_median = np.nanmedian(sky_array)
    #sky_stdev = np.nanstd(sky_array)

    #subtract median from image
    image.data = image.data - sky_median

    #take the standard deviation of the subtracted image
    sky_array = image.data[sky_region[0]:sky_region[1], sky_region[2]:sky_region[3]]
    sky_stdev = np.nanstd(sky_array)

    #propogate the errors
    if image.var:
        image.var = image.var + sky_stdev**2
    else:
        image.var = sky_stdev * image.data

    #save the results
    if save_to_file:
        #create new filename
        new_filename = image.filename.split('.fits')[0] + '_sky_subt.fits'

        #add to header
        image.data_header['HISTORY'] = 'Sky subtracted'
        image.data_header['SKY_STD'] = (sky_stdev, 'st dev of sky region')

        #write to file
        image.write(new_filename)

    return image, sky_stdev

def background_subtract_iterative_clipping(data_filename, var_filename=None, save_to_file=False):
    """
    Reads in the image that needs to be background subtracted, with its variance
    image if given, and subtracts the background value from the map.  Uses the
    mpdaf iterative sigma clipping method to find the background value

    Parameters
    ----------
    data_filename : str
        the location of the data image file to read in
    var_filename : str or None
        the location of the variance image file to read in (Default is None)
    save_to_file : boolean
        whether to save to file or not
    """
    #read in the data files
    image = read_in_image_data(data_filename)

    #if the variance image filename has been given, use it to fill the variance
    #array
    if var_filename:
        image_var = read_in_image_data(var_filename)
        image.var = image_var.data


    #find the background value
    background, background_stdev = image.background()
    try:
        print(image.data_header['OBJECT'],
                'Background:',
                '{0:.2f}'.format(background),
                r'$\pm$',
                '{0:.2f}'.format(background_stdev),
                image.data_header['BUNIT']
                )
    except:
        print('Background:',
                '{0:.2f}'.format(background),
                r'$\pm$',
                '{0:.2f}'.format(background_stdev)
                )

    #apply background value to the galaxy
    image.data = image.data - background

    #propogate the errors
    if image.var is not None:
        image.var = image.var + background_stdev**2

    #save the results
    if save_to_file:
        #create new filename
        new_filename = image.filename.split('.fits')[0] + '_bkgd_subt.fits'

        #add to header
        image.data_header['HISTORY'] = 'Background subtracted'
        image.data_header['BKGD_STD'] = (background_stdev, 'stdev of subtracted bkgnd')
        image.data_header['BKGD'] = (background, 'Background subtracted from image')

        #write to file
        image.write(new_filename)

    return image


#-------------------------------------------------------------------------------
# COMPARE 2MASS AND SPITZER FLUX RATIOS
#-------------------------------------------------------------------------------

def compare_spitzer_2MASS(spitzer_image_list, image_2MASS_list, r90_list):
    """
    Compares the list of spitzer images and 2MASS images and returns the median
    flux ratio
    """
    #check that the lists have the same length
    assert len(spitzer_image_list) == len(image_2MASS_list), 'Image lists must be the same length'

    #create an array to keep the ratios in
    flux_ratio_array = np.array([])

    #iterate through each list
    list_len = len(spitzer_image_list)
    for i in np.arange(list_len):
        #Read in spitzer data
        spitzer_image = read_in_image_data(spitzer_image_list[i])

        #read in 2MASS data
        image_2MASS = read_in_image_data(image_2MASS_list[i])

        #convert to MJy/sr to match the spitzer data
        image_2MASS = dn_to_magnitude(image_2MASS)
        image_2MASS = magnitude_to_MJy_per_sr(image_2MASS)

        #create a radius array using the spitzer data
        xx, yy, rad = md.data_coords(spitzer_image)

        #total flux within r_90
        flux_mask = (rad<r90_list[i]) #& (spitzer_image.data<spitzer_image.var)
        spitzer_flux = spitzer_image.data*spitzer_image.unit
        spitzer_flux_r90 = np.nansum(spitzer_flux[flux_mask])
        print('Total Spitzer flux:', spitzer_flux_r90)

        flux_mask = (rad<r90_list[i]) #& (image_2MASS.data<image_2MASS.var)
        flux_2MASS_r90 = np.nansum(image_2MASS.data[flux_mask])
        print('Total 2MASS flux:', flux_2MASS_r90)

        #take the ratio
        flux_ratio = spitzer_flux_r90/flux_2MASS_r90
        print('Spitzer-to-2MASS ratio:', flux_ratio)
        print(' ')

        #add to the list
        flux_ratio_array = np.append(flux_ratio_array, flux_ratio)


    #take the median of the flux ratios
    flux_ratio = np.nanmedian(flux_ratio_array)

    #take the standard deviation of the flux ratios
    flux_ratio_stdev = np.nanstd(flux_ratio_array)

    return flux_ratio, flux_ratio_stdev

#-------------------------------------------------------------------------------
# GET STELLAR COMPONENT OF SPITZER DATA
#-------------------------------------------------------------------------------

def median_perc_diff(array1, array2):
    """
    Calculates the median percentage difference between two arrays
    """
    #difference between the arrays
    array_diff = array1 - array2
    #percentage as an absolute value
    array_perc = abs(array_diff/array1) * 100
    #and find the median
    return np.nanmedian(array_perc)


def estimate_36um_PAH_with_8um(spitzer_image_36, spitzer_image_8, alpha_factor=0.049, alpha_err=0.0004):
    """
    Uses the 8um flux to estimate the PAH flux in 3.6um

    Default alpha and alpha_err are taken from Meidt et al 2012.  Use the plotting
    function below to figure out a good alpha +/- alpha_err to use for each galaxy
    """
    #get the non-stellar flux at 8um
    non_stellar_8um = spitzer_image_8.data - 0.232*spitzer_image_36.data
    non_stellar_8um_var = spitzer_image_8.var + spitzer_image_36.var*(0.232**2)

    #calculate the non-stellar flux at 3.6um
    non_stellar_36um = alpha_factor * non_stellar_8um

    non_stellar_36um_var = non_stellar_36um**2 * ((alpha_err/alpha_factor)**2 + non_stellar_8um_var/(non_stellar_8um**2))

    #create a new image to keep the non-stellar 3.6um flux in
    non_stellar_36um_image = spitzer_image_36.clone()
    non_stellar_36um_image.data = non_stellar_36um
    non_stellar_36um_image.var = non_stellar_36um_var

    #create a mew image to keep the non-stellar 8um flux in
    non_stellar_8um_image = spitzer_image_8.clone()
    non_stellar_8um_image.data = non_stellar_8um
    non_stellar_8um_image.var = non_stellar_8um_var

    return non_stellar_36um_image, non_stellar_8um_image

def estimate_36um_stellar_component(spitzer_image_36, non_stellar_36um_image):
    """
    Takes the non-stellar 3.6um flux estimated from estimate_36um_PAH_with_8um
    and calculates the stellar component of the 3.6um flux
    """
    #create an image to save the results in
    stellar_36um_image = spitzer_image_36.clone()

    #subtract the non-stellar data from the stellar component
    stellar_36um_image.data = spitzer_image_36.data - non_stellar_36um_image.data

    stellar_36um_image.var = spitzer_image_36.var + non_stellar_36um_image.var

    return stellar_36um_image


def check_stellar_comp_is_positive(stellar_36um_image, spitzer_image_36, spitzer_image_8, alpha_factor=0.049, alpha_err=0.0004):
    """
    Checks that the 3.6um stellar component is positive, and if it is not, run
    through the PAH calculation again slowly lowering the alpha factor by 0.01
    until it reaches zero.  Replace the negative pixels with the results from
    this.
    """
    #first check if there are any negative pixels, otherwise can skip this
    #whole thing
    while (np.where(stellar_36um_image.data<-stellar_36um_image.var)[0].shape[0] > 0) and (alpha_factor>0.0):
        print('Replacing', np.where(stellar_36um_image.data<-stellar_36um_image.var)[0].shape[0],
                'pixels with PAH correction using alpha value', alpha_factor)
        #run through the whole calculation again with Sharon Meidt's alpha factor
        non_stellar_36um_image_new, non_stellar_8um_image_new = estimate_36um_PAH_with_8um(spitzer_image_36,
                        spitzer_image_8,
                        alpha_factor=alpha_factor,
                        alpha_err=alpha_err
                        )
        stellar_36um_image_new = estimate_36um_stellar_component(spitzer_image_36, non_stellar_36um_image_new)

        #replace the negative pixels with results from this run
        negative_mask = stellar_36um_image.data < 0

        stellar_36um_image.data[negative_mask] = stellar_36um_image_new.data[negative_mask]
        stellar_36um_image.var[negative_mask] = stellar_36um_image_new.var[negative_mask]

        #update the alpha value
        alpha_factor = alpha_factor-0.01

    return stellar_36um_image


def iterate_on_stellar_component(spitzer_image_36, spitzer_image_8, alpha_factor=0.049, alpha_err=0.0004):
    """
    Iterate on the stellar component, doing:
            F_non-stellar,8um = F_8 - 0.232 F_stellar,3.6um
            F_non-stellar,3.6um = alpha * F_non-stellar,8um
            F_stellar,3.6um = F_3.6um - F_non-stellar,3.6um
    using F_3.6um as the F_stellar,3.6um in the first iteration.
    """
    #do the first run
    non_stellar_36um_image, non_stellar_8um_image = estimate_36um_PAH_with_8um(spitzer_image_36,
                    spitzer_image_8,
                    alpha_factor=alpha_factor,
                    alpha_err=alpha_err
                    )

    stellar_36um_image = estimate_36um_stellar_component(spitzer_image_36, non_stellar_36um_image)

    #check that the stellar 3.6um component is positive
    stellar_36um_image = check_stellar_comp_is_positive(stellar_36um_image, spitzer_image_36, spitzer_image_8, alpha_factor=alpha_factor, alpha_err=alpha_err)

    #calculate the median percentage difference
    stellar_36um_perc_diff = median_perc_diff(spitzer_image_36.data, stellar_36um_image.data)

    i = 1
    print('Iteration', i, 'Median Percentage difference:', stellar_36um_perc_diff)

    #iterate until the median percentage difference is less than 5%
    while stellar_36um_perc_diff > 1.0:
        #use the calculated stellar 3.6um image to get the non-stellar 3.6um and
        #8um components
        non_stellar_36um_image, non_stellar_8um_image = estimate_36um_PAH_with_8um(stellar_36um_image,
                        spitzer_image_8,
                        alpha_factor=alpha_factor,
                        alpha_err=alpha_err
                        )
        #recalculate the stellar 3.6um component
        stellar_36um_image_new = estimate_36um_stellar_component(spitzer_image_36, non_stellar_36um_image)

        #check that the stellar 3.6um component is positive
        stellar_36um_image_new = check_stellar_comp_is_positive(stellar_36um_image_new, spitzer_image_36, spitzer_image_8, alpha_factor=alpha_factor, alpha_err=alpha_err)

        #calculate the median percentage difference
        stellar_36um_perc_diff = median_perc_diff(stellar_36um_image.data, stellar_36um_image_new.data)

        #set the new stellar component to be the original one
        stellar_36um_image.data = stellar_36um_image_new.data.copy()
        stellar_36um_image.var = stellar_36um_image_new.var.copy()

        #add iteration numbers
        i = i + 1
        print('Iteration', i, 'Median Percentage difference:', stellar_36um_perc_diff)

    return stellar_36um_image, non_stellar_36um_image, non_stellar_8um_image


def main_spitzer_stellar_component(spitzer_36_filename, spitzer_8_filename, alpha_factor=0.049, alpha_err=0.0004):
    """
    Reads in the Spitzer 3.6um and 8um data, iterates to create the non-stellar
    and stellar components of the 3.6um flux, and then saves to file
    """
    #read in data
    spitzer_image_36 = read_in_image_data(spitzer_36_filename)
    spitzer_image_8 = read_in_image_data(spitzer_8_filename)

    #iterate over data
    stellar_36um_image, non_stellar_36um_image, non_stellar_8um_image = iterate_on_stellar_component(spitzer_image_36,
            spitzer_image_8,
            alpha_factor=alpha_factor,
            alpha_err=alpha_err
            )

    #save to file
    stellar_36um_image.write(spitzer_image_36.filename.split('.fits')[0]+'_stellar.fits')
    non_stellar_36um_image.write(spitzer_image_36.filename.split('.fits')[0]+'_non_stellar.fits')
    non_stellar_8um_image.write(spitzer_image_8.filename.split('.fits')[0]+'_non_stellar.fits')

    return stellar_36um_image, non_stellar_36um_image


def plot_36um_against_8um_PAH(spitzer_36_filename, spitzer_8_filename, galaxy_name, percentage_of_points=25, alpha_factor=0.049):
    """
    Plots the 3.6um flux against the 8um PAH
    """
    #read in data
    spitzer_image_36 = read_in_image_data(spitzer_36_filename)
    spitzer_image_8 = read_in_image_data(spitzer_8_filename)

    #get the non-stellar flux at 8um
    non_stellar_8um = spitzer_image_8.data - 0.232*spitzer_image_36.data
    non_stellar_8um_var = spitzer_image_8.var + spitzer_image_36.var*(0.232**2)

    #make the line
    x_array = np.linspace(np.nanmin(abs(non_stellar_8um)), np.nanmax(abs(non_stellar_8um)))
    y_array = alpha_factor * x_array

    #get the lowest 1% of non_stellar_8um values
    threshold = np.nanpercentile(abs(spitzer_image_36.data/non_stellar_8um), percentage_of_points)
    mask = spitzer_image_36.data/non_stellar_8um < threshold

    #take the median of the points
    new_alpha_factor = np.nanmedian(abs(spitzer_image_36.data[mask])/abs(non_stellar_8um[mask]))
    new_alpha_factor_stdev = np.nanstd(abs(spitzer_image_36.data[mask])/abs(non_stellar_8um[mask]))

    new_y_array = new_alpha_factor * x_array

    #plot the non-stellar flux against the 3.6um flux
    fig, ax = plt.subplots(1, 2,
                        sharex=True,
                        sharey=False,
                        constrained_layout=True,
                        figsize=(8, 4)
                        )

    ax[0].scatter(np.log10(non_stellar_8um), np.log10(spitzer_image_36.data), alpha=0.3, s=3)
    ax[0].scatter(np.log10(non_stellar_8um[mask]), np.log10(spitzer_image_36.data)[mask], alpha=0.3, s=3)

    ax[0].plot(np.log10(x_array), np.log10(y_array), label='{0:.2f}'.format(alpha_factor)+r' $F_{8um}$')

    ax[0].plot(np.log10(x_array), np.log10(new_y_array), label='{0:.2f}'.format(new_alpha_factor)+r'$\pm$'+'{0:.2f}'.format(new_alpha_factor_stdev)+r' $F_{8um}$')

    ax[0].legend(frameon=False)

    ax[0].set_xlabel(r'Log(Non stellar 8um Flux [MJy sr$^{-1}$])')
    ax[0].set_ylabel(r'Log(3.6um Flux [MJy sr$^{-1}$])')

    ax[1].scatter(np.log10(non_stellar_8um), spitzer_image_36.data/non_stellar_8um, alpha=0.3, s=3)
    ax[1].scatter(np.log10(non_stellar_8um)[mask], (spitzer_image_36.data/non_stellar_8um)[mask], alpha=0.3, s=3)

    ax[1].text(0.04, 0.96,
                'Lowest '+str(percentage_of_points)+'% of points',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax[1].transAxes
                )

    ax[1].set_xlabel(r'Log(Non stellar 8um Flux [MJy sr$^{-1}$])')
    ax[1].set_ylabel('3.6um / non-stellar 8um')

    fig.suptitle(galaxy_name)

    plt.show()



#-------------------------------------------------------------------------------
# UNIT CONVERSIONS
#-------------------------------------------------------------------------------

def MJy_per_sr_to_Jy(spitzer_image):
    """
    Converts the data from MJy/sr to Janskys
    """
    #first check that the spitzer image has a unit and that it is in MJy/sr
    #if the image doesn't get the unit from the header, then it will default
    #to units.dimensionless_unscaled
    if spitzer_image.unit == units.dimensionless_unscaled:
        spitzer_image.unit = (units.MJy/units.sr)

    elif spitzer_image.unit == (units.MJy/units.sr):
        print('Spitzer input image has units', spitzer_image.unit)

    else:
        print('Spitzer input image must be in units of MJy/sr. ',
            'Currently has the units', spitzer_image.unit)
        return spitzer_image

    #get the data from the image and give it units (MJy/sr)
    spitzer_flux = spitzer_image.data * spitzer_image.unit
    spitzer_var = spitzer_image.var * spitzer_image.unit

    #now go from MJy/sr to Jy/arcsec^2
    spitzer_flux = spitzer_flux.to(units.Jy/units.arcsec**2)
    spitzer_var = spitzer_var.to(units.Jy/units.arcsec**2)

    #get rid of the arcsec^2 bit
    axis_increments = spitzer_image.get_axis_increments(unit=units.arcsec)
    spitzer_flux = spitzer_flux * abs(axis_increments[0]*axis_increments[1])*units.arcsec**2
    spitzer_var = spitzer_var * abs(axis_increments[0]*axis_increments[1])**2*units.arcsec**2

    #put back into the spitzer image
    spitzer_image.data = spitzer_flux
    spitzer_image.var = spitzer_var
    spitzer_image.unit = spitzer_flux.unit
    spitzer_image.data_header['BUNIT'] = str(spitzer_flux.unit)

    return spitzer_image

def Jy_to_solLum(spitzer_image, z):
    """
    Converts the data from Jy to solar luminosities
    """
    #first check that the spitzer image has a unit and that it is in Jy
    #if the image doesn't get the unit from the header, then it will default
    #to units.dimensionless_unscaled
    if spitzer_image.unit == units.dimensionless_unscaled:
        print('Double check that data is in correct input units.  Data has been given units Jy')
        spitzer_image.unit = (units.Jy)

    elif spitzer_image.unit == (units.Jy):
        print('Spitzer input image has units', spitzer_image.unit)

    else:
        print('Spitzer input image must be in units of Jy. ',
            'Currently has the units', spitzer_image.unit)
        return spitzer_image

    #get the data from the image and give it units (Jy)
    spitzer_flux = spitzer_image.data * spitzer_image.unit

    #Janskys to erg/s/Hz/cm^2
    spitzer_flux = spitzer_flux.to(units.erg/(units.s*units.Hz*(units.cm)**2))

    #central frequency of the band
    if spitzer_image.data_header['CHNLNUM'] == 1:
        band_freq = (3.551*units.um).to('Hz', equivalencies=units.spectral())

    elif spitzer_image.data_header['CHNLNUM'] == 2:
        band_freq = (4.493*units.um).to('Hz', equivalencies=units.spectral())

    elif spitzer_image.data_header['CHNLNUM'] == 3:
        band_freq = (5.730*units.um).to('Hz', equivalencies=units.spectral())

    elif spitzer_image.data_header['CHNLNUM'] == 4:
        band_freq = (7.873*units.um).to('Hz', equivalencies=units.spectral())

    ##now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist.to('Mpc'))

    #convert from flux density to luminosity
    luminosity = spitzer_flux * (4*np.pi*dist**2) * band_freq

    #convert to solar luminosities
    solar_lum = luminosity.to('solLum')

    #put back into the spitzer image
    spitzer_image.data = solar_lum
    spitzer_image.unit = solar_lum.unit
    spitzer_image.data_header['BUNIT'] = str(solar_lum.unit)

    return spitzer_image


def Jy_to_Vega(spitzer_image):
    """
    Converts the data from Jy to Vega Magnitudes
    """
    #first check that the spitzer image has a unit and that it is in Jy
    #if the image doesn't get the unit from the header, then it will default
    #to units.dimensionless_unscaled
    if spitzer_image.unit == units.dimensionless_unscaled:
        print('Double check that data is in correct input units.  Data has been given units Jy')
        spitzer_image.unit = (units.Jy)

    elif spitzer_image.unit == (units.Jy):
        print('Spitzer input image has units', spitzer_image.unit)

    else:
        print('Spitzer input image must be in units of Jy. ',
            'Currently has the units', spitzer_image.unit)
        return spitzer_image

    #get the data from the image and give it units (Jy)
    spitzer_flux = spitzer_image.data * spitzer_image.unit

    #get rid of anything with a S/N less than 5
    spitzer_flux[spitzer_flux.value/spitzer_image.var < 5.0] = np.nan

    #go from Jy to uJy
    spitzer_flux = spitzer_flux.to('uJy')

    #band constant K
    #taken from the S-COSMOS overview webpage:
    #https://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/scosmos_irac_colDescriptions.html
    if spitzer_image.data_header['CHNLNUM'] == 1:
        band_K = -2.788

    elif spitzer_image.data_header['CHNLNUM'] == 2:
        band_K = -3.255

    elif spitzer_image.data_header['CHNLNUM'] == 3:
        band_K = -3.743

    elif spitzer_image.data_header['CHNLNUM'] == 4:
        band_K = -4.372

    #now convert to Vega magnitudes
    spitzer_flux = -2.5 * np.log10(spitzer_flux.value) + 23.9 + band_K

    #put back into the spitzer image
    spitzer_image.data = spitzer_flux
    spitzer_image.unit = VEGAMAG
    spitzer_image.data_header['BUNIT'] = str(VEGAMAG)

    return spitzer_image


def dn_to_magnitude(image_2MASS):
    """
    Take the input image from 2MASS data and convert the original data to be
    in Vega magnitudes using the zero point in the header:
        mag = MAGZP - 2.5 log10(S)
    """
    #double check that this hasn't already been done:
    if image_2MASS.unit == units.mag or image_2MASS.data_header['BUNIT']==str(units.mag):
        print('This image has already been converted to magnitudes')
        return image_2MASS

    else:
        #find the zero point
        zp = image_2MASS.data_header['MAGZP']

        #calculate the magnitudes
        mag = zp - 2.5 * np.log10(image_2MASS.data)

        #convert the background (sky) standard deviation to magnitudes
        #treating this as an uncertainty propogation
        sky_stdev = image_2MASS.data_header['SKY_STD']
        sky_stdev = abs((-2.5*sky_stdev) / (image_2MASS.data * np.log(10)))

        #turn into the variance cube
        image_2MASS.var = sky_stdev

        #update the image data
        image_2MASS.data = mag.data
        image_2MASS.unit = units.mag
        image_2MASS.data_header['BUNIT'] = str(units.mag)

        return image_2MASS

def magnitude_to_MJy_per_sr(image_2MASS):
    """
    Convert from Vega magnitudes to Jy/sr for the 2MASS data in H-band
    Using the flux for zero magnitude from Cohen+2003
    https://ui.adsabs.harvard.edu/abs/2003AJ....126.1090C/abstract
    """
    #check the data is not already in Jy/sr
    if image_2MASS.unit == units.Jy/units.sr or image_2MASS.data_header['BUNIT'] == str(units.Jy/units.sr):
        print('This image has already been converted to Jy/sr')
        return image_2MASS

    else:
        #set the flux for zero magnitude
        if image_2MASS.data_header['FILTER'] == 'j':
            f_nu = 1594 * units.Jy
            f_nu_err = 27.8 * units.Jy
        if image_2MASS.data_header['FILTER'] == 'h':
            f_nu = 1024 * units.Jy
            f_nu_err = 20.0 * units.Jy
        if image_2MASS.data_header['FILTER'] == 'k':
            f_nu = 666.7 * units.Jy
            f_nu_err = 12.6 * units.Jy

        #calculate the flux density in Jy
        flux_jy = f_nu * 10**(-0.4*image_2MASS.data)

        #calculate the error in Jy
        flux_err = image_2MASS.var

        flux_err = abs(flux_jy) * np.sqrt((f_nu_err/f_nu)**2 +
                    (-0.4 * np.log(10) * flux_err)**2)

        #divide by the pixel size to get Jy/sr
        axis_increments = image_2MASS.get_axis_increments(unit=units.arcsec)
        pixel_size = abs(axis_increments[0] * axis_increments[1])*(units.arcsec**2)
        flux_MJy_sr = (flux_jy/pixel_size).to(units.MJy/units.sr)

        #and do the error...
        flux_MJy_sr_err = (flux_err/pixel_size).to(units.MJy/units.sr)

        #update the image data
        image_2MASS.data = flux_MJy_sr
        image_2MASS.unit = units.MJy/units.sr
        image_2MASS.data_header['BUNIT'] = str(units.MJy/units.sr)
        image_2MASS.var = flux_MJy_sr_err

        return image_2MASS


def calc_abs_magnitude(image, z):
    """
    Calculates the absolute magnitude M = m - 5log(d/10pc)
    """
    #calculate the distance in parsecs
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in pc
    dist = (c*z/H_0).decompose().to('pc')
    print('distance:', dist.to('Mpc'))

    #put into the distance modulus
    abs_mag = image.data - 5*np.log10(dist/(10*units.pc))

    image.data = abs_mag
    image.unit = units.mag
    image.data_header['BUNIT'] = str(units.mag)

    return image



def abs_mag_to_solar_luminosity(image):
    """
    Takes the data in absolute magnitudes, finds the filter used and converts to
    solar luminosities.  Magnitudes taken from:
        https://mips.as.arizona.edu/~cnaw/sun.html
    Cite Willmer+2018, ApJS, 236, 47
        gal_lum = 10^0.4*(M_sun - gal_mag)
    """
    if image.data_header['FILTER'] == 'h':
        sun_mag = 3.32
    elif image.data_header['FILTER'] == 'j':
        sun_mag = 3.67
    elif image.data_header['CHNLNUM'] == 1:
        sun_mag = 3.26
    elif image.data_header['CHNLNUM'] == 2:
        sun_mag = 3.28
    elif image.data_header['CHNLNUM'] == 3:
        sun_mag = 3.28
    elif image.data_header['CHNLNUM'] == 4:
        sun_mag = 3.26
    else:
        print("Don't have the solar magnitude for that filter, sorry.")
        return image

    gal_lum = 10**(0.4*(sun_mag - image.data))

    image.data = gal_lum
    image.unit = units.solLum
    image.data_header['BUNIT'] = str(units.solLum)

    return image


#-------------------------------------------------------------------------------
# CALCULATE STELLAR MASS
#-------------------------------------------------------------------------------

def calc_mass_to_light_36um(flux_36, flux_45):
    """
    Uses the flux from the 3.6um and 4.5um Spitzer bands to calculate the
    mass to light ratio at 3.6um.

    Parameters
    ----------
    flux_36 :
        the flux magnitude in Vega magnitudes of the 3.6um band
    flux_45 :
        the flux magnitude in Vega magnitudes of the 3.6um band
    """
    #first find the colour difference between 3.6um and 4.5um
    colour_diff = flux_36 - flux_45

    #next apply the formula from Meidt+ 2014 to find log(M/L)
    log_ml = 3.98 * colour_diff + 0.13
    log_ml_err = np.sqrt((0.98)**2 + (0.08)**2)

    #take the log of the M/L
    ml = 10**log_ml.data
    ml_err = abs(ml * np.log(10) * log_ml_err)

    return ml, ml_err


def calc_stellar_mass(ml, ml_err, spitzer_image):
    """
    Takes the mass to light ratio at 3.6um and converts it to a stellar mass
    Note: the spitzer_image must be in bolometric luminosities
    """
    #multiply the mass to light ratio by the absolute solar mag to get
    #the stellar mass
    stellar_mass = ml * spitzer_image.data

    return stellar_mass

def calc_stellar_mass_Querejeta_2015(spitzer_image, z):
    """
    Uses the equation from Querejeta et al. 2015
    Note: image must be in MJy/sr
    """
    #get the dist in Mpc
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in pc
    dist = (c*z/H_0).decompose().to('Mpc')
    print('distance:', dist)

    #change in pixel size - Querejeta et al. 2015 assume a pixel size of 0.75"
    pixel_size_ratio = abs(spitzer_image.get_axis_increments(unit=units.arcsec)[0]*spitzer_image.get_axis_increments(unit=units.arcsec)[1])/(0.75*0.75)

    #assume a M/L ratio of 0.6 for SFing gals; or
    #assume a M/L ratio of 0.44 for LMC
    #multiply by 1.05 to adjust to Kroupa IMF
    #ml = 0.6 * 1.05
    ml = 0.44 * 1.05

    #stellar mass
    stellar_mass = 9308.23 * spitzer_image.data * (dist.value**2) * ml * pixel_size_ratio

    #stellar_mass_err = abs(9308.25*dist.value**2*pixel_size_ratio*0.6) * spitzer_image.var
    stellar_mass_err = abs(9308.23*(dist.value**2)*ml) * spitzer_image.var

    return stellar_mass, stellar_mass_err


def calc_stellar_mass_Eskew_2012(spitzer_image_36, spitzer_image_45, z):
    """
    Uses the equation from Eskew et al. 2012
    Note: spitzer images must be in Jy
    """
    #get the dist in Mpc
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in pc
    dist = (c*z/H_0).decompose().to('Mpc')
    print('distance:', dist)

    #calculate stellar mass
    stellar_mass = 10**5.65 * spitzer_image_36.data**2.85 * spitzer_image_45.data**-1.85 * (dist/0.05)**2

    #stellar_mass_err = stellar_mass * np.sqrt()

    return stellar_mass



def calc_stellar_mass_surface_density(stellar_mass, stellar_mass_err, z, axis_increments):
    """
    Divide by the spaxel area to get the Sigma_star
    Note: axis_increments must be in arcseconds
    """
    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(units.pc/units.arcsec)

    x = abs(axis_increments[0])
    y = abs(axis_increments[1])

    print('Spaxel Area:', (x*y)*(proper_dist)**2)

    sm_surface_density = stellar_mass/((x*y)*(proper_dist**2))
    sm_surface_density_err = stellar_mass_err/((x*y)*(proper_dist**2))

    return sm_surface_density, sm_surface_density_err


#-------------------------------------------------------------------------------
# MAIN FUNCTIONS
#-------------------------------------------------------------------------------

def main_with_conversions(filename_36, filename_45, z):
    """
    DEPRECATED
    Takes the spitzer data and converts it to a stellar mass map
    """
    #read in the data
    spitzer_image_36 = read_in_image_data(filename_36)
    spitzer_image_45 = read_in_image_data(filename_45)

    #convert from MJy/sr to Jy
    #spitzer_image_36 = MJy_per_sr_to_Jy(spitzer_image_36)
    #spitzer_image_45 = MJy_per_sr_to_Jy(spitzer_image_45)

    #convert froom Jy to Vega magnitudes
    #spitzer_image_36 = Jy_to_Vega(spitzer_image_36)
    #spitzer_image_45 = Jy_to_Vega(spitzer_image_45)

    #calculate the mass to light ratio
    #ml, ml_err = calc_mass_to_light_36um(spitzer_image_36, spitzer_image_45)

    #convert from Vega to absolute magnitude
    #spitzer_image_36 = calc_abs_magnitude(spitzer_image_36, z)

    #convert frrom absolute magnitude to absolute solar luminosity
    #spitzer_image_36 = abs_mag_to_solar_luminosity(spitzer_image_36)

    #calculate the stellar mass
    #stellar_mass = calc_stellar_mass(ml, ml_err, spitzer_image_36)
    stellar_mass, stellar_mass_err = calc_stellar_mass_Querejeta_2015(spitzer_image_36, z)

    #calculate the stellar mass surface density
    sigma_sm = calc_stellar_mass_surface_density(stellar_mass, z,
            axis_increments=spitzer_image_36.get_axis_increments(unit=units.arcsec)*units.arcsec
            )

    #return ml, ml_err, stellar_mass
    return stellar_mass, sigma_sm


def main(filename, z, convert_from_2MASS=False):
    """
    Takes the spitzer data and converts it to a stellar mass map
    """
    #read in the data (usually 3.6um spitzer data)
    flux_image = read_in_image_data(filename)

    #convert from 2MASS data to spitzer using the flux ratio
    #that I worked out from the galaxies that have both
    if convert_from_2MASS == True:
        #use the variance array as the stellar mass err
        stellar_mass_err = flux_image.var

        #convert to solar luminosities
        flux_image = dn_to_magnitude(flux_image)
        #flux_image = calc_abs_magnitude(flux_image, z)
        #flux_image = abs_mag_to_solar_luminosity(flux_image)

        #calculate the stellar mass
        #assume a M/L ratio of 0.6 for SFing gals; or
        #assume a M/L ratio of 0.44 for LMC
        #multiply by 1.05 to adjust to Kroupa IMF
        #ml = 0.6 * 1.05
        #ml = 0.44 * 1.05

        #stellar_mass = ml * flux_image.data
        #stellar_mass_err = np.full_like(stellar_mass, 0.0)

        flux_image = magnitude_to_MJy_per_sr(flux_image)
        #multiply by the flux ratio to bring to spitzer levels
        #new_data = flux_image.data.data.value * 0.09 #0.29
        #flux_image.data = flux_image.data.data.value * 0.29
        new_data = flux_image.data.data.value * 0.23

        #flux_image.var = new_data**2 * (flux_image.var.data/(flux_image.data.data.value**2) * (0.03/0.29)**2)
        #flux_image.var = new_data**2 * (flux_image.var.data/(flux_image.data.data.value**2) * (0.13/0.09)**2)
        #flux_image.var = new_data**2 * (flux_image.var.data/(flux_image.data.data.value**2) * (0.13/0.23)**2)
        flux_image.data = new_data

        #calculate the stellar mass
        stellar_mass, _ = calc_stellar_mass_Querejeta_2015(flux_image, z)

    else:
        #calculate the stellar mass
        stellar_mass, stellar_mass_err = calc_stellar_mass_Querejeta_2015(flux_image, z)

    #calculate the stellar mass surface density
    sigma_sm, sigma_sm_err = calc_stellar_mass_surface_density(stellar_mass, stellar_mass_err, z,
            axis_increments=flux_image.get_axis_increments(unit=units.arcsec)*units.arcsec
            )

    #return ml, ml_err, stellar_mass
    return stellar_mass, stellar_mass_err, sigma_sm, sigma_sm_err
