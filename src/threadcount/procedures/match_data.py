"""
NAME:
    match_data.py
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from mpdaf.obj import Image, Cube
from mpdaf.obj import WCS as mpWCS

def read_in_cube_data(filename, **kwargs):
    """
    Read in the KCWI data as an mpdaf cube
    """
    kcwi_cube = Cube(filename, **kwargs)

    return kcwi_cube


def read_in_image_data(filename, **kwargs):
    """
    Read in the Spitzer and PANSTARRs data
    """
    image_data = Image(filename, **kwargs)

    return image_data


def kcwi_cube_to_image(kcwi_cube, plot=False):
    """
    Takes the KCWI cube and turns it into an image, using the redmost wavelengths
    """
    #we're going to use the built-in filters from mpdaf
    #using the SDSS g-band filter, which is very similar to PanSTARRs g-band
    try:
        kcwi_image = kcwi_cube.get_band_image('SDSS_g')
    except ValueError:
        print('The filter curve does not overlap the wavelength coverage '
                r'of the cube.  Using the last 500 A instead.')
        #use the last 500 Angstroms of the cube instead
        kcwi_image = kcwi_cube.get_image((kcwi_cube.wave.get_end()-500,
                        kcwi_cube.wave.get_end())
                        )

    if plot == True:
        plt.figure()
        kcwi_image.plot(scale='log', use_wcs=True)
        plt.show()

    return kcwi_image


def data_coords(gal_image, shiftx=None, shifty=None):
    """
    Takes the data cube and creates coordinate arrays that are centred on the
    galaxy.  The arrays can be shifted manually.  If this is not given to
    the function inputs, the function finds the centre using the maximum continuum
    value.

    Parameters
    ----------
    gal_image : mpdaf image object
        image of the galaxy

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
    s = gal_image.data.shape

    #create x and y ranges
    x = np.arange(s[0]) #RA
    y = np.arange(s[1]) #DEC

    #multiply through by wcs_step values
    x = abs(x*gal_image.get_axis_increments(unit=u.arcsec)[0])
    y = abs(y*gal_image.get_axis_increments(unit=u.arcsec)[1])

    print("x shape, y shape:", x.shape, y.shape)

    #shift the x and y
    if None not in (shiftx, shifty):
        x = x + shiftx
        y = y + shifty

    #otherwise use the flux fits to find the centre of the galaxy
    else:
        #find the centre of the galaxy by fitting a 2D gaussian model
        gfit = gal_image.gauss_fit(plot=False, unit_center=None)
        i, j = gfit.center

        shiftx = i*abs(gal_image.get_axis_increments(unit=u.arcsec)[0])
        shifty = j*abs(gal_image.get_axis_increments(unit=u.arcsec)[1])

        print("shiftx, shifty:", shiftx, shifty)
        x = x - shiftx
        y = y - shifty

    #create x and y arrays
    xx, yy = np.meshgrid(x,y, indexing='ij')

    print("xx shape, yy shape", xx.shape, yy.shape)

    #create radius array
    rad = np.sqrt(xx**2+yy**2)

    return xx, yy, rad



def map_images(image_list):
    """
    Plot the images in the image list in panels
    """
    fig = plt.figure(tight_layout=True,
                        figsize=(3*len(image_list), 3),
                        #sharex=True,
                        #sharey=True
                        )

    #iterate through the image_list
    for i, im in enumerate(image_list):
        ax = fig.add_subplot(1, len(image_list), i+1, projection=im.wcs.wcs)

        im.plot(ax=ax,
                scale='log',
                aspect=im.get_step()[0]/im.get_step()[1]
                )


        label = ' '
        for key in im.data_header.keys():
            if 'TELESCOP' in key:
                label = label+im.data_header[key]+' '

            elif 'INSTRUME' in key:
                label = label+im.data_header[key]+' '

            elif 'FILTERID' in key:
                label = label+im.data_header[key]+' '

            elif 'CHNLNUM' in key:
                label = label+'CHNL '+str(im.data_header[key])+' '

        ax.set_title(label)

        #use the axis limits of the first subplot to set the bounds for the
        #following subplots

    return fig


def map_image_with_contour(image_array, contour_array1, contour_array2=None):
    """
    Maps the image in image_array with contours from contour_array
    """
    fig = plt.figure(tight_layout=True,
                        figsize=(3,4)
                        )

    ax = fig.add_subplot(1, 1, 1, projection=image_array.wcs.wcs)

    image_array.plot(ax=ax,
                        scale='linear',
                        aspect=image_array.get_step()[0]/image_array.get_step()[1]
                        )

    plt.contour(contour_array1.data,
                    colors='k',
                    transform=ax.get_transform(contour_array1.wcs.wcs),
                    )

    if contour_array2 is not None:
        plt.contour(contour_array2.data,
                    colors='r',
                    transform=ax.get_transform(contour_array2.wcs.wcs),
                    )

    return fig


def map_overlaid_images(first_image_array, second_image_array):
    """
    Maps the overlapping images
    """
    fig = plt.figure(tight_layout=True,
                        figsize=(3,4)
                        )

    ax = fig.add_subplot(1, 1, 1, projection=first_image_array.wcs.wcs)

    first_image_array.plot(ax=ax,
                            scale='linear',
                            cmap='Greys',
                            alpha=1.0,
                            aspect=first_image_array.get_step()[0]/first_image_array.get_step()[1]
                            )

    second_image_array.plot(ax=ax,
                    scale='linear',
                    cmap='Reds',
                    alpha=0.5,
                    transform=ax.get_transform(second_image_array.wcs.wcs),
                    aspect=second_image_array.get_step()[0]/second_image_array.get_step()[1]
                    )

    return fig



def main(kcwi_filename, spitzer_filenames, spitzer_err_filenames=None, intermediate_data=None, plot=False, save_result=False):
    """
    Matches the Spitzer data to the KCWI data.  If intermediate_data is given,
    uses this as the intermediate wcs to match both to.

    Parameters
    ----------
    kcwi_filename : str
        file path of the input KCWI cube
    spitzer_filenames : list of str
        list of file paths of the input spitzer images.  Make sure they are in
        order of increasing wavelength, since the first one will be used to correct
        the WCS of the next one.
    spitzer_err_filenames : list of str or None
        list of file paths of the corresponding error arrays for the input spitzer
        images. Must be same length as spitzer_filename, or None.
        (default is None)
    intermediate_data : list of str or None
        list of two file paths to use as intermediaries between KCWI and spitzer
        data. For example, the PanSTARRs or SDSS g-band and z-band images. We
        assume these have a consistant WCS between them, and match the g-band WCS
        to KCWI, copy the WCS shift to the z-band data and then match the Spitzer
        WCS to the z-band image.
    plot : boolean
        whether to plot the results or not (default is False)
    save_result : boolean
        whether to save the result as a fits file, with 'wcs_corrected' added
        into the filename.  This will overwrite any other previous results with
        the same name.  (default is False)
    """
    #read in kcwi data
    kcwi_cube = read_in_cube_data(kcwi_filename, ext=[0,1])

    #turn kcwi cube into an image
    kcwi_image = kcwi_cube_to_image(kcwi_cube, plot=False)

    #check that the spitzer_err_filename is the same length as the spitzer_filename
    if spitzer_err_filenames is not None:
        assert len(spitzer_err_filenames) == len(spitzer_filenames), \
                "Length of spitzer_err_filenames must be same as spitzer_filenames"

    #iterate through the list of filenames
    for i in np.arange(len(spitzer_filenames)):
        #read in the spitzer data
        spitzer_image = read_in_image_data(spitzer_filenames[i])

        #read in the spitzer uncertainty if given
        if spitzer_err_filenames is not None:
            if spitzer_err_filenames[i] is not None:
                with fits.open(spitzer_err_filenames[i]) as hdu:
                    spitzer_image.var = hdu[0].data
                hdu.close()

        if intermediate_data is not None:
            #read in the intermediate data
            go_between_image1 = read_in_image_data(intermediate_data[0], ext=0)
            go_between_image2 = read_in_image_data(intermediate_data[1], ext=0)

            #get the estimated coordinate offset of the intermediate data from
            #the kcwi data
            dy, dx = go_between_image1.estimate_coordinate_offset(kcwi_image)

            #apply it to both go betweens
            go_between_image1.wcs.set_crpix1(go_between_image1.wcs.get_crpix1()+dx)
            go_between_image1.wcs.set_crpix2(go_between_image1.wcs.get_crpix2()+dy)

            go_between_image2.wcs.set_crpix1(go_between_image2.wcs.get_crpix1()+dx)
            go_between_image2.wcs.set_crpix2(go_between_image2.wcs.get_crpix2()+dy)

            # Calculate the resulting shift in pixel coordinates, for display
            # to the user.
            wcs_units = u.arcsec if go_between_image1.wcs.unit is u.deg else go_between_image1.wcs.unit
            offset = np.array([-dy, -dx]) * go_between_image1.wcs.get_axis_increments(wcs_units)
            go_between_image1._logger.info("Shifted the coordinates by dy=%.3g dx=%.3g %s" %
                              (offset[0], offset[1], wcs_units))

            wcs_units = u.arcsec if go_between_image2.wcs.unit is u.deg else go_between_image2.wcs.unit
            offset = np.array([-dy, -dx]) * go_between_image2.wcs.get_axis_increments(wcs_units)
            go_between_image2._logger.info("Shifted the coordinates by dy=%.3g dx=%.3g %s" %
                              (offset[0], offset[1], wcs_units))

            #shift the spitzer to match the second go between
            spitzer_image.adjust_coordinates(go_between_image2,
                                inplace=True)

            #rotate the spitzer image to have the same orientation as the kcwi image
            #the rotae function has a side effect of correcting the image for shear
            #terms in the CD matrix, so this gets done even if no rotation is needed
            spitzer_image._rotate(kcwi_image.wcs.get_rot()-spitzer_image.wcs.get_rot(),
                                reshape=True, #reshapes the array to only encompass
                                #the rotated image
                                regrid=True, #
                                flux=False, #spitzer data is in MJy/sr
                                cutoff=0.5 #mask pixels where at least this fraction
                                #of the pixel was interpolated from dummy values given
                                #to masked input pixels
                                )

            #get the pixel index and Dec, RA coordinate at the center of the KCWI image
            centerpix = np.asarray(kcwi_image.shape) / 2.0
            centersky = kcwi_image.wcs.pix2sky(centerpix)[0]

            #resample the rotated spitzer image to have the same axis increments,
            #offset and number of pixels as the kcwi image
            spitzer_image.regrid(kcwi_image.shape,
                                centersky,
                                centerpix,
                                kcwi_image.wcs.get_axis_increments(unit=u.deg),
                                flux=False,
                                unit_inc=u.deg,
                                inplace=True,
                                cutoff=0.5)

            if plot:
                map_images([kcwi_image,
                            go_between_image1,
                            go_between_image2,
                            spitzer_image
                            ])

                map_image_with_contour(kcwi_image,
                            spitzer_image,
                            go_between_image1
                            )

                map_overlaid_images(kcwi_image,
                            spitzer_image
                            )

        else:
            #match spitzer straight to kcwi
            spitzer_image = spitzer_image.adjust_coordinates(kcwi_image)

            spitzer_image = spitzer_image.align_with_image(kcwi_image,
                                flux=False, #spitzer data is in MJy/sr
                                cutoff=0.5 #mask pixels where at least this fraction
                                #of the pixel was interpolated from dummy values given
                                #to masked input pixels
                                )

            if plot:
                map_images([kcwi_image,
                            spitzer_image
                            ])

                map_image_with_contour(kcwi_image,
                            spitzer_image
                            )

                map_overlaid_images(kcwi_image,
                            spitzer_image
                            )

        if save_result:
            spitzer_image.write(spitzer_image.filename.split('.fits')[0]+'_wcs_corrected.fits')

        #if there's more than one spitzer image to match, make the corrected
        #spitzer image be the 'kcwi_image' to match to, and turn off the
        #intermediate data
        if len(spitzer_filenames) > 1:
            kcwi_image = spitzer_image.copy()
            intermediate_data = None
            print('MATCHING SPITZER DATA TO SPITZER INPUT IMAGE Channel',
                    spitzer_image.data_header['CHNLNUM'])
            print(' ')

    return spitzer_image
