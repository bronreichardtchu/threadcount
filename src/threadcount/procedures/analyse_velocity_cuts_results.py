"""
NAME:
	analyse_velocity_cuts_results.py

FUNCTIONS INCLUDED:


"""

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import importlib 
import glob

from mpdaf.obj import Image, WCS
from astropy import units
from astropy.constants import G
from astropy.cosmology import WMAP9 as cosmo

import cmasher as cmr

import threadcount as tc
from threadcount.procedures import calculate_star_formation_rate as calc_sfr
from threadcount.procedures import calculate_outflow_velocity as calc_outvel
from threadcount.procedures import calculate_mass_outflow as calc_mout


def read_velcuts_results(filename):
    """Reads in the results from the velocity cuts method

    Parameters
    ----------
    filename : str
        The file location

    Returns
    -------
    dict
        A dictionary containing the results from the velocity cuts method
    """
    # read in the dictionary
    vel_cuts_dict = tc.fit.ResultDict.loadtxt(filename)

    return vel_cuts_dict

def extract_from_comments(comment_lines, search_string):
    """
    Extracts info from the comments in the threadcount output text file
    Copied from a extract_wcs() in analyze_outflow_extent.py
    """
    #search_string = "wcs_step:"
    wcs_line = [x for x in comment_lines if x.startswith(search_string)][0]
    return eval(wcs_line[len(search_string) :].strip().replace(" ", ","))


def create_filelist(folder_name, file_expression):
    # create the list 
    file_list = glob.glob(folder_name+file_expression)

    return file_list 


def plot_some_stuff(file_list_vel_cuts, file_list_tc):

    # create figure
    fig, ax = plt.subplots(1,1)

    # iterate through file list 
    for file in file_list:
        # read in file 
        vel_cuts_dict = read_velcuts_results(file)

        # plot the things 
        ax.scatter(vel_cuts_dict['total_residual_flux'], vel_cuts_dict['high_velocity_outflow'], c='blue')

    plt.show()


def main():
    OIII_file_list = create_filelist("/Volumes/BronsData/nonparametric_outflow_results/", "*final_highvel_[*_dict.txt")
    Hbeta_file_list = create_filelist("/Volumes/BronsData/nonparametric_outflow_results/", "*final_highvel_Hb*_dict.txt")

    tc_res_file_list_Hbeta = create_filelist("/Volumes/BronsData/DUVET_sample_paper/threadcount_results/", "*_final_4861_mc_best_fit.txt")
    tc_res_file_list_OIII = create_filelist("/Volumes/BronsData/DUVET_sample_paper/threadcount_results/", "*_final_5007_mc_best_fit.txt")

    stellar_mass_list = create_filelist("/Volumes/BronsData/DUVET_sample_paper/final_results/", "*_stellar_mass_results.txt")

    #OIII_file_list = create_filelist("/Volumes/Bron's Data/nonparametric_outflow_results/", "iras08*final_highvel_[*_dict.txt")
    #Hbeta_file_list = create_filelist("/Volumes/Bron's Data/nonparametric_outflow_results/", "iras08*final_highvel_Hb*_dict.txt")

    #tc_res_file_list = create_filelist("/Volumes/Bron's Data/DUVET_sample_paper/threadcount_results/", "iras08*_final_4861_mc_best_fit.txt")
    

    # make sure all the lists are sorted alphabetically
    OIII_file_list = sorted(OIII_file_list, key=str.lower)
    Hbeta_file_list = sorted(Hbeta_file_list, key=str.lower)
    tc_res_file_list_Hbeta = sorted(tc_res_file_list_Hbeta, key=str.lower)
    tc_res_file_list_OIII = sorted(tc_res_file_list_OIII, key=str.lower)
    stellar_mass_list = sorted(stellar_mass_list, key=str.lower)

    # create figure
    fig, ax = plt.subplots(1, 3, figsize=(7,3), layout='constrained')

    # create main sequence figure
    fig_ms, ax_ms = plt.subplots(1, 1)

    # sort through the galaxy names and match the vel cuts results to the tc results files 
    for i, file in enumerate(OIII_file_list):
        # read in the files 
        OIII_dict = read_velcuts_results(file)
        #Hbeta_dict = read_velcuts_results(Hbeta_file_list[i])
        tc_dict_hbeta = read_velcuts_results(tc_res_file_list_Hbeta[i])
        tc_dict_OIII = read_velcuts_results(tc_res_file_list_OIII[i])

        print('Using the following files:')
        print(file)
        #print(Hbeta_file_list[i])
        print(tc_res_file_list_Hbeta[i])
        print(tc_res_file_list_OIII[i])
        print(' ')

        #get the comment lines
        comment_lines = tc_dict_hbeta.comment.split('\n')

        #get the WCS from the comment lines
        wcs_step = extract_from_comments(comment_lines, 'wcs_step:')

        #get the redshift from the comment lines
        z = extract_from_comments(comment_lines, 'z_set:')

        # calculate the SFR surface density
        sfr, sfr_err, total_sfr, sigma_sfr, sigma_sfr_err = calc_sfr.calc_sfr(tc_dict_hbeta, z, wcs_step, include_outflow=False)

        #calculate the outflow velocity
        vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(tc_dict_OIII)

        # calculate the outflow flux 
        sigma_out, sigma_out_err = calc_mout.calc_mass_outflow_flux(tc_dict_hbeta, z, wcs_step)

        # calculate mass loading factor 
        mlf = sigma_out/sigma_sfr

        # create high vel mask
        high_vel_mask = OIII_dict['high_velocity_outflow'] > 0.0

        # plot the sigma_sfr against vel_out
        #ax[0].scatter(np.log10(sigma_sfr.value), np.log10(vel_out), alpha=0.5)#, c='grey', alpha=0.5)

        ax[0].scatter(np.log10((sigma_sfr[high_vel_mask]).value), np.log10(vel_out[high_vel_mask]), 
                      c=np.log10(OIII_dict['high_velocity_outflow'][high_vel_mask]),
                      cmap='Blues', 
                      vmin=-2.0,
                      vmax=2.0,
                      alpha=0.3, 
                      marker='o'
                      )
        ax[0].scatter(np.log10((sigma_sfr[~high_vel_mask]).value), np.log10(vel_out[~high_vel_mask]), 
                      c=np.log10(OIII_dict['low_velocity_outflow'][~high_vel_mask]),
                      cmap='Reds', 
                      vmin=-2.0,
                      vmax=2.0,
                      alpha=0.3, 
                      marker='s'
                      )

        # plot the sigma_sfr against sigma_out
        #ax[1].scatter(np.log10(sigma_sfr.value), np.log10(sigma_out.value), alpha=0.5)#, c='grey', alpha=0.5)

        ax[1].scatter(np.log10((sigma_sfr[high_vel_mask]).value), np.log10((sigma_out[high_vel_mask]).value), 
                      c=np.log10(OIII_dict['high_velocity_outflow'][high_vel_mask]),
                      cmap='Blues', 
                      vmin=-2.0,
                      vmax=2.0,
                      alpha=0.3, 
                      marker='o'
                      )
        ax[1].scatter(np.log10((sigma_sfr[~high_vel_mask]).value), np.log10((sigma_out[~high_vel_mask]).value), 
                      c=np.log10(OIII_dict['low_velocity_outflow'][~high_vel_mask]),
                      cmap='Reds', 
                      vmin=-2.0,
                      vmax=2.0,
                      alpha=0.3, 
                      marker='s')
        
        high_vel_flux_col = ax[2].scatter(np.log10((sigma_sfr[high_vel_mask]).value), np.log10((mlf[high_vel_mask]).value), 
                      c=np.log10(OIII_dict['high_velocity_outflow'][high_vel_mask]),
                      cmap='Blues', 
                      vmin=-2.0,
                      vmax=2.0,
                      alpha=0.3, 
                      marker='o'
                      )
        low_vel_flux_col = ax[2].scatter(np.log10((sigma_sfr[~high_vel_mask]).value), np.log10((mlf[~high_vel_mask]).value), 
                      c=np.log10(OIII_dict['low_velocity_outflow'][~high_vel_mask]),
                      cmap='Reds', 
                      vmin=-2.0,
                      vmax=2.0,
                      alpha=0.3, 
                      marker='s')
        
        # do the main sequence plot
        stellar_mass = np.loadtxt(stellar_mass_list[i])

        ax_ms.scatter(np.log10(stellar_mass[high_vel_mask]), np.log10((sigma_sfr[high_vel_mask]).value),
                    c=np.log10(OIII_dict['high_velocity_outflow'][high_vel_mask]),
                    cmap='Blues', 
                    vmin=-2.0,
                    vmax=2.0,
                    alpha=0.3, 
                    marker='o'
                    )
        ax_ms.scatter(np.log10(stellar_mass[~high_vel_mask]), np.log10((sigma_sfr[~high_vel_mask]).value), 
                    c=np.log10(OIII_dict['low_velocity_outflow'][~high_vel_mask]),
                    cmap='Reds', 
                    vmin=-2.0,
                    vmax=2.0,
                    alpha=0.3, 
                    marker='s'
                    )
        

    # add colourbars 
    #divider = make_axes_locatable(ax[2])
    #cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(high_vel_flux_col, orientation='vertical', ax=ax[2], label=r'Log High Velocity Flux'+'\n'+'[10$^{-16}$ erg cm$^{-2}$ s$^{-1}$]')
    #cax = divider.append_axes('right', size='5%', pad=0.5)
    fig.colorbar(low_vel_flux_col, orientation='vertical', ax=ax[2], label=r'Log Low Velocity Flux')

    fig_ms.colorbar(high_vel_flux_col, orientation='vertical', ax=ax_ms, label=r'Log High Velocity Flux'+'\n'+'[10$^{-16}$ erg cm$^{-2}$ s$^{-1}$]')
    fig_ms.colorbar(low_vel_flux_col, orientation='vertical', ax=ax_ms, label=r'Log Low Velocity Flux')
    

    # add figure labels
    ax[0].set_xlabel(r'Log $\Sigma_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_xlabel(r'Log $\Sigma_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_xlabel(r'Log $\Sigma_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    
    ax[0].set_ylabel(r'Log $v_{\rm out}$ [km s$^{-1}$]')
    ax[1].set_ylabel(r'Log $\dot{\Sigma}_{\rm out}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_ylabel(r'Log $\eta$')

    ax_ms.set_xlabel(r'Log $\Sigma_\star$ [M$_\odot$]')
    ax_ms.set_ylabel(r'Log $\Sigma_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

