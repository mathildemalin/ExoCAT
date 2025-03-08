#     Functions for measuring the photometry of point sources and planets using MIRI coronagraph data.
#  ____________________________________________________________________________________________

import webbpsf
import matplotlib.pyplot as plt
import numpy as np
from photutils.aperture import CircularAperture
import scipy
import scipy.optimize as optimize
from scipy.stats import t

# import other packages
from ExoMIRI.utils import * 
from ExoMIRI.miri_corono import * # for FPNeg methods



def generate_psf(MIRI_filter, MIRI_image_mask, position_fov, fov_arcsec=20):
    """
    Generate a PSF for a given MIRI filter, coronagraph mask, and position.
    """
    miri = webbpsf.MIRI()
    miri.filter = MIRI_filter
    miri.image_mask = MIRI_image_mask
    miri.pupil_mask = 'MASKFQPM'
    
    # Apply shear shift
    shearx, sheary = 1.5 / np.sqrt(2), 1.5 / np.sqrt(2)
    miri.options['pupil_shift_x'] = shearx / 100  # Convert shear to fractional shift
    miri.options['pupil_shift_y'] = sheary / 100
    
    # Set coronagraph shift
    miri.options['coron_shift_x'] = position_fov[0]
    miri.options['coron_shift_y'] = position_fov[1]
    
    psf = miri.calc_psf('outfile.fits', fov_arcsec=fov_arcsec)  # Generate PSF
    return psf[1].data

def measure_attenuation(MIRI_filter, MIRI_image_mask, position_fov):
    """
    Evaluate the attenuation and aperture correction based on the coronagraph used and 
    the position of the point source.
    
    Parameters:
        - MIRI_filter (str): 'F1065C', 'F1140C', or 'F1550C'
        - MIRI_image_mask (str): 'FQPM1065', 'FQPM1140', or 'FQPM1550'
        - position_fov (tuple): (x, y) position of the point source in the detector frame
    
    Returns:
        - attenuation_full (float): Full field attenuation
        - attenuation (float): Attenuation in the aperture
        - aperture_corr (float): Aperture correction factor
    """
    # Define filter wavelengths
    filter_wavelengths = {'F1065C': 10.575, 'F1140C': 11.30, 'F1550C': 15.50}
    
    if MIRI_filter not in filter_wavelengths:
        raise ValueError("Invalid filter name. Choose from 'F1065C', 'F1140C', 'F1550C'")
    
    wavelength_filter = filter_wavelengths[MIRI_filter]
    pix_size = 0.11  # Pixel size in arcseconds
    D = 6.57  # JWST primary mirror diameter in meters
    lambdaD_pix = ((wavelength_filter * 1e-6) / D) * (180 * 3600 / np.pi) / pix_size  # Lambda/D in pixels
    
    # Generate PSFs
    psf_away = generate_psf(MIRI_filter, MIRI_image_mask, (10, 10))  # Off-axis PSF
    psf_planet = generate_psf(MIRI_filter, MIRI_image_mask, position_fov)  # On-axis PSF
    
    # Measure peak positions
    pos = np.unravel_index(np.nanargmax(psf_planet), psf_planet.shape)
    pos_away = np.unravel_index(np.nanargmax(psf_away), psf_away.shape)
    
    # Measure flux in 1.5 * lambda/D aperture
    aperture = CircularAperture(pos, r=1.5 * lambdaD_pix)
    aperture_away = CircularAperture(pos_away, r=1.5 * lambdaD_pix)
    
    flux_psf = aperture.to_mask(method='center').multiply(psf_planet).sum()
    flux_psf_away = aperture_away.to_mask(method='center').multiply(psf_away).sum()
    
    # Measure total flux
    flux_full = np.nansum(psf_planet)
    flux_full_away = np.nansum(psf_away)
    
    # Compute attenuation and aperture correction
    attenuation = flux_psf / flux_psf_away if flux_psf_away != 0 else np.nan
    attenuation_full = flux_full / flux_full_away if flux_full_away != 0 else np.nan
    aperture_corr = flux_full / flux_psf if flux_psf != 0 else np.nan
    
    return attenuation_full, attenuation, aperture_corr

def aperture_photom(pos_planet, data, aperture):
    """
    Perform aperture photometry by summing flux within a circular aperture, 
    ensuring that only positive flux values are included.
    
    Parameters:
    - pos_planet: Tuple (x, y) coordinates of the planet in the image.
    - data: 2D array representing the image data.
    - aperture: Radius of the circular aperture.
    
    Returns:
    - flux: Total flux within the aperture, excluding negative values.
    """
    disque_b = CircularAperture((pos_planet[0], pos_planet[1]), r=aperture)
    mask = disque_b.to_mask(method='center').multiply(data)
    flux = np.nansum(mask[mask > 0])  # Exclude negative values for robustness.
    return flux

def aperture_photom2(pos_planet, data, aperture):
    """
    Perform aperture photometry by summing flux within a circular aperture.
    
    Parameters:
    - pos_planet: Tuple (x, y) coordinates of the planet in the image.
    - data: 2D array representing the image data.
    - aperture: Radius of the circular aperture.
    
    Returns:
    - flux: Total flux within the aperture.
    """
    disque_b = CircularAperture((pos_planet[0], pos_planet[1]), r=aperture)
    flux = disque_b.to_mask(method='center').multiply(data).sum()
    return flux

def measure_attenuation_full(MIRI_filter, MIRI_image_mask, position_fov):
    """
    Evaluate the attenuation and aperture correction for different aperture sizes,
    useful for contrast-based methods in MIRI coronagraphic data.

    Parameters:
    - MIRI_filter (str): Filter used, options are 'F1065C', 'F1140C', 'F1550C'.
    - MIRI_image_mask (str): Corresponding coronagraph mask, options are 'FQPM1065', 'FQPM1140', 'FQPM1550'.
    - position_fov (tuple): (x, y) position of the point source in the detector frame.

    Returns:
    - attenuation (float): Attenuation factor within a 1.5 λ/D aperture.
    - attenuation_full (float): Attenuation factor for the entire field of view.
    - aperture_corr (float): Aperture correction factor for a 1.5 λ/D aperture.
    - aperture_corr_5 (float): Aperture correction factor for a 5 λ/D aperture.
    - aperture_corr_15 (float): Aperture correction factor for a 15 λ/D aperture.
    """

    # Define the wavelength associated with the given filter
    filter_wavelengths = {'F1065C': 10.575, 'F1140C': 11.30, 'F1550C': 15.50}
    wavelength_filter = filter_wavelengths.get(MIRI_filter)

    if wavelength_filter is None:
        raise ValueError("Invalid filter name. Choose from 'F1065C', 'F1140C', or 'F1550C'.")

    pix_size = 0.11  # Pixel size in arcseconds
    D = 6.57  # Telescope diameter in meters
    lambdaD_pix = ((wavelength_filter * 10**-6) / D) * (180 * 3600 / np.pi) / pix_size  # λ/D in pixels

    # Generate PSFs using the provided function
    psf_away = generate_psf(MIRI_filter, MIRI_image_mask, (10, 10))
    psf_planet = generate_psf(MIRI_filter, MIRI_image_mask, position_fov)

    # Measure the peak position of the PSFs
    pos = np.unravel_index(np.nanargmax(psf_planet), psf_planet.shape)
    pos_away = np.unravel_index(np.nanargmax(psf_away), psf_away.shape)

    # Measure flux within different apertures using aperture_photom()
    flux_psf = aperture_photom(pos, psf_planet, 1.5 * lambdaD_pix)
    flux_psf_away = aperture_photom(pos_away, psf_away, 1.5 * lambdaD_pix)
    flux_full = np.nansum(psf_planet)
    flux_full_away = np.nansum(psf_away)

    # Compute attenuation factors
    attenuation = flux_psf / flux_psf_away
    attenuation_full = flux_full / flux_full_away

    # Compute aperture correction factor for 1.5 λ/D aperture
    aperture_corr = flux_full / flux_psf

    # Compute aperture correction factors for 5 λ/D and 15 λ/D apertures
    aperture_corr_5 = flux_full / aperture_photom(pos, psf_planet, 5 * lambdaD_pix)
    aperture_corr_15 = flux_full / aperture_photom(pos, psf_planet, 15 * lambdaD_pix)

    return attenuation, attenuation_full, aperture_corr, aperture_corr_5, aperture_corr_15

###########################################################################
##   Version with FPNeg before stellar subtraction: More reliable with PCA
##
def best_model_planet_sub(data, data_refs, new_center, pos_planet_b, wavelength, model_PSF, method_subtraction, Kopt=None, priors=(1e2, 0, 0)):
    """ 
    Measure the best-fit PSF model for a planet by subtracting an optimized model
    before performing stellar subtraction.

    Parameters:
    - data (numpy.ndarray): Target data array containing the observed image.
    - data_refs (numpy.ndarray): Reference library for PSF subtraction.
    - new_center (tuple): Center coordinates for both target and reference images.
    - pos_planet_b (tuple): Precomputed x, y position of the planet.
    - wavelength (float): Wavelength in microns (valid values: 10.57, 11.3, or 15.5).
    - model_PSF (numpy.ndarray): Simulated PSF model from WebbPSF.
    - method_subtraction (str): Stellar subtraction method; options:
        - 'classical_PCA': Standard PCA-based subtraction.
        - '4QPCA': PCA applied separately to each quadrant.
        - 'PCA_rings': PCA performed on concentric annuli.
        - 'opt_lin': Optimal linear combination of reference images.
        - 'opt_lin_4Q': Optimized reference combination with 4Q.
    - Kopt (int, optional): Number of PCA components to remove (depends on pre-analysis).
    - priors (tuple): Initial guess for fitting (flux, x_offset, y_offset).

    Returns:
    - numpy.ndarray: Best-fit PSF model for the planet.
    """

    pix_size = 0.11  # Pixel scale in arcseconds
    lambdaD_pix = ((wavelength * 10**-6) / 6.57) * (180 * 3600 / np.pi) / pix_size  # λ/D in pixels

    ### 1. Define the region of interest (ROI) for minimization
    h, w = np.shape(data)
    mask_planets = create_circular_mask(h, w, center=(pos_planet_b[0], pos_planet_b[1]), radius=3. * lambdaD_pix)
    # This mask isolates the planetary signal, avoiding contamination from stellar residuals.

    ### 2. Define the cost function to minimize
    def function_min(P):
        """
        Function to minimize residuals by optimizing the model PSF parameters.
        The free parameters are flux, x_offset, and y_offset.
        """
        
        # Shift the PSF model to match the estimated planet position
        final_model = scipy.ndimage.shift(model_PSF, (P[1], P[2])) * P[0]
        
        # Compute "Fake Planet Negative" (FPNeg) by subtracting the shifted model from the data
        FPNeg_images = data - final_model

        # Apply stellar subtraction using the chosen method
        if method_subtraction == 'classical_PCA':
            data_sub = sub_PCA(FPNeg_images, data_refs, new_center, Ktrunc=Kopt)
        elif method_subtraction == '4QPCA':
            data_sub = sub_4Q_PCA(FPNeg_images, data_refs, new_center, Ktrunc=Kopt)
        elif method_subtraction == 'PCA_rings':
            data_sub = sub_PCA_rings(FPNeg_images, data_refs, new_center, Ktrunc=Kopt)
        elif method_subtraction == 'opt_lin':
            data_sub = optimize_refs(FPNeg_images, data_refs, new_center)
        elif method_subtraction == 'opt_lin_4Q':
            data_sub = optimize_refs_4Q(FPNeg_images, data_refs, new_center)
        else:
            print('Error: Invalid stellar subtraction method')
            return np.inf  # Return a large value to prevent invalid methods from being chosen

        # Apply the planet mask to extract residuals
        data_mask = np.ma.array(data_sub, mask=~mask_planets)
        model_mask = np.ma.array(final_model, mask=~mask_planets)

        # Extract non-masked pixels
        x_region, y_region = np.where(data_mask.mask == False)
        val_mask = [data_mask[x_region[i], y_region[i]] for i in range(len(x_region))]
        val_data = [model_mask[x_region[i], y_region[i]] for i in range(len(x_region))]

        # Compute the residual sum of squares (RSS) as the minimization metric
        res = np.nansum((np.array(val_mask) - np.array(val_data)) ** 2)
        return res

    # Optimize the PSF parameters using Nelder-Mead minimization
    P = optimize.minimize(function_min, x0=priors, method='Nelder-Mead')

    # Extract the best-fit parameters
    flux_factor, x_offset, y_offset = P["x"]
    print('Best-fit parameters:')
    print(f'Flux: {flux_factor}, X Offset: {x_offset}, Y Offset: {y_offset}')

    ### 3. Compute the final best-fit model using the optimized parameters
    final_model = scipy.ndimage.shift(model_PSF, (x_offset, y_offset)) * flux_factor * 2
    return final_model

###########################################################################
##  Version for data with stellar diffraction already subtracted
## 
def best_model_planet(data_sub, pos_planet_b, wavelength, model_PSF, priors=(1e2, 0, 0)):
    """ 
    Compute the best-fit PSF model for a planetary signal.
    
    Parameters:
    data_sub (numpy.ndarray): Subtracted data array containing the planetary signal.
    pos_planet_b (tuple): (x, y) position of the planet (precomputed).
    wavelength (float): Wavelength in microns (10.57, 11.3, or 15.5).
    model_PSF (numpy.ndarray): WebbPSF-generated model PSF.
    priors (tuple): Initial guesses for flux, x-offset, and y-offset.

    Returns:
    numpy.ndarray: Best-fit PSF model.
    """

    # Pixel scale in arcseconds per pixel
    pix_size = 0.11

    # Compute the diffraction limit in pixels (λ/D)
    lambdaD_pix = ((wavelength * 10**-6) / 6.57) * (180 * 3600 / np.pi) / pix_size
    
    # Define a circular mask around the planet to minimize residuals
    h, w = np.shape(data_sub)
    mask_planets = create_circular_mask(h, w, center=(pos_planet_b[0], pos_planet_b[1]), radius=3. * lambdaD_pix)  
    
    # Function to minimize the residuals between the model and data
    def function_min(P):
        # Generate the shifted PSF model with given parameters (flux, x-offset, y-offset)
        final_model = scipy.ndimage.shift(model_PSF, (P[1], P[2])) * P[0]

        # Apply the mask to isolate the planetary region
        data_mask = np.ma.array(data_sub, mask=~mask_planets)
        model_mask = np.ma.array(final_model, mask=~mask_planets)

        # Extract unmasked pixel values for comparison
        x_region, y_region = np.where(data_mask.mask == False)
        val_mask = [data_mask[x_region[i], y_region[i]] for i in range(len(x_region))]
        val_data = [model_mask[x_region[i], y_region[i]] for i in range(len(x_region))]
    
        # Compute the sum of squared residuals as the cost function
        res = np.nansum((np.array(val_mask) - np.array(val_data)) ** 2)
        return res

    # Optimize parameters using the Nelder-Mead method
    P = optimize.minimize(function_min, x0=priors, method='Nelder-Mead')  
    flux_factor, x_offset, y_offset = P["x"]
    
    # Print the best-fit parameters
    print('Best parameters:')
    print(flux_factor, x_offset, y_offset)
     
    # Generate the final best-fit model with optimized parameters
    final_model = scipy.ndimage.shift(model_PSF, (x_offset, y_offset)) * flux_factor 
    
    return final_model

########################################################################### 
## Function to fit the stellar coronagraphic image with a WebbPSF model model 
##
def best_corono_model(data, miri_model, center_corono, priors=(1e7, 0, 0), rmin=1, rmax=5, bounds=None, plot_model=False, vmin=-1, vmax=4):
    """ 
    Measure the best-fit coronagraphic image model by optimizing flux and position.

    Parameters:
    - data (numpy.ndarray): Observed target image.
    - miri_model (numpy.ndarray): WebbPSF coronagraphic model.
    - center_corono (tuple): x, y coordinates of the coronagraph center.
    - priors (tuple, optional): Initial guess for fitting (flux, x_offset, y_offset).
                                Default is (1e7, 0, 0).
    - rmin, rmax (float, optional): Region (in arcseconds) where residual minimization is performed.
                                    Default is rmin=1, rmax=5, targeting the stellar PSF region.
    - bounds (tuple, optional): Bounds for optimization (default is None).
    - plot_model (bool, optional): If True, plots the observed image, model, and residuals.
    - vmin, vmax (float, optional): Display range for image plots.

    Returns:
    - final_model (numpy.ndarray): Optimized coronagraphic PSF model.
    - flux_factor (float): Best-fit flux scaling factor.
    """

    pix_size = 0.11  # Pixel scale in arcseconds
    h, w = np.shape(miri_model)

    # Define the optimization region as an annular mask to focus on the stellar PSF area
    mask_coro = createAnnularMask(h, w, (center_corono[1], center_corono[0]), rmin / pix_size, rmax / pix_size)

    def function_min(P):    
        """
        Compute residuals between the observed data and the model.
        The model is shifted and scaled before comparison.
        """
        # Generate the shifted and scaled model
        corono_model = scipy.ndimage.shift(miri_model, (P[1], P[2])) * P[0]
        
        # Apply the mask to focus on the region of interest
        data_mask = np.ma.array(data, mask=~mask_coro)
        model_mask = np.ma.array(corono_model, mask=~mask_coro)

        # Compute residual sum of squares (RSS) as the cost function
        res = np.sum((data_mask - model_mask) ** 2)
        return res

    # Perform optimization using the Nelder-Mead method
    P = optimize.minimize(function_min, x0=priors, method='Nelder-Mead', bounds=bounds)
    flux_factor, x_offset, y_offset = P["x"]

    # Compute the final best-fit model using the optimized parameters
    final_model = scipy.ndimage.shift(miri_model, (x_offset, y_offset)) * flux_factor

    # If plotting is enabled, visualize the original image, model, and residuals
    if plot_model:
        print('Best-fit parameters: Flux Factor, X Offset, Y Offset')
        print(flux_factor, x_offset, y_offset)

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        # Plot the observed data
        im = ax[0].imshow(data, origin="lower", vmin=vmin, vmax=vmax)
        ax[0].plot(center_corono[1], center_corono[0], 'rP')  # Mark the coronagraph center
        cbar = fig.colorbar(im, ax=ax[0], orientation="vertical")
        cbar.set_label('DN/s', rotation=270, labelpad=10, fontsize=16)

        # Plot the optimized model
        im = ax[1].imshow(final_model, origin="lower", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax[1], orientation="vertical")
        cbar.set_label('DN/s', rotation=270, labelpad=10, fontsize=16)

        # Plot the residuals (data - model)
        im = ax[2].imshow(data - final_model, origin="lower", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax[2], orientation="vertical")
        cbar.set_label('DN/s', rotation=270, labelpad=10, fontsize=16)

        plt.show()

    return final_model, flux_factor

###########################################################################
## Generate off-axis PSF fitted on the data + provide the max to nomralise contrast curves

def best_off_axis_PSF(data, center_corono, MIRI_filter, priors=(1e5,0,0), bounds=None, rmin=1, rmax=5, plot_model=True, vmin=-1, vmax=4):
    """
    Computes the best off-axis Point Spread Function (PSF) for MIRI coronagraphic imaging.
    
    Parameters:
    - data: 2D numpy array representing the input image.
    - center_corono: Tuple of (x, y) coordinates for the coronagraph center.
    - MIRI_filter: String indicating the MIRI filter ('F1065C', 'F1140C', 'F1550C').
    - priors: Tuple representing prior values for model fitting.
    - bounds: Bounds for the model fitting (default: None).
    - rmin, rmax: Radial limits for fitting (default: 1 to 5 pixels).
    - plot_model: Boolean flag to indicate whether to plot the model.
    - vmin, vmax: Limits for visualization.
    
    Returns:
    - miri_psf_off_axis_star: 2D numpy array representing the off-axis PSF.
    - max_off_axis_psf: Maximum value of the off-axis PSF in appropriate units.
    """
    
    pixel_area = 2.84403609523084E-13  # Pixel area in steradians
    
    # Assign wavelength and mask based on the filter
    if MIRI_filter == 'F1065C':
        wavelength_filter = 10.575  # Wavelength in microns
        MIRI_image_mask = 'FQPM1065'
    elif MIRI_filter == 'F1140C':
        wavelength_filter = 11.30
        MIRI_image_mask = 'FQPM1140'
    elif MIRI_filter == 'F1550C':
        wavelength_filter = 15.50
        MIRI_image_mask = 'FQPM1550'
    else:
        raise ValueError("Error in the filter name. Must be one of: 'F1065C', 'F1140C', 'F1550C'")
    
    # Define pixel scale and diffraction parameter
    pix_size = 0.11  # Pixel size in arcseconds
    D = 6.57  # Telescope aperture in meters
    lambdaD_pix = ((wavelength_filter * 1e-6) / D) * (180 * 3600 / np.pi) / pix_size  # Lambda/D in pixels
    
    # Generate a MIRI PSF model
    miri = webbpsf.MIRI()
    miri.filter = MIRI_filter
    miri.image_mask = MIRI_image_mask
    miri.pupil_mask = 'MASKFQPM'
    
    # Apply shear to the pupil mask
    shearx, sheary = 1.5 / np.sqrt(2), 1.5 / np.sqrt(2)
    miri.options['pupil_shift_x'] = shearx / 100  # Convert shear amount to fraction
    miri.options['pupil_shift_y'] = sheary / 100
    
    # Compute the PSF
    miri_psf_F1140 = miri.calc_psf('outfile.fits', fov_arcsec=32)
    miri_psf_images = miri_psf_F1140[1].data
    
    # Center the model on the data
    center_simu = np.array([miri_psf_images.shape[0] // 2, miri_psf_images.shape[1] // 2])
    offset_center = center_simu - center_corono
    miri_model_red = scipy.ndimage.shift(np.nan_to_num(miri_psf_images), -offset_center)
    
    # Crop the field of view to match the data size
    a = data.shape[0] // 2  # Assuming square image
    miri_model = miri_model_red[:a*2, :a*2]
    
    # Fit the model to the data
    model_corono, flux_factor = best_corono_model(
        data, miri_model, center_corono, priors=priors, bounds=bounds, rmin=rmin, rmax=rmax,
        plot_model=plot_model, vmin=vmin, vmax=vmax
    )
    
    # Generate the off-axis PSF
    miri = webbpsf.MIRI()
    miri.filter = 'F1140C'
    miri.image_mask = 'FQPM1140'
    miri.options['coron_shift_x'] = 10
    miri.options['coron_shift_y'] = 10
    
    # Apply shear again
    miri.options['pupil_shift_x'] = shearx / 100
    miri.options['pupil_shift_y'] = sheary / 100
    
    # Compute off-axis PSF
    miri_psf_F1140 = miri.calc_psf('outfile.fits', fov_arcsec=32)
    miri_psf_off_axis = miri_psf_F1140[1].data
    miri_psf_off_axis_star = miri_psf_off_axis * flux_factor
    
    # Compute max off-axis PSF value in appropriate units
    max_value_off_axis_psf = np.nanmax(miri_psf_off_axis_star)  # In MJy/str
    max_value_off_axis_psf *= pixel_area * 1e6 * 1e-26 * 299792458e6 / (wavelength_filter ** 2)
    sum_value_off_axis_psf = np.nansum(miri_psf_off_axis_star)
    
    return miri_psf_off_axis_star, max_value_off_axis_psf, sum_value_off_axis_psf



def plot_photometry(data_sub, final_model_photom, new_center, pos_planet_b, output_path, name_planet, method_subtraction, 
                    cplanet="darkorange", ccorono='lime', stretch="asinh", starsize=80, starsize2=60, lp=8, a=30, vmin=-0.1, vmax=1):
    """
    Plots a 3-panel photometry comparison of the original data, the model, and their difference.

    Parameters:
    - data_sub: 2D numpy array, the original data.
    - final_model_photom: 2D numpy array, the model data.
    - new_center: Tuple (x, y), coordinates of the star.
    - pos_planet_b: Tuple (x, y), coordinates of the planet.
    - output_path: str, directory where the output image will be saved.
    - name_planet: str, name of the planet.
    - method_subtraction: str, subtraction method identifier.
    - Other optional parameters control aesthetics and limits.
    """
    xlim = [int(pos_planet_b[0] - a), int(pos_planet_b[0] + a)]
    ylim = [int(pos_planet_b[1] - a), int(pos_planet_b[1] + a)]
    
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    
    for i, data in enumerate([data_sub, final_model_photom, data_sub - final_model_photom]):
        im = ax[i].imshow(data, origin='lower', cmap="BuGn_r", vmin=vmin, vmax=vmax)
        ax[i].scatter(new_center[0], new_center[1], marker='*', color=ccorono, s=starsize)
        cbar = fig.colorbar(im, ax=ax[i], orientation="vertical")
        cbar.set_label('MJy/s', rotation=270, labelpad=lp, fontsize=12)
        ax[i].set_xlim(xlim[0], xlim[1])
        ax[i].set_ylim(ylim[0], ylim[1])
        ax[i].set_xticklabels([], color='w')
        ax[i].set_yticklabels([], color='w')
        ax[i].tick_params(direction='in', width=1.2, size=5)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f"{output_path}{name_planet}_model_photom_{method_subtraction}.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()




###########################################################################
## Practical functions that effectively handle NaN values and statistics, from pyKLIP (Wang et al, 2015, Astrophysics Source Code Library, record ascl:1506.001).

def nan_gaussian_filter(img, sigma, ivar=None):
    """
    Gaussian low-pass filter that handles nans

    Args:
        img: 2-D image
        sigma: float specifiying width of Gaussian
        ivar: inverse variance frame for the image, optional

    Returns:
        filtered: 2-D image that has been smoothed with a Gaussian

    """
    if ivar is None:
        ivar = np.ones(img.shape)
    if img.shape != ivar.shape or len(img.shape) != 2:
        raise ValueError("image and ivar must be 2D ndarrays of the same shape")
    # make a copy to mask with nans
    masked = np.copy(img)
    masked_ivar = np.copy(ivar)
    nan_locs = np.where(np.isnan(img) | np.isnan(ivar))
    masked[nan_locs] = 0
    masked_ivar[nan_locs] = 0

    # filter the image
    filtered = scipy.ndimage.gaussian_filter(masked * masked_ivar, sigma=sigma, truncate=4)

    # because of NaNs, we need to renormalize the gaussian filter, since NaNs shouldn't contribute
    filter_norm = scipy.ndimage.gaussian_filter(masked_ivar, sigma=sigma, truncate=4)
    filtered /= filter_norm
    filtered[nan_locs] = np.nan

    # for some reason, the fitlered image peak pixel fluxes get decreased by 2
    # (2020-09-10: checking the gain for each image, it seems this is no longer a problem, next line is commented out)
    #filtered *= 2

    return filtered

def meas_contrast(dat, iwa, owa, resolution, center=None, low_pass_filter=True):
    
    """
    Measures the contrast in the image. Image must already be in contrast units and should be corrected for algorithm
    thoughput.

    Args:
        dat: 2D image - already flux calibrated
        iwa: inner working angle
        owa: outer working angle
        resolution: size of noise resolution element in pixels (for speckle noise ~ FWHM or lambda/D)
                    but it can be 1 pixel if limited by pixel-to-pixel noise. 
        center: location of star (x,y). If None, defaults the image size // 2.
        low_pass_filter: if True, run a low pass filter.
                         Can also be a float which specifices the width of the Gaussian filter (sigma).
                         If False, no Gaussian filter is run

    Returns:
        (seps, contrast): tuple of separations in pixels and corresponding 5 sigma FPF

    """

    if center is None:
        starx = dat.shape[1]//2
        stary = dat.shape[0]//2
    else:
        starx, stary = center

    # figure out how finely to sample the radial profile
    dr = resolution/2.0
    numseps = int((owa-iwa)/dr)
    # don't want to start right at the edge of the occulting mask
    # but also want to well sample the contrast curve so go at twice the resolution
    seps = np.arange(numseps) * dr + iwa + resolution/2.0
    dsep = resolution
    # find equivalent Gaussian PSF for this resolution

    # run a low pass filter on the data, check if input is boolean or a number
    if not isinstance(low_pass_filter, bool):
        # manually passed in low pass filter size
        sigma = low_pass_filter
        filtered = nan_gaussian_filter(dat, sigma)
    elif low_pass_filter:
        # set low pass filter size to be same as resolution element
        sigma = dsep / 2.355  # assume resolution element size corresponds to FWHM
        filtered = nan_gaussian_filter(dat, sigma)
    else:
        # no filtering
        filtered = dat

    contrast = []
    # create a coordinate grid
    x,y = np.meshgrid(np.arange(float(dat.shape[1])), np.arange(float(dat.shape[0])))
    r = np.sqrt((x-starx)**2 + (y-stary)**2)
    theta = np.arctan2(y-stary, x-starx) % 2*np.pi
    for sep in seps:
        # calculate noise in an annulus with width of the resolution element
        annulus = np.where((r < sep + resolution/2) & (r > sep - resolution/2))
        noise_mean = np.nanmean(filtered[annulus])
        noise_std = np.nanstd(filtered[annulus], ddof=1)
        # account for small sample statistics
        num_samples = int(np.floor(2*np.pi*sep/resolution))

        # find 5 sigma flux using student-t statistics
        # Correction based on Mawet et al. 2014
        fpf_flux = t.ppf(0.99999971334, num_samples-1, scale=noise_std) * np.sqrt(1 + 1./num_samples) + noise_mean
        contrast.append(fpf_flux)

    return seps, np.array(contrast)

