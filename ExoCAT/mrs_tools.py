import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, CircularAnnulus
from astropy.modeling import models, fitting
from astropy.modeling.models import Gaussian2D


########################################################################################
# Functions to load all cubes and spectra contained in a directory for MIRI-MRS data.

def load_cube(path):
    with fits.open(path) as hdu:
        cube = hdu[1].data
        hdr = hdu[1].header
        err = hdu[2].data
   
    l_range = np.linspace(float(hdr["CRVAL3"]), float(hdr["CRVAL3"]) + (cube.shape[0]) * float(hdr["CDELT3"]),
                          endpoint=True, num=cube.shape[0])
    return cube, l_range, err

def load_cube_dir(path, idx = 1, suffixe = 's3d'):
    files = [f for f in os.listdir(path) if f.split("_")[-1]==f"{suffixe}.fits"]
    file_id = [f.split("_")[idx] for f in files]
    # print(file_id)
    band_cubes = {"ch1-short": "1A", "ch1-medium": "1B", "ch1-long": "1C",
                  "ch2-short": "2A", "ch2-medium": "2B", "ch2-long": "2C",
                  "ch3-short": "3A", "ch3-medium": "3B", "ch3-long": "3C",
                  "ch4-short": "4A", "ch4-medium": "4B", "ch4-long": "4C"}
    out_dict = {}
    for b in band_cubes.keys():
        if b in file_id:
            f = files[np.where([b == s for s in file_id])[0][0]]
            # print(f"Loading cube for band {band_cubes[b]}: {f}..")
            c, l, err = load_cube(path+f)
            axes_data = {}
            with fits.open(path+f) as hdu:
                pxar = float(hdu["SCI"].header["PIXAR_SR"])
                pav3 = float(hdu["SCI"].header["PA_V3"])
                axis1 = float(hdu["SCI"].header["CRVAL1"])+\
                        (np.arange(c.shape[1])-float(hdu["SCI"].header["CRPIX1"]))*float(hdu["SCI"].header["CDELT1"])
                axis2 = float(hdu["SCI"].header["CRVAL2"]) + (
                    np.arange(c.shape[2]) - float(hdu["SCI"].header["CRPIX2"])) * float(hdu["SCI"].header["CDELT2"])
                axes_data["axis1"] = axis1
                axes_data["axis2"] = axis2
            out_dict[band_cubes[b]] = {"wav": l, "cube": c*pxar*1E+6, "err":err*pxar*1E+6,"filename": path+f, "pixar_sr":pxar, 'pav3':pav3,
                                       'axes':axes_data, "pxsc": hdu["SCI"].header["CDELT2"]*3600}
                                       #float(hdu["SCI"].header["CDELT1"])*units.deg.to(units.arcsec)}
    return out_dict



def load_spec(path):
    with fits.open(path) as hdu:
        data = hdu[1].data
        hdr = hdu[1].header
        err = hdu[2].data
    return data, err

def load_spec_dir(path, idx = 1, suffixe = 'x1d'):
    files = [f for f in os.listdir(path) if f.split("_")[-1]==f"{suffixe}.fits"]
    file_id = [f.split("_")[idx] for f in files]
    band_cubes = {"ch1-short": "1A", "ch1-medium": "1B", "ch1-long": "1C",
                  "ch2-short": "2A", "ch2-medium": "2B", "ch2-long": "2C",
                  "ch3-short": "3A", "ch3-medium": "3B", "ch3-long": "3C",
                  "ch4-short": "4A", "ch4-medium": "4B", "ch4-long": "4C"}
    out_dict = {}
    for b in band_cubes.keys():
        if b in file_id:
            f = files[np.where([b == s for s in file_id])[0][0]]
            d, err = load_spec(path+f)
            
            with fits.open(path+f) as hdu:
                wave = hdu[1].data["WAVELENGTH"]
                flux = hdu[1].data["FLUX"]
                flux_err = hdu[1].data["FLUX_ERROR"]
                
            out_dict[band_cubes[b]] = {"wav": wave, "flux":flux, "err":flux_err,"filename": path+f}
    return out_dict

########################################################################################
# Aperture photometry

def extract_spectrum(cubes, wavelengths, position_planet, pixel_arcsec, fwhm_size=1):
    """
    Extracts the flux within a PSF at each wavelength from a data cube.

    Parameters:
    cubes (3D array): Array of image slices (2D arrays) for each wavelength.
    wavelengths (1D array): Array of wavelength values corresponding to each image slice.
    position_planet (tuple): Position where to extract the flux in pixels (x, y).
    pixel_arcsec (float): Size of one pixel in arcsecs.
    fwhm_size (float, optional): Scaling factor for the FWHM size. Default is 1.

    Returns:
    tuple: Two 1D arrays containing the extracted flux and PSF sizes for each wavelength.
    """
    
    # MIRI/MRS PSF FWHM (Polychronis's values ; in agreement with Argyriou et al. 2023) 
    fwhm_poly = np.poly1d([1.15504083e-08, -1.70009986e-06,  8.73285027e-05, -2.16106801e-03,2.83057945e-02, -1.59613156e-01,  6.60276371e-01])

    # Calculate PSF sizes for each wavelength
    psf_sizes = fwhm_size * fwhm_poly(wavelengths) / pixel_arcsec
    
    # Extract flux for each wavelength
    flux_psf = [CircularAperture(position_planet, r=psf_radius).to_mask(method='center').multiply(cube_slice).sum() 
                for psf_radius, cube_slice in zip(psf_sizes, cubes)]

    return np.array(flux_psf), psf_sizes



def extract_spectrum_ring(cubes, wavelengths, position_planet, pixel_arcsec, rmin, rmax):
    """
    Extracts the flux within a ring (annular region) at each wavelength from a data cube.

    Parameters:
    cubes (3D array): Array of image slices (2D arrays) for each wavelength.
    wavelengths (1D array): Array of wavelength values corresponding to each image slice.
    position_planet (tuple): Position where to extract the flux in pixels (x, y).
    pixel_arcsec (float): Size of one pixel in arcsecs.
    rmin (float): Minimum radius of the annular region (in pixels).
    rmax (float): Maximum radius of the annular region (in pixels).

    Returns:
    tuple: Two 1D arrays containing the extracted flux and PSF sizes for each wavelength.
    """
    
    # MIRI/MRS PSF FWHM (Polychronis's values ; in agreement with Argyriou et al. 2023) 
    fwhm_poly = np.poly1d([1.15504083e-08, -1.70009986e-06,  8.73285027e-05, -2.16106801e-03,
                            2.83057945e-02, -1.59613156e-01,  6.60276371e-01])

    # Calculate PSF sizes for each wavelength
    psf_sizes_min = rmin * fwhm_poly(wavelengths) / pixel_arcsec
    psf_sizes_max = rmax * fwhm_poly(wavelengths) / pixel_arcsec
    
    flux_psf = []
    for psf_min, psf_max, cube_slice in zip(psf_sizes_min, psf_sizes_max, cubes):
        # Create the annular aperture (excluding the center)
        annulus = CircularAnnulus(position_planet, r_in=psf_min, r_out=psf_max)
        
        # Extract flux within the annulus region
        flux = annulus.to_mask(method='center').multiply(cube_slice).sum()
        flux_psf.append(flux)

    return np.array(flux_psf) #, psf_sizes_min, psf_sizes_max


########################################################################################

def measure_position_fit(cube, initial_position, plot=False, window=(5, 3)):
    """
    Measures the position of a source in a data cube by fitting a 2D Gaussian model.

    Parameters:
        cube (ndarray): 3D array where the first axis represents different frames.
        initial_position (tuple): Initial (x, y) coordinates for the position estimation.
        plot (bool, optional): If True, generates plots of the data, fit, and residuals.
        window (tuple, optional): (y_half_width, x_half_width) defining the crop size around initial position.

    Returns:
        list: Refined (x, y) position based on Gaussian fitting.
    """

    # Compute the median image along the cube axis
    median_projection = np.nanmedian(cube, axis=0)

    # Crop a small region around the initial position for fitting
    cropped_region = median_projection[
        initial_position[1] - window[0] : initial_position[1] + window[1],
        initial_position[0] - window[0] : initial_position[0] + window[1]
    ]

    # Identify the peak position within the cropped region
    max_y, max_x = np.where(cropped_region == np.nanmax(cropped_region))

    # Set initial parameters for the 2D Gaussian fit
    initial_gauss = Gaussian2D(
        amplitude=np.nanmax(cropped_region),  # Peak brightness
        x_mean=max_x[0],  # Initial guess for x center
        y_mean=max_y[0],  # Initial guess for y center
        x_stddev=1,  # Assumed standard deviation in x
        y_stddev=1,  # Assumed standard deviation in y
        theta=1.0  # Rotation angle
    )

    # Create a coordinate grid for fitting
    y, x = np.mgrid[:cropped_region.shape[0], :cropped_region.shape[1]]

    # Fit the Gaussian model to the data
    fitter = fitting.LevMarLSQFitter()
    fit_result = fitter(initial_gauss, x, y, cropped_region)

    # Compute the final fitted position, adjusting for cropping offset
    final_position = [
        fit_result.x_mean.value + (initial_position[0] - window[0]),
        fit_result.y_mean.value + (initial_position[1] - window[0])
    ]

    # Plot the data, fit, and residuals if requested
    if plot:
        print(fit_result)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Original cropped data
        im = ax[0].imshow(cropped_region, origin='lower', cmap="inferno")
        ax[0].set_title("Data")
        fig.colorbar(im, ax=ax[0], orientation="vertical")

        # Fitted Gaussian model
        im = ax[1].imshow(fit_result(x, y), origin='lower', cmap="inferno")
        ax[1].set_title("Fit")
        fig.colorbar(im, ax=ax[1], orientation="vertical")

        # Residual between data and fit
        im = ax[2].imshow(cropped_region - fit_result(x, y), origin='lower', cmap="seismic")
        ax[2].set_title("Residual")
        fig.colorbar(im, ax=ax[2], orientation="vertical")

        plt.show()

    return final_position


