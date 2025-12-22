import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import h5py

from ExoCAT.degrade_spectrum import degrade_spectrum 

##########################################################################################
# Cross-correlation map calculation

def calcul_cc_maps_coef(cube, spectre_th, sigma, with_err=False, err=None):
    """
    Compute 2D map of correlation coefficients between a 3D data cube and a theoretical spectrum.

    Parameters:
    - cube: 3D array (wavelength x declination x right ascension), the data cube.
    - spectre_th: 1D array, theoretical spectrum interpolated on the cube's wavelength range.
    - sigma: float, parameter for Gaussian filtering.
    - with_err: bool, optional, if True, include uncertainties in calculations.
    - err: 3D array, optional, uncertainties for each value in the cube.

    Returns:
    - coeff: 2D array (declination x right ascension), map of correlation coefficients.
    """
    if cube.shape[0] != len(spectre_th):
        raise ValueError("The wavelength dimension of 'cube' and 'spectre_th' must match.")
    if with_err and (err is None or err.shape != cube.shape):
        raise ValueError("'err' must be provided and have the same shape as 'cube' when 'with_err' is True.")

    wavelength, dec, ra = cube.shape
    coeff = np.empty((dec, ra), dtype=np.float64)

    for i in range(dec):
        for j in range(ra):
            A = cube[:, i, j].copy()
            B = spectre_th.copy()

            # Gaussian high-pass filtering
            A -= gaussian_filter(A, sigma)
            B -= gaussian_filter(B, sigma)

            # Correlation coefficient
            if with_err:
                C = err[:, i, j]
                weight = 1 / (C ** 2)
                cc = np.sum(A * B * weight) / np.sqrt(np.sum(A**2 * weight) * np.sum(B**2 * weight))
            else:
                cc = np.sum(A * B) / np.sqrt(np.sum(A**2) * np.sum(B**2))

            coeff[i, j] = cc

    return coeff

##########################################################################################
# ExoREM spectrum loader

def load_spectrum(path, path_wave="/Users/mmalin/Models_spectra/ExoREM_2025/wavenumber.h5", Rp=1, dstar=1, degraded=False, res=None, plot=False, savefig=False):
	
    """
    Processes an ExoREM spectrum (version 2025) for MIRI-MRS.

    Parameters:
        path (str): Path to the HDF5 file containing the ExoREM spectrum.
        Rp (float): Planetary radius in Jupiter radii.
        dstar (float): Distance to the star in parsecs.
        degraded (bool, optional): If True, degrades the spectrum to a specified resolution.
        res (float, optional): Spectral resolution to degrade the spectrum (required if degraded=True).
        plot (bool, optional): If True, plots the original and degraded spectra.
        savefig (bool, optional): If True, saves the plot as an image file.

    Returns:
        tuple: 
            - Wavelength array (µm).
            - Flux array (µJy).
            - If degraded=True, returns the degraded spectrum.
    """

    # Planet and star parameters
    R = Rp * 69911 * 1e5          # Planet radius in cm (Jupiter radius -> cm)
    d = dstar * 3.26 * 9.46e18    # Distance in cm (pc -> cm)

    # Load ExoREM spectral flux
    try:
        with h5py.File(path, "r") as hf:
            fk = hf["flux"][:]  # flux [W/m^2/cm^-1]
    except Exception as e:
        raise RuntimeError(f"Error reading HDF5 file {path}: {e}")

    # Dilution by planet size and distance
    flux_dil = fk * (R**2 / d**2)  # scaled flux [W/m^2/cm^-1]

    # Load wavenumber
    filename_wavenumber = path_wave
    with h5py.File(filename_wavenumber, "r") as kf:
        wn = kf["wavenumber"][:]  # wavenumber [cm^-1]

    # Convert wavenumber to wavelength and flux to µJy
    wavelength = 1e4 / wn
    flux_conv = flux_dil * (1 / 3e10) * 1e26 * 1e6  # W/m^2/cm^-1 -> µJy

    # Reverse arrays to have increasing wavelength
    wavelength = np.flip(wavelength)
    flux_conv = np.flip(flux_conv)

    # Degrade spectrum if requested
    if degraded:
        if res is None:
            raise ValueError("Resolution 'res' must be provided when degraded=True")
        convlam, convflux = degrade_spectrum(wavelength, flux_conv, res)

    # Plot if requested
    if plot:
        plt.figure(figsize=(14, 4))
        plt.plot(wavelength, flux_conv, color="cornflowerblue", label="Exo-REM model")
        if degraded:
            plt.plot(convlam, convflux, color="crimson", label=f"Degraded at R={res}")
        plt.legend(fontsize="large")
        plt.ylabel("Flux ($\mu$Jy)", fontsize=14)
        plt.xlabel("Wavelength ($\mu$m)", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(0, 30)
        if savefig:
            plt.savefig("spectre_planet.png")
        plt.show()

    return (convlam, convflux) if degraded else (wavelength, flux_conv)
