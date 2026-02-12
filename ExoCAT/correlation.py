import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import interpolate, stats, signal
import h5py

from ExoCAT.degrade_spectrum import degrade_spectrum 
from ExoCAT.utils import create_circular_mask, mean_full_ring
from ExoCAT.degrade_spectrum import interp_cumsum

##########################################################################################
# Cross-correlation map calculation

def compute_cc_maps_coef(cube, spectre_th, sigma, with_err=False, err=None):
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


##########################################################################################
# Empirical method to measure the S/N (detailed in Mâlin et al. 2023)

def SNR_profil_cc(cc_maps, position_planet, plot_dist=False, plot_noise=False, plot_profil=False, plot_final=False, save_fig=False, path=None):
    """
    Measure the signal-to-noise ratio (SNR) of a correlation map around a planet position.
    SNR is defined experimentally based on the size of the correlation pattern (from MIRI/MRS simulations).

    Returns:
        SNR_final: SNR using integrated signal above noise
        SNR_1spaxel: SNR at the planet's peak spaxel
        cc: maximum correlation value around the planet
    """

    # Corrected planet position
    cc_patch = cc_maps[position_planet[1]-1:position_planet[1]+2, position_planet[0]-1:position_planet[0]+2]
    cc = np.nanmax(cc_patch)
    pos_y, pos_x = np.where(cc_maps == cc)

    # Mask planet for noise estimation
    h, w = cc_maps.shape[:2]
    mask_planets = create_circular_mask(h, w, center=(pos_x, pos_y), radius=6)
    data_noise = np.ma.array(cc_maps, mask=mask_planets)

    # Noise statistics
    noise_values = data_noise.compressed()
    std, mean = np.nanstd(noise_values), np.nanmean(noise_values)
    cc_val_noise = np.linspace(np.nanmin(noise_values), np.nanmax(noise_values))
    # SNR for a single spaxel
    SNR_1spaxel = cc / std

    # Plot noise distribution
    if plot_dist:
        plt.figure(figsize=(6, 4))
        plt.hist(data_noise.flatten(), bins=60, color="gray", alpha=0.5, density=True)
        plt.plot(cc_val_noise, stats.norm.pdf(cc_val_noise, mean, std), "r-", 
                 label=f"Gaussian fit: mean={mean:.3f}, σ={std:.3f}")
        plt.plot([cc_maps[pos_y, pos_x]]*10, np.linspace(0, 10, 10), "b--", 
                 label=f"Planet correlation = {cc_maps[pos_y, pos_x][0]:.3f}")
        plt.xlabel("Correlation values", fontsize=14)
        plt.ylabel("Histogram", fontsize=14)
        plt.legend(fontsize=12)
        plt.savefig("histogramme_noise.pdf")
        plt.show()

    # Optional noise plotting
    if plot_noise:
        print('Planet correlation:', cc)
        print('Noise std:', std)
        print('3*Noise:', 3*std)
        plt.figure(figsize=(6, 4))
        plt.title("Noise with masked planets")
        plt.imshow(data_noise, origin="lower", cmap="Blues")
        plt.colorbar()
        plt.show()

    # Profile of correlation pattern
    rmax = 5
    mean_cc, separation_cc = mean_full_ring(cc_maps, [pos_x, pos_y], width_ring=1, r_min=0, r_max=rmax+1)
    f = interpolate.interp1d(separation_cc, mean_cc)
    range_sep = np.arange(0, rmax+0.01, 0.01)

    try:
        idx_noise = np.where(f(range_sep) < 3*std)[0][0]
        r_cc = range_sep[idx_noise]
    except:
        r_cc = 5 if np.nanmin(f(range_sep)) > 3*std else 0

    if np.nanmin(f(range_sep)) > cc/2:
        r_cc = 0

    # Plot correlation profile
    if plot_profil:
        plt.plot(separation_cc, mean_cc, "+-", color="dodgerblue", label="Correlation profile")
        plt.plot(separation_cc, np.ones(len(separation_cc))*std, "g--", label="Noise level")
        plt.plot(separation_cc, np.ones(len(separation_cc))*3*std, "--", color='darkgreen', label="3 x Noise")
        plt.plot([r_cc]*len(separation_cc), np.linspace(np.nanmin(mean_cc), np.nanmax(mean_cc), len(separation_cc)), "k--", label="r_cc")
        plt.xlabel("Separation from planet (pixels)", fontsize=14)
        plt.ylabel("Mean correlation", fontsize=14)
        plt.legend()
        plt.show()

    # Recover pixels above noise
    mask_planets = create_circular_mask(h, w, center=(pos_x, pos_y), radius=int(round(r_cc)))
    data_planets = np.ma.array(cc_maps, mask=~mask_planets)
    mask_pl = np.ma.masked_where(data_planets < 0.5*cc, data_planets)

    x, y = np.where(mask_pl.mask == False)
    val_pl = data_planets[x, y]
    signal_pl = np.sum(val_pl)
    nb_px_pl = len(val_pl)

    if r_cc == -1:
        SNR_final = np.nan
    elif nb_px_pl == 0:
        SNR_final = SNR_1spaxel
    else:
        SNR_final = signal_pl / (np.sqrt(nb_px_pl) * std)

    # Final plotting
    if plot_final:
        print("FWHM cc:", cc/2)
        print("r_cc:", r_cc)
        print("3*std:", 3*std)
        print("Noise std:", std)
        print("Nb pixels planet:", nb_px_pl)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        # Profile plot
        ax[0].plot(separation_cc, mean_cc, "+-", color="dodgerblue", label='Correlation profile')
        ax[0].plot(separation_cc, np.ones(len(separation_cc))*std, "g--", label="Noise level")
        ax[0].plot(separation_cc, np.ones(len(separation_cc))*3*std, "--", color='darkgreen', label="3 x Noise")
        ax[0].plot([r_cc]*len(separation_cc), np.linspace(np.nanmin(mean_cc), np.nanmax(mean_cc), len(separation_cc)), "k--", label="r_cc")
        ax[0].set_xlabel("Separation (pixels)", fontsize=14)
        ax[0].set_ylabel("Mean correlation", fontsize=14)
        ax[0].legend()

        # Mask and noise plot
        ax[1].set_title(f"S/N planet = {SNR_final:.3f}", fontsize=14)
        im = ax[1].imshow(data_noise, origin="lower", cmap="Blues")
        fig.colorbar(im, ax=ax[1], orientation="vertical")
        im2 = ax[1].imshow(mask_pl, origin="lower", cmap="Reds")
        cbar2 = fig.colorbar(im2, ax=ax[1], orientation="vertical")
        cbar2.set_label("Correlation coefficient", rotation=270, size=12, labelpad=75)
        if save_fig and path:
            plt.savefig(path + "SNR_method.pdf")
        plt.show()

    return SNR_final, SNR_1spaxel, cc


##########################################################################################
# Preprocessing steps for stellar light subtraction in a data cube before cross-correlation

def res_cube(cube, wavelength, pc=0.2):
    """
    Estimate the stellar spectrum in a data cube and remove it to obtain the residual cube.
    Uses the method described in Hoeijmakers et al. 2018.
    
    Parameters:
    - cube: 3D numpy array (l, x, y)
        l = number of wavelength slices
        x, y = spatial dimensions
    - wavelength: 1D array of wavelength values
    - pc: float (default=0.2)
        Fraction of brightest pixels used for estimating the star spectrum.

    Returns:
    - res_cube: 3D numpy array (l, x, y) after stellar spectrum removal
    """
    

    l, x, y = cube.shape
    nb_px = x * y
    px = int(np.around(nb_px * pc))  # Number of brightest pixels to use

    # Compute total flux per pixel across wavelengths
    lum_tot = np.nansum(cube, axis=0).reshape(-1)  # flatten x*y pixels

    # Get indices of the top 'px' brightest pixels
    idx_max_list = np.argpartition(lum_tot, -px)[-px:]

    # Extract spectra from the brightest pixels
    cube_flat = cube.reshape(l, -1)  # shape: (l, x*y)
    spect_star = cube_flat[:, idx_max_list].T  # shape: (px, l)

    # Compute master spectrum as mean over selected pixels
    master_spec = np.nanmean(spect_star, axis=0)  # shape: (l,)

    # Initialize residual cube
    res_cube_arr = np.empty_like(cube)

    # Compute residual cube
    for i in range(x):
        for j in range(y):
            spaxel = cube[:, i, j]
            ratio = spaxel / master_spec
            smoothed = gaussian_filter(ratio, sigma=10)
            scaled_spec = master_spec * smoothed
            res_cube_arr[:, i, j] = spaxel - scaled_spec

    return res_cube_arr



##########################################################################################
# ExoREM spectrum loader for the library of molecular spectra

def load_spectra_molec(path, num_col, Rp, dstar, degraded=False, res=None, plot=False, savefig=False):
    """
    Extracts and processes the spectral contribution of a specific molecule from an ExoREM spectrum.

    Parameters:
        path (str): Path to the ExoREM spectrum file.
        num_col (int): Column index corresponding to the molecular contribution.
        Rp (float): Planet radius in Jupiter radii.
        dstar (float): Distance to the star in parsecs.
        degraded (bool, optional): If True, degrades the spectrum to a specified resolution.
        res (float, optional): Spectral resolution (required if degraded=True).
        plot (bool, optional): If True, plots the extracted (and degraded) spectrum.
        savefig (bool, optional): If True, saves the plot.

    Returns:
        tuple: wavelength (µm), flux (µJy)
    """

    # --- Load data ---
    data = np.loadtxt(path)
    spectral_flux = data[:, num_col]  # Flux in W.m⁻².cm⁻¹
    wavenumber = data[:, 0]           # Wavenumber in cm⁻¹

    # --- Apply dilution ---
    Rp_km = Rp * 69911.                # Jupiter radii -> km
    d_km = dstar * 3.26 * 9.46e12     # parsec -> km
    flux_dil = spectral_flux * (Rp_km**2 / d_km**2)

    # --- Convert wavenumber to wavelength and flux to µJy ---
    wavelength = 1e4 / wavenumber       # cm⁻¹ -> µm
    flux_conv = flux_dil * (1 / 3e10) * 1e26 * 1e6  # W.m⁻².cm⁻¹ -> µJy
    wavelength = wavelength[::-1]
    flux_conv = flux_conv[::-1]

    # --- Degrade spectrum if requested ---
    if degraded:
        if res is None:
            raise ValueError("Resolution 'res' must be provided when degraded=True")
        convlam, convflux = degrade_spectrum(wavelength, flux_conv, res)

    # --- Plot ---
    if plot:
        plt.figure(figsize=(14, 4))
        plt.plot(wavelength, flux_conv, color="cornflowerblue", label="Exo-REM model")
        if degraded:
            plt.plot(convlam, convflux, color="crimson", label=f"Degraded at R={res}")
        plt.xlabel("Wavelength ($\mu$m)", fontsize=14)
        plt.ylabel("Flux ($\mu$Jy)", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize="large")
        plt.xlim(4.7, 12)
        if savefig:
            plt.savefig("spectre_planet.png", bbox_inches="tight", pad_inches=0.1)
        plt.show()

    return (convlam, convflux) if degraded else (wavelength, flux_conv)


##########################################################################################
# Functions to properly copute the cCF between two spectra, 
# + compute, and correct for autocorrelation, 
# + measure the S/N of the peak at a give wavelength

def compute_ccf(wave_data,flux_data, wave_model,flux_model, sigma, drv, lim=1000, plot=False):
    """
    Calculate the Cross-Correlation Function (CCF) between a spectrum and a model.

    Parameters:
    - data: 1D array, observed spectrum.
    - wave_data: 1D array, wavelength grid for the observed spectrum (logarithmic scale).
    - model: 1D array, theoretical spectrum model.
    - wave_model: 1D array, wavelength grid for the model (logarithmic scale).
    - sigma: float, parameter for Gaussian high-pass filtering.
    - drv: float, radial velocity step in km/s.
    - lim: float, range to limit the plotting of the CCF (default = 1000 km/s).
    - plot: bool, optional, if True, generates plots for visualization.

    Returns:
    - rvs: 1D array, radial velocities in km/s.
    - ccf: 1D array, normalized cross-correlation function.
    """

    c = 299792.458  # Speed of light in km/s

    # Validate input dimensions
    if len(flux_data) != len(wave_data):
        raise ValueError("'data' and 'wave_data' must have the same length.")
    if len(flux_model) != len(wave_model):
        raise ValueError("'model' and 'wave_model' must have the same length.")

    # Interpolate the model to match the wavelength range of the data
    model_interp_func = interpolate.interp1d(wave_model, flux_model, fill_value="extrapolate")
    model_resampled = model_interp_func(wave_data)

    # Apply Gaussian high-pass filtering to the data and model
    model_filtered = model_resampled - gaussian_filter(model_resampled, sigma)
    data_filtered = flux_data - gaussian_filter(flux_data, sigma)

    # Generate a new logarithmic wavelength grid for interpolation
    log_wavelength_new = np.exp(np.arange(np.log(np.min(wave_data)), np.log(np.max(wave_data)), drv / c))
    data_interp = interp_cumsum(wave_data, data_filtered, log_wavelength_new)
    model_interp = interp_cumsum(wave_data, model_filtered, log_wavelength_new)

    # Calculate the normalized cross-correlation function
    ccf = signal.correlate(data_interp, model_interp, mode="full", method="direct")
    ccf /= np.sqrt(np.sum(model_interp**2) * np.sum(data_interp**2))

    # Compute the radial velocity grid
    lags = signal.correlation_lags(len(data_interp), len(model_interp))
    rvs = lags * drv

    # Plot results if requested
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), tight_layout=True)
        fig.suptitle("Cross-Correlation Function Analysis", fontsize=16)

        # Plot the filtered model spectrum
        axes[0].plot(log_wavelength_new, model_interp, color="black", label="Filtered Model Spectrum")
        axes[0].set_xlabel("Wavelength (log scale)", fontsize=12)
        axes[0].set_ylabel("Flux", fontsize=12)
        axes[0].legend()
        axes[0].grid()

        # Plot the raw data spectrum and its Gaussian-filtered version
        axes[1].plot(wave_data, flux_data, color="navy", label="Raw Data Spectrum")
        axes[1].plot(wave_data, gaussian_filter(flux_data, sigma), color="green", linestyle="dashed", label="Gaussian Filtered Data")
        axes[1].set_xlabel("Wavelength (log scale)", fontsize=12)
        axes[1].set_ylabel("Flux", fontsize=12)
        axes[1].legend()
        axes[1].grid()

        # Plot the interpolated and filtered data spectrum
        axes[2].plot(log_wavelength_new, data_interp, color="navy", label="Filtered & Interpolated Data")
        axes[2].set_xlabel("Wavelength (log scale)", fontsize=12)
        axes[2].set_ylabel("Flux", fontsize=12)
        axes[2].legend()
        axes[2].grid()

        # Plot the Cross-Correlation Function
        axes[3].plot(rvs, ccf, color="navy", label="CCF")
        axes[3].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[3].set_xlim(-lim, lim)
        axes[3].set_xlabel("Radial Velocity (km/s)", fontsize=12)
        axes[3].set_ylabel("CCF", fontsize=12)
        axes[3].legend()
        axes[3].grid()

        plt.show()

    return rvs, ccf


def compute_ccf_autocorr(wave_data, model, wave_model, sigma, drv):
    """
    Compute the Cross-Correlation Function (CCF) between a spectrum and itself using Gaussian filtering and interpolation.

    Parameters:
    - lg_data: 1D array, logarithmic wavelength grid of the observed data.
    - model: 1D array, spectrum model corresponding to lg_model.
    - lg_model: 1D array, logarithmic wavelength grid of the model.
    - sigma: float, parameter for the Gaussian high-pass filter.
    - drv: float, velocity step in km/s (used to calculate radial velocity).

    Returns:
    - rvs: 1D array, radial velocities in km/s.
    - ccf: 1D array, normalized cross-correlation function.
    """
    c = 299792.458  # Speed of light in km/s

    # Interpolate the model onto the wavelength grid of lg_data
    spectrum_model = interpolate.interp1d(wave_model, model, fill_value="extrapolate")(wave_data)

    # Apply Gaussian high-pass filtering on the model spectrum
    spectrum_model_filter = spectrum_model - gaussian_filter(spectrum_model, sigma)

    # Create a new wavelength grid for interpolation
    wavelengths = np.exp(np.arange(np.log(np.min(wave_data)), np.log(np.max(wave_data)), drv / c))

    # Interpolate the filtered model spectrum onto the new wavelength grid
    spectrum_model_interp = interp_cumsum(wave_data, spectrum_model_filter, wavelengths)

    # Compute the Cross-Correlation Function (CCF)
    ccf = signal.correlate(spectrum_model_interp, spectrum_model_interp, mode="full", method="direct")
    ccf /= np.sqrt(np.sum(spectrum_model_interp**2) * np.sum(spectrum_model_interp**2))

    # Calculate the lags and radial velocities
    lags = signal.correlation_lags(len(spectrum_model_interp), len(spectrum_model_interp))
    rvs = lags * drv

    return rvs, ccf

def corr_ccf_from_autocorr(rv_autocorr, ccf_autocorr, rvs, ccf, 
                           plot=False, molec_name=None, rv_min=500, rv_max=3000):
    """
    Correct the Cross-Correlation Function (CCF) using autocorrelation data.

    Parameters:
    - rv_autocorr: 1D array, radial velocity for autocorrelation data.
    - ccf_autocorr: 1D array, autocorrelation values.
    - rvs: 1D array, radial velocities for the CCF.
    - ccf: 1D array, CCF values.
    - plot: bool, optional, if True, generate plots for visualization.
    - molec_name: str, optional, name of the molecule (used in plot titles).

    Returns:
    - rvs: 1D array, radial velocities (unchanged).
    - ccf_corr: 1D array, corrected CCF values.
    """

    if len(rvs) != len(ccf):
        raise ValueError("Length of 'rvs' and 'ccf' must be the same.")

    # Find the value of CCF at rvs == 0
    zero_idx = np.where(rvs == 0)[0]
    if len(zero_idx) == 0:
        raise ValueError("'rvs' must contain a value of 0 for proper scaling.")
    cc_0 = ccf[zero_idx[0]]

    # Rescale the autocorrelation function
    scale_factor = ccf_autocorr[np.where(rv_autocorr > 0)][0]
    ccf_autocorr_rescaled = (ccf_autocorr / scale_factor) * cc_0

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(rv_autocorr, ccf_autocorr, "r-", label="Autocorrelation Molecule")
        plt.plot(rv_autocorr, ccf_autocorr_rescaled, "r--", label="Rescaled Autocorrelation")
        plt.plot(rvs, ccf, "k-", label="Original CCF")
        plt.ylabel("Cross-Correlation", fontsize=14)
        plt.xlabel("Radial Velocity (km/s)", fontsize=14)
        plt.title(f"Autocorrelation and CCF Comparison ({molec_name})" if molec_name else "Autocorrelation and CCF Comparison", fontsize=16)
        plt.xlim(-2000, 2000)
        plt.legend()
        plt.grid()
        plt.show()

    # Correct the CCF
    correction_mask = ((rvs > rv_min) & (rvs < rv_max)) | ((rvs > -rv_max) & (rvs < -rv_min))
    ccf_corr = np.where(correction_mask, ccf - ccf_autocorr_rescaled, ccf)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.title(f"CCF Before and After Correction ({molec_name})" if molec_name else "CCF Before and After Correction", fontsize=16)
        plt.plot(rvs, ccf_corr, "k-", label="Corrected CCF")
        plt.plot(rvs, ccf, "b-", label="Original CCF")
        plt.ylabel("Cross-Correlation", fontsize=14)
        plt.xlabel("Radial Velocity (km/s)", fontsize=14)
        plt.xlim(-3000, 3000)
        plt.legend()
        plt.grid()
        plt.show()

    return rvs, ccf_corr


def compute_SN_ccf(rvs, ccf, rv_wish, rv_min=500, rv_max=3000):
    """
    Calculate the Signal-to-Noise (SN) ratio of the Cross-Correlation Function (CCF) at a specific radial velocity (rv_wish).
    
    Parameters:
    - rvs: 1D array, radial velocities in km/s.
    - ccf: 1D array, Cross-Correlation Function (CCF) values.
    - rv_wish: float, the desired radial velocity (in km/s) at which to calculate the SNR.
    - rv_min, rv_max : bounds of the wings where to measure the std : radial velocities between 500 < rvs < 3000 or -3000 < rvs < -500
    
    Returns:
    - SNR_ccf: float, Signal-to-Noise Ratio (SNR) at the desired radial velocity (rv_wish).
    """
    
    # Extract the CCF value at the specific radial velocity (rv_wish)
    signal_1spaxel = ccf[rvs == rv_wish]
    
    # Ensure that rv_wish corresponds to an actual value in the rvs array
    if signal_1spaxel.size == 0:
        raise ValueError(f"rv_wish ({rv_wish}) not found in rvs array.")
    
    # Define noise region: radial velocities between 500 < rvs < 3000 or -3000 < rvs < -500
    noise_mask = (rvs > rv_min) & (rvs < rv_max) | (rvs < -rv_min) & (rvs > -rv_max)
    
    # Extract noise values
    noise = ccf[noise_mask]
    
    # Calculate the SN as the signal over the standard deviation of the noise
    SN_ccf = signal_1spaxel / np.nanstd(noise)
    
    return SN_ccf
