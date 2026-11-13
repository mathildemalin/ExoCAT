import os
import numpy as np
import asdf, stpsf
from astropy.io import fits

from ExoCAT.correlation import compute_cc_maps_coef, SNR_profil_cc
from ExoCAT.mrs_tools import load_cube_dir, load_psfs, extract_spectrum
from ExoCAT.utils import match_cube_to_shape, pad_cube_to_shape,measure_position_fit

from scipy.interpolate import interp1d
from scipy.ndimage import shift

# optimisation code for the simulation
from scipy.optimize import brentq 

# for the plotting options : 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#####################################################################################################################################################################


def simulation_MRS_single_band(wave_model, flux_model_planet,
                               wave_model_star=None,flux_model_star=None,
                               band='1A',
                               offset_planet=1.0,theta_deg=90,
                               scale_star=True,
                               mrs_wavelengths=None,mrs_star_cube=None,mrs_planet_cube=None,
                               target_shape=(50, 50),
                               wave_model_cc=None, model_cc=None,
                               pos_star = None, # useful if the PSF core is outside the FoV
                               # save + plot options :
                               system_name=None, output_dir=None,
                               plot_models=False,
                               plots_maps=False,plot_cubes=False,):
    """
    Simulate MIRI/MRS observations of a star+planet system for a single band.
    The star and planet cubes can either be provided explicitly or, if omitted,
    simulated PSFs are used by default.
    If a stellar model is provided, the stellar PSF cube is rescaled at each
    wavelength so that its extracted flux matches the stellar model spectrum.
    Parameters
    ----------
    wave_model : array
        Wavelength grid of the planet model [µm].
    flux_model_planet : array
        Flux of the planet model [µJy].
    band : str
        MRS band to simulate ('1A', '1B', ..., '3C').
    offset_planet : float
        Angular separation between star and planet [arcsec].
    theta_deg : float
        Position angle of the planet [deg, East of North].
    mrs_wavelengths : array
        MRS wavelength grid for this band [µm].
    mrs_star_cube : array, optional
        Stellar PSF/data cube with shape (n_wave, ny, nx).
        If None, use the simulated PSF.
    mrs_planet_cube : array, optional
        Planet PSF cube with shape (n_wave, ny, nx).
        If None, use the simulated PSF.
    wave_model_star : array, optional
        Wavelength grid of the stellar model [µm].
    flux_model_star : array, optional
        Stellar model flux [same flux units as extracted PSF].
    target_shape : tuple, optional
        Spatial shape of the simulated PSFs.
    system_name : str, optional
        Name of the system, used for plots.
    output_dir : str, optional
        Directory where plots are saved.
    plot_models : bool, optional
        Plot the input planet and stellar spectra.
    plots_maps : bool, optional
        Plot the correlation map.
    plot_cubes : bool, optional
        Plot the median star, planet, and simulated cubes side by side
        in a single row.
    scale_star : bool, optional
        If True, rescale the stellar PSF cube to the stellar model.
        Requires wave_model_star and flux_model_star.
    Returns
    -------
    SNR : float
        Signal-to-noise ratio at the expected planet position.
    cc_maps_res_masked : array
        Cross-correlation map with the FoV masked.
    simulated_data : array
        Simulated star + planet cube.
    mrs_wavelengths : array
        MRS wavelength grid.
    psfs_star : array
        Final stellar cube after optional rescaling.
    psfs_planet_scaled : array
        Final scaled planet cube.
    """
    # =========================================================================
    # Configuration
    # =========================================================================
    fs=14 # for plots
    mrs_pixel_scales = {
        "1A": 0.196, "1B": 0.196, "1C": 0.196,
        "2A": 0.196, "2B": 0.196, "2C": 0.196,
        "3A": 0.245, "3B": 0.245, "3C": 0.245,}
    if band not in mrs_pixel_scales:
        raise ValueError(f"Unknown MRS band '{band}'. " 
                         f"Expected one of {list(mrs_pixel_scales)}.")
    if mrs_wavelengths is None:
        raise ValueError("mrs_wavelengths must be provided for the selected band.")
    mrs_wavelengths = np.asarray(mrs_wavelengths,dtype=float,)
    
    # =========================================================================
    # Check stellar model
    # =========================================================================
    if scale_star and (
        wave_model_star is None or flux_model_star is None):
        raise ValueError(
            "scale_star=True requires both wave_model_star and flux_model_star.")
    
    # =========================================================================
    # Load or generate simulated PSFs
    # =========================================================================
    fits_filename = (f"./Inputs/STPSF/psfs_mrs_{band}.fits")
    if os.path.exists(fits_filename):
        mrs_simulated_psfs = load_psfs(fits_filename)[0]
        mrs_simulated_psfs = np.asarray(mrs_simulated_psfs,dtype=float)
    else:
        mrs_simulated_psfs = []
        for lam in mrs_wavelengths:
            psf = stpsf.calc_psf(fov_pixels=target_shape,monochromatic=lam * 1e-6)
            mrs_simulated_psfs.append(psf[3].data)
        mrs_simulated_psfs = np.asarray(mrs_simulated_psfs,dtype=float)
        fits.PrimaryHDU(mrs_simulated_psfs).writeto(fits_filename,overwrite=True)
    # =========================================================================
    # Stellar cube
    # Use provided cube if available, otherwise simulated PSF.
    # =========================================================================
    if mrs_star_cube is None:
        psfs_star = match_cube_to_shape(mrs_simulated_psfs,target_shape=target_shape)
    else:
        # If a stellar cube is provided, its spatial dimensions + define the target shape.
        target_shape = mrs_star_cube.shape[1:]
        psfs_star = match_cube_to_shape(mrs_star_cube,target_shape=target_shape)
    # =========================================================================
    # Determine positions
    # =========================================================================
    if pos_star is None :
        # Initial guess: center of the spatial dimensions of the cube
        image_median = np.nanmedian(psfs_star, axis=0)
        y_init, x_init = np.unravel_index(np.nanargmax(image_median), image_median.shape)
        pos_init = (x_init, y_init)
        # Fit the stellar position
        pos_star = measure_position_fit(psfs_star, pos_init, window=(10, 10), plot=False)
        # Theoretical planet position
        theta_rad = np.deg2rad(theta_deg)
        pos_planet = (pos_star[0] + (offset_planet * np.cos(theta_rad)) / mrs_pixel_scales[band],
                      pos_star[1] + (offset_planet * np.sin(theta_rad)) / mrs_pixel_scales[band],)
    
    # =========================================================================
    # Load aperture correction -> useful to correct for aperture correction
    # =========================================================================
    path_ref = ("./Inputs/jwst_miri_apcorr_0008.asdf")
    with asdf.open(path_ref) as af:
        wavelength_apcorr = np.asarray(af["apcorr_table"]["wavelength"],dtype=float)
        apcorr = af["apcorr_table"]["apcorr"]
        apcorr3 = interp1d(wavelength_apcorr,np.asarray(apcorr[4], dtype=float),
                           bounds_error=False,fill_value="extrapolate",)(mrs_wavelengths)
    # =========================================================================
    # Rescale stellar PSF to stellar model
    # =========================================================================
    if scale_star:
        # Interpolate stellar model onto MRS wavelengths
        star_flux_interp = interp1d(wave_model_star,flux_model_star,kind="linear",
                                    bounds_error=False,fill_value="extrapolate",)(mrs_wavelengths)
        star_flux_interp = np.asarray(star_flux_interp,dtype=float,)
        # Extract stellar spectrum from the PSF cube
        psf_flux_star, _ = extract_spectrum(psfs_star,mrs_wavelengths,pos_star,mrs_pixel_scales[band],fwhm_size=3,)
        # Apply aperture correction
        psf_flux_star = (np.asarray(psf_flux_star, dtype=float)* apcorr3)
        # Calculate scale factors safely
        scale_factors = np.ones_like( star_flux_interp,dtype=float)
        valid = (np.isfinite(star_flux_interp) & np.isfinite(psf_flux_star) & (psf_flux_star > 0))
        scale_factors[valid] = (star_flux_interp[valid]/ psf_flux_star[valid])
        # ---------------------------------------------------------------------
        # Apply wavelength-dependent scaling
        psfs_star = np.asarray([ psfs_star[l] * scale_factors[l]for l in range(len(psfs_star))])
        # Replace invalid/negative values
        psfs_star = np.nan_to_num(psfs_star,nan=1e-5, posinf=1e-5, neginf=1e-5)
        psfs_star[psfs_star <= 0] = 1e-5
    # =========================================================================
    # Planet cube : Use provided cube if available, otherwise simulated PSF.
    if mrs_planet_cube is None:
        mrs_data_planet_padded = match_cube_to_shape(mrs_simulated_psfs,target_shape=target_shape,)
    else:
        mrs_data_planet_padded = match_cube_to_shape(mrs_planet_cube,target_shape=target_shape,)
    # =========================================================================
    image_median_planet = np.nanmedian(mrs_data_planet_padded, axis=0)
    y_init_planet, x_init_planet = np.unravel_index(np.nanargmax(image_median_planet),image_median_planet.shape)
    offset = (pos_planet[1] - y_init_planet,pos_planet[0] - x_init_planet,)
    psfs_planet_shift = np.asarray([shift(mrs_data_planet_padded[l], offset, order=1) for l in range(len(mrs_data_planet_padded))])
    # =========================================================================
    # Interpolate planet model
    planet_flux_interp = interp1d(wave_model, flux_model_planet,kind="linear",
                                  bounds_error=False,fill_value="extrapolate",)(mrs_wavelengths)
    planet_flux_interp = np.asarray(planet_flux_interp,dtype=float,)
    # =========================================================================
    # Rescale planet PSF to planet model
    # =========================================================================
    # Interpolate planet model onto MRS wavelengths
    planet_flux_interp = interp1d(wave_model, flux_model_planet, kind="linear",
                                  bounds_error=False, fill_value="extrapolate",)(mrs_wavelengths)
    planet_flux_interp = np.asarray(planet_flux_interp, dtype=float,)
    # Extract planet spectrum from the PSF cube
    psf_flux_planet, _ = extract_spectrum(psfs_planet_shift, mrs_wavelengths, pos_planet, mrs_pixel_scales[band], fwhm_size=3,)
    psf_flux_planet = (np.asarray(psf_flux_planet, dtype=float) * apcorr3) # Apply aperture correction
    # Calculate scale factors safely
    planet_scale_factors = np.ones_like(planet_flux_interp, dtype=float)
    valid = (np.isfinite(planet_flux_interp) & np.isfinite(psf_flux_planet) & (psf_flux_planet > 0))
    planet_scale_factors[valid] = (planet_flux_interp[valid] / psf_flux_planet[valid])
    # ---------------------------------------------------------------------
    # Apply wavelength-dependent scaling
    psfs_planet_scaled = np.asarray([psfs_planet_shift[l] * planet_scale_factors[l] for l in range(len(psfs_planet_shift))])
    # Replace invalid/negative values
    psfs_planet_scaled = np.nan_to_num(psfs_planet_scaled, nan=1e-5, posinf=1e-5, neginf=1e-5)
    psfs_planet_scaled[psfs_planet_scaled <= 0] = 1e-5
    # =========================================================================
    # Create final star + planet simulated cube
    mask_radius = 8
    ny, nx = psfs_planet_scaled[0].shape
    Y, X = np.ogrid[:ny, :nx]
    y0 = pos_planet[1]
    x0 = pos_planet[0]
    planet_mask = ((X - x0) ** 2+ (Y - y0) ** 2<= mask_radius ** 2)
    simulated_data = []
    for l in range(len(psfs_planet_scaled)):
        planet_masked = np.zeros_like(psfs_planet_scaled[l])
        planet_masked[planet_mask] = (psfs_planet_scaled[l][planet_mask])
        simulated_data.append(planet_masked + psfs_star[l])
    simulated_data = np.asarray(simulated_data)
    
    # =========================================================================
    # Plot star, planet, and simulated cubes in a single row
    # =========================================================================
    if plot_cubes:
        cubes = [psfs_star, psfs_planet_scaled, simulated_data]
        titles = ["Star cube", "Planet cube", "Simulated data"]
        norms = [LogNorm(vmin=1, vmax=1e5), None, LogNorm(vmin=1, vmax=1e5)]  # (linear scale for the planet image)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, cube, title, norm in zip(axes, cubes, titles, norms):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(np.nanmedian(cube, axis=0), cmap="inferno", origin="lower", norm=norm)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            cbar.set_label(r"Flux [$\mu$Jy]", color="black")
            ax.set_title(f"{title} - Band {band}")
            ax.scatter(pos_star[0], pos_star[1], marker="*", color="lime", s=100, label="Star")
            ax.scatter(pos_planet[0], pos_planet[1], marker="*", color="cyan", s=100, label="Planet")
        axes[0].legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"Cubes_{band}_{system_name}.png"), bbox_inches="tight")
        plt.show()

    # =========================================================================
    # Plot input spectra (before / after PSF rescaling) + contrast
    # =========================================================================
    if plot_models:
        # Extracted planet PSF flux after rescaling, for comparison with the model
        psf_flux_planet_scaled, _ = extract_spectrum(
            psfs_planet_scaled, mrs_wavelengths, pos_planet, mrs_pixel_scales[band], fwhm_size=3,)
        psf_flux_planet_scaled = np.asarray(psf_flux_planet_scaled, dtype=float) * apcorr3

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True,
            gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.05},)

        # --- Top panel: input spectra ---------------------------------------
        ax1.plot(wave_model, flux_model_planet, label="Planet model", color="blueviolet")
        ax1.plot(mrs_wavelengths, psf_flux_planet_scaled, label="Planet PSF (after scaling)", color="dodgerblue")
        if wave_model_star is not None and flux_model_star is not None:
            ax1.plot(wave_model_star, flux_model_star, label="Star model", color="orange")

        psf_flux_star_scaled = None
        if scale_star:
            # Extracted stellar PSF flux after rescaling, for comparison with the model
            psf_flux_star_scaled, _ = extract_spectrum(
                psfs_star, mrs_wavelengths, pos_star, mrs_pixel_scales[band], fwhm_size=3,)
            psf_flux_star_scaled = np.asarray(psf_flux_star_scaled, dtype=float) * apcorr3
            ax1.plot(mrs_wavelengths, psf_flux_star_scaled, label="Star PSF (after scaling)", color="crimson")
        
        ax1.set_ylabel(r"Flux [$\mu$Jy]", fontsize=fs)
        ax1.set_yscale("log")
        ax1.set_xlim(4, 25)
        ax1.set_ylim(1e-1, 1e9)
        ax1.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax1.tick_params(which="major", length=8, width=1.4, labelsize=fs)
        ax1.tick_params(which="minor", length=4, width=1.0)
        ax1.minorticks_on()
        ax1.legend(fontsize=fs - 1, loc="upper right")
        ax1.grid(alpha=0.2, which="both", linestyle="--")

        # --- Bottom panel: contrast (planet / star, after scaling) ----------
        if psf_flux_star_scaled is not None:
            contrast = psf_flux_planet_scaled / psf_flux_star_scaled
            ax2.plot(mrs_wavelengths, contrast, color="darkblue", label="Contrast")

        ax2.set_xlabel(r"Wavelength [$\mu$m]", fontsize=fs)
        ax2.set_ylabel("Flux ratio", fontsize=fs)
        ax2.set_yscale("log")
        #ax2.set_ylim(1e-6, 1e-3)
        ax2.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax2.tick_params(which="major", length=6, width=1.2, labelsize=fs)
        ax2.tick_params(which="minor", length=3, width=1.0)
        ax2.minorticks_on()
        ax2.legend(fontsize=fs - 1, loc="upper right")
        ax2.grid(alpha=0.2, which="both", linestyle="--")

        if output_dir is not None:
            plt.savefig(
                os.path.join(output_dir, f"Input_spectra_{band}_{system_name}.png"),
                bbox_inches="tight", pad_inches=0,)
        plt.show()
        
    # =========================================================================
    # Apply Molecular mapping post processing
    # =========================================================================
    
    # Molecular mapping
    if (wave_model_cc is None or model_cc is None):
        with np.errstate(invalid="ignore"): # avoid low values and nan
            cc_maps_res = compute_cc_maps_coef(simulated_data, planet_flux_interp, sigma=10)
    else : 
        model_cc_interp = interp1d(wave_model_cc, model_cc, kind="linear",
                                   bounds_error=False, fill_value="extrapolate",)(mrs_wavelengths)
        with np.errstate(invalid="ignore"):
            cc_maps_res = compute_cc_maps_coef(simulated_data, model_cc_interp, sigma=10)

    pos = (int(np.around(pos_planet[0])),int(np.around(pos_planet[1])),)
    # Mask outside the FoV
    background_value = psfs_star[0, 0, 0]
    mask_fov = (psfs_star[0] == background_value)
    cc_maps_res_masked = np.where(~mask_fov,cc_maps_res, np.nan,)
    # There might be other ways to measure the S/N on correlation maps !
    SNR, SNR1, _ = SNR_profil_cc(cc_maps_res_masked,pos,)
    # Plot correlation map
    if plots_maps:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(f"Band {band} - S/N = {SNR:.2f}")
        im = ax.imshow(cc_maps_res_masked, origin="lower", cmap="inferno", vmin=-0.05)
        circle = plt.Circle((pos[0], pos[1]), radius=2, edgecolor="dodgerblue", facecolor="none", linewidth=2, label="Planet")
        ax.add_patch(circle)
        ax.scatter(pos_star[0], pos_star[1], marker="*", s=120, color="lime", label="Star")
        fig.colorbar(im, ax=ax)
        ax.legend()
        plt.show()
    return (SNR, cc_maps_res_masked,simulated_data,mrs_wavelengths, pos_planet,pos_star)


#####################################################################################################################################################################
#######         
#######     Functions to measure detection limits for GJ 504 b (to update)
#######

def simulation_MRS_single_band_GJ504b(wave_model, flux_model_planet,
                               band,
                               offset_planet, theta_deg,
                               path_mrs_data = '/Users/mmalin/MIRI/MRS/GJ504/DATA/FINAL/stage_3_comb_ch12nocosm_ch34wcosm/cubes_obs3/',
                               system_name = None,                               
                               output_dir = None, # useful to save the plots
                               plot_models = False, plots_maps=False):
   
    """
    Simulate MIRI/MRS observations of a star+planet system for a single band.

    Parameters
    ----------
    wave_model : array
        Wavelength grid of the planet model [µm].
    flux_model_planet : array
        Flux of the planet model [µJy].
    band : str
        MRS band to simulate ('1A', '1B', ..., '3C').
    offset_planet : float
        Angular separation between star and planet [arcsec].
    theta_deg : float
        Position angle of the planet [deg, East of North].
    output_dir : str
        Directory where outputs are saved.
    system_name : str, optional
        Name of the system (for plots).
    plot_models : bool, optional
        Plot input spectra.
    plots : bool, optional
        Plot cubes and PSFs.
    plots_maps : bool, optional
        Plot correlation maps.

        Default is shown for GJ 504 b 

    """

    # Pixel scales
    mrs_pixel_scales = {"1A": 0.196, "1B": 0.196, "1C":0.196,
                        "2A": 0.196, "2B": 0.196, "2C": 0.196,
                        "3A": 0.245, "3B": 0.245, "3C": 0.245}

    target_shape = (61, 61)

    # Plot planet model
    if plot_models:
        plt.figure(figsize=(8,4))
        plt.plot(wave_model, flux_model_planet, label="Planet model", color='blueviolet')
        plt.legend()
        plt.xlim(4, 25)
        plt.ylim(1e-1, 1e7)
        plt.xlabel('Wavelength [$\mu$m]')
        plt.ylabel('Flux [$\mu$Jy]')
        plt.yscale('log')
        plt.grid(alpha=0.1, which='both', linestyle='--', color='gray') 
        if output_dir is not None:
            plt.savefig(output_dir+f'Input_spectra.png')
        plt.show()

    # Load star data
    mrs_star = load_cube_dir(path_mrs_data)
    mrs_wavelengths = mrs_star[band]['wav']
    mrs_star_cube = fits.open(f'/Users/mmalin/MIRI/MRS/GJ504/DATA/FINAL/stage_3_comb_ch12nocosm_ch34wcosm/Results_no_offsets/Cubes_nobkg/ref_nobkg_{band}.fits')[0].data
    
    # Pad stellar cube
    psfs_star_final = pad_cube_to_shape(mrs_star_cube, target_shape=target_shape)

    # Star position (predefined)
    pos_star = {'1A': np.array([31.82041297,  7.58439186]),'1B': np.array([31.35557586,  6.79486828]),'1C': np.array([31.28487482,  8.95539697]),
                '2A': np.array([31.65481364, 11.26014188]),'2B': np.array([32.11232667,  9.50777295]),'2C': np.array([31.06547142,  9.9496038 ]),
                '3A': np.array([34.67124189, 15.23826755]),'3B': np.array([34.75905618, 14.10499495]), '3C': np.array([34.12884825, 14.25775176])}

    # Load aperture correction
    path_ref = "/Users/mmalin/crds_cache/references/jwst/miri/jwst_miri_apcorr_0008.asdf"
    af = asdf.open(path_ref)
    apcorr = af['apcorr_table']['apcorr']
    apcorr3 = interp1d(af['apcorr_table']['wavelength'], apcorr[4])(mrs_wavelengths)
    apcorr1 = interp1d(af['apcorr_table']['wavelength'], apcorr[1])(mrs_wavelengths)

    # Load or compute simulated PSFs
    fits_filename = f"/Users/mmalin/MIRI/MRS/Faint_planets/STPSF/psfs_mrs_{band}.fits"
    if not os.path.exists(fits_filename):
        # Compute PSFs (simplified)
        mrs_simulated_psfs = []
        for l, lam in enumerate(mrs_wavelengths):
            lam_m = lam*1e-6
            psf = stpsf.calc_psf(fov_pixels=target_shape, monochromatic=lam_m)
            mrs_simulated_psfs.append(psf[3].data)
        fits.PrimaryHDU(np.array(mrs_simulated_psfs)).writeto(fits_filename, overwrite=True)
    else:
        mrs_simulated_psfs = load_psfs(fits_filename)[0]

    # Pad planet PSFs
    mrs_data_planet_padded = match_cube_to_shape(mrs_simulated_psfs, target_shape=target_shape)

    # Planet theoretical position
    theta_rad = np.deg2rad(theta_deg)
    pos_planet_th = (pos_star[band][0] + (offset_planet * np.cos(theta_rad)) / mrs_pixel_scales[band],
                     pos_star[band][1] + (offset_planet * np.sin(theta_rad)) / mrs_pixel_scales[band])

    # Shift planet PSF
    image_median = np.nanmedian(mrs_data_planet_padded, axis=0)
    pos_init = np.unravel_index(np.nanargmax(image_median), image_median.shape)
    offset = np.array([pos_planet_th[1], pos_planet_th[0]]) - np.array(pos_init)  # y,x for shift
    psfs_planet_shift = [shift(mrs_data_planet_padded[l], offset, order=1) for l in range(len(mrs_data_planet_padded))]

    # Scale planet PSF to model
    model_interp = interp1d(wave_model, flux_model_planet, kind='linear', bounds_error=False, fill_value='extrapolate')(mrs_wavelengths)
    psf_flux, _ = extract_spectrum(psfs_planet_shift, mrs_wavelengths, pos_planet_th, mrs_pixel_scales[band], fwhm_size=3)
    psf_flux *= apcorr3
    psfs_planet_scaled = [psfs_planet_shift[l] * (model_interp[l]/psf_flux[l]) for l in range(len(psfs_planet_shift))]

    # Final simulated data (star + planet)
    mask_radius = 6
    ny, nx = psfs_planet_scaled[0].shape
    Y, X = np.ogrid[:ny, :nx]
    y0, x0 = pos_planet_th[1], pos_planet_th[0]
    mask = (X - x0)**2 + (Y - y0)**2 <= mask_radius**2

    simulated_data = []
    for l in range(len(psfs_planet_scaled)):
        planet_masked = np.zeros_like(psfs_planet_scaled[l])
        planet_masked[mask] = psfs_planet_scaled[l][mask]
        simulated_data.append(planet_masked + psfs_star_final[l])

    # Molecular mapping
    flux_model_interp = interp1d(wave_model, flux_model_planet, kind='linear', bounds_error=False, fill_value='extrapolate')(mrs_wavelengths)
    simulated_data = np.array(simulated_data)
    # Add a mask to make sure that the S/N does not compute outside of the FoV !
    cc_maps_res = compute_cc_maps_coef(simulated_data, flux_model_interp, sigma=10)
    pos = int(np.around(pos_planet_th[0])), int(np.around(pos_planet_th[1]))
    # mask outside the FoV to make sure we don't overestimate the S/N by taking into account part of the PSF injected outside the FoV
    background_value = psfs_star_final[0, 0, 0]  # background at first wavelength
    mask_fov = (psfs_star_final[0] == background_value)  # mask of same shape    
    cc_maps_res_masked = np.where(~mask_fov, cc_maps_res, np.nan)  # np.nan outside FoV
    SNR, SNR1, _ = SNR_profil_cc(cc_maps_res_masked, pos)

    if plots_maps == True:
        plt.title(f'Band {band} - S/N = {SNR:.2f}')
        im=plt.imshow(cc_maps_res_masked, origin='lower',cmap='inferno',vmin=0, vmax=1)
        plt.scatter(pos[0],pos[1], marker='*', color='cyan')
        plt.scatter(pos_star[band][0],pos_star[band][1], marker='*', color='lime')
        plt.colorbar(im)
        plt.show()

    return SNR, cc_maps_res_masked, simulated_data, mrs_wavelengths


####################################################
def solve_min_flux_GJ504b(sep, wave, flux, band, theta_deg, SN_target, f_low=1e-3, f_high=1, verbose=False):
    """
    Find minimal flux to reach S/N >= SN_target using a robust root-finder.
    Returns: f_final, final_SNR, sim_data, mrs_waves
    """

    def sn_residual(f):
        try:
            sn, _, _ = simulation_MRS_single_band_GJ504b(
                wave, flux*f, band, sep, theta_deg,
                system_name=None, plot_models=False, plots_maps=False)
            # If sn is invalid, return large positive residual so brentq avoids it
            return np.inf if not np.isfinite(sn) else sn - SN_target
        except:
            return np.inf

    # If lower bound already sufficient
    if sn_residual(f_low) >= 0:
        f_final = f_low
    else:
        try:
            # Brentq automatically finds a root in the interval
            f_final = brentq(sn_residual, f_low, f_high, xtol=1e-3, maxiter=10000)
        except ValueError:
            if verbose:
                print(f"Cannot find minimal flux for sep={sep}, band={band}")
            return None, None, None, None

    # Compute final SNR once
    final_SNR, sim_data, mrs_waves = simulation_MRS_single_band_GJ504b(wave, flux*f_final, band, sep, theta_deg,
        system_name=None, plot_models=False, plots_maps=True)

    # Ensure final SNR meets or exceeds target (in case of numerical tolerance)
    if final_SNR < SN_target:
        f_final *= SN_target / final_SNR
        final_SNR, sim_data, mrs_waves = simulation_MRS_single_band_GJ504b(wave, flux*f_final, band, sep, theta_deg,
                                                                    system_name=None, 
                                                                    plot_models=False, plots_maps=True)

    if verbose:
        print(f"sep={sep:.2f}, band={band}, minimal flux={f_final:.6e}, final S/N={final_SNR:.3f}")

    return f_final, final_SNR, sim_data, mrs_waves



######################################################################

