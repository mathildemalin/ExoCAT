import numpy as np
import matplotlib.pyplot as plt

# JWST Pipeline
from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Image2Pipeline
from jwst.associations.lib.member import Member
from jwst.associations.asn_from_list import asn_from_list
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase

# Stellar PSF estimation
import scipy.optimize as optimize
from numpy.linalg import eig
from scipy import ndimage

# Image processing tools
from ExoMIRI.utils import *

# Astropy functions
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling import fitting
from astropy.io import fits
from astropy.io.fits import getheader

pix_size = 0.11


###################################################################################################
#  JWST PIPELINE FUNCTIONS : stage 1 & 2
###################################################################################################

def detector1(filenames,det1_dir):
    """Detector 1 Pipeline stage"""
    det1 = Detector1Pipeline()
    det1.output_dir = det1_dir
    det1.ramp_fit.maximum_cores = "quarter" #number of cores used ("all", "half","quarter" or direclty the number)
    det1.jump.maximum_cores = "quarter" 
    det1.jump.find_showers = False 
    #det1.ipc.skip = True (by default now)
    det1.jump.rejection_threshold = 10 #(to increase if not many group in the observation, about less than 500)
    #det1.ramp_fit.save_opt = True (to save the ramp)
    det1.save_results = True
    det1(filenames)
    return det1

def image2_nocal(filenames, image2_dir):
    """Image 2 Pipeline stage"""
    image2 = Image2Pipeline()
    image2.output_dir = image2_dir
    image2.bkg_subtract.skip = False # Subtract background : if given in filenames
    image2.assign_wcs.skip = False
    image2.flat_field.skip = True # Increase the "glowstick" -> skipped
    image2.photom.skip = True # Photometric calibration = skip to apply custom calibration (pipeline version below 12.5 and crds 1140) 
    image2.save_results = True
    image2(filenames)
    return image2

def image2_cal(filenames, cal2_dir):
    """Image 2 Pipeline stage"""
    image2 = Image2Pipeline()
    image2.output_dir = cal2_dir
    image2.bkg_subtract.skip = False # Subtract background
    image2.assign_wcs.skip = False
    image2.flat_field.skip = True # Increase the "glowstick" -> skipped
    image2.photom.skip = False # Photometric calibration : ok after pipeline 12.5 and crds 1140
    image2.save_results = True
    image2(filenames)
    return image2

def writeasnfile(files, files_bkg, name): 
    """ Write the asn files from the uncalibrated files """
    
    # setup an empty level 2 association structure
    asn = asn_from_list(files, rule=DMSLevel2bBase)
    asn.data['products'] = None

    # set the association name
    asn_name = name

    # set some metadata
    asn['asn_pool'] = asn_name + '_pool'
    asn['asn_type'] = 'image2'

    for n, sci in enumerate(files):
        asn.new_product('{}_dither{}'.format(asn_name, str(n+1)))
        sci_member = Member({'expname': sci, 'exptype': 'science'})    
        new_members = asn.current_product['members']
        new_members.append(sci_member)

        for bkg in files_bkg:
            bkg_member = Member({'expname': bkg, 'exptype': 'background'})
            new_members.append(bkg_member)

    # print the association and save to file
    name, ser = asn.dump()
    print(ser)

    asn_file = asn_name+ '_lvl2_asn.json'
    with open(asn_file, 'w') as f:
        f.write(ser)
    
    return asn_file

# Second way to write the ASN file, considering all files as science data.
def writeasnfile2(files, name):
    # setup an empty level 2 association structure
    asn = asn_from_list(files, rule=DMSLevel2bBase)
    asn.data['products'] = None

    # set the association name
    asn_name = name

    # set some metadata
    asn['asn_pool'] = asn_name + '_pool'
    asn['asn_type'] = 'image2'

    for n, sci in enumerate(files):
        asn.new_product('{}_dither{}'.format(asn_name, str(n+1)))
        sci_member = Member({'expname': sci, 'exptype': 'science'})    
        new_members = asn.current_product['members']
        new_members.append(sci_member)

    name, ser = asn.dump()
    print(ser)

    asn_file = asn_name + '_lvl2_asn.json'
    with open(asn_file, 'w') as f:
        f.write(ser)
    
    return asn_file

########## Applied sigma clipping on the data : could be done with faster and easier methods

def clean_data(data_planet, sigma=3): 
    """
    Cleans the input data by replacing NaN values with the local median 
    and applying sigma clipping to remove outliers.

    Parameters:
    data_planet (numpy.ndarray): 2D array containing the input data.
    sigma (float, optional): Threshold for sigma clipping. Defaults to 3.

    Returns:
    numpy.ndarray: Cleaned version of the input data.
    """
    
    win = 2  # Size of the local window for median filtering
    size_i, size_j = np.shape(data_planet)  # Get dimensions of the input data
    
    data_corr = np.copy(data_planet)  # Create a copy to avoid modifying the original data

    # Iterate over the pixels within a valid range, avoiding edges
    for i in range(win, size_i - win):
        for j in range(win, size_j - win):
            # Extract a local window of pixels around the current pixel
            data_window = data_planet[i-win:i+win+1, j-win:j+win+1]
            
            # Select only valid (non-NaN) values in the window
            valid_data = data_window[~np.isnan(data_window)]
            
            # Proceed only if there are valid pixels in the window
            if valid_data.size > 0:
                std = np.nanstd(valid_data)  # Compute standard deviation of valid data
                median = np.nanmedian(valid_data)  # Compute median of valid data
                
                # Replace NaN values in the original data with the local median
                pixel = np.nan_to_num(data_planet[i, j], nan=median)
                
                # Apply sigma clipping: if pixel deviates significantly from median, replace it
                if abs(pixel - median) > sigma * std:
                    data_corr[i, j] = median  # Replace outlier with median

    return data_corr  # Return the cleaned data array

################################################
### Measure S/N

def measure_sn(data_sub, pos_planet_b, center_corono, wavelength):
    """
    Measure the Signal-to-Noise Ratio (SNR) at a given planet's position.

    Parameters:
    data_sub (numpy.ndarray): Processed image data.
    pos_planet_b (tuple): (x, y) coordinates of the planet.
    center_corono (tuple): (x, y) coordinates of the coronagraphic center.
    wavelength (float): Observing wavelength in microns.

    Returns:
    float: Signal-to-Noise Ratio (SNR).
    """

    pix_size = 0.11  # Pixel scale in arcseconds
    # Compute the diffraction-limited resolution element in pixels
    lambdaD_pix = ((wavelength * 10**-6) / 6.57) * (180 * 3600 / np.pi) / pix_size

    # Compute the separation of the planet from the coronagraphic center
    sep_planet = np.sqrt((pos_planet_b[0] - center_corono[0])**2 + 
                         (pos_planet_b[1] - center_corono[1])**2)

    h, w = np.shape(data_sub)  # Get image dimensions

    # Create a circular mask around the planet position to exclude it from the noise calculation
    mask_planets = create_circular_mask(h, w, pos_planet_b, 2 * lambdaD_pix)

    # Create an annular mask centered on the coronagraphic center at the planet's separation
    mask_anneau = createAnnularMask(h, w, (center_corono[1], center_corono[0]), 
                                    sep_planet - 2 * lambdaD_pix / 2, 
                                    sep_planet + 2 * lambdaD_pix / 2)

    # Combine masks to exclude the planet while defining the noise region
    mask_err = np.ma.mask_or(~mask_anneau, mask_planets)
    data_mask = np.ma.array(data_sub, mask=mask_err)

    # Extract noise values from the annular region and compute standard deviation
    noise_data = data_mask.data[data_mask.mask == False]
    noise = np.nanstd(noise_data)

    # Measure the planet's signal by summing pixel values in a small circular aperture
    mask_planets = create_circular_mask(h, w, pos_planet_b, 1.2 * lambdaD_pix)
    planet = np.ma.array(data_sub, mask=~mask_planets)
    planet_data = planet.data[planet.mask == False]
    planet_sum = np.nansum(planet_data)

    # Compute signal as the sum of planet pixels normalized by the square root of their count
    signal = planet_sum / np.sqrt(len(planet_data))

    # Compute the Signal-to-Noise Ratio (SNR)
    SNR = np.array(signal) / np.array(noise)

    return SNR

################################################
### Measure the position of the planet

def measure_pos_planet(data_sub, RA_Offset, Dec_Offset, center, rotation=False, angle=None, plot_fit=False):
    """ 
    Measure the planet's position using a Gaussian fit.
    
    Parameters:
    data_sub (numpy.ndarray): Subtracted data image.
    RA_Offset (float): Initial guess for the RA offset.
    Dec_Offset (float): Initial guess for the DEC offset.
    center (tuple): Coordinates of the coronagraph center.
    rotation (bool, optional): Whether to apply a rotation to the image. Default is False.
    angle (float, optional): Rotation angle (in degrees) if rotation is applied.
    plot_fit (bool, optional): Whether to plot the Gaussian fit result. Default is False.

    Returns:
    tuple: The best-fit position of the planet in the image (RA, DEC in pixels).
    """

    # Pixel scale of the image (arcsec per pixel)
    pix_size = 0.11  
    a = 10  # Size of the region around the estimated planet position for fitting

    # Initial estimate of the planet's position in pixels
    pos_planet_init = [center[0] - (RA_Offset / pix_size), center[1] + Dec_Offset / pix_size]
    
    # Offset in pixels relative to the center
    offset_pix = [-RA_Offset / pix_size, Dec_Offset / pix_size]

    if rotation:  
        # Convert angle to radians and apply rotation transformation
        theta = np.radians(-angle)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta),  np.cos(theta)]])
        offset_planet = np.dot(rotation_matrix, offset_pix)

        # Convert back to arcseconds
        RA_Offset_det = offset_planet[0] * pix_size
        Dec_Offset_det = offset_planet[1] * pix_size
        
        # Compute the new estimated position after rotation
        pos_planet = [center[0] + (RA_Offset_det / pix_size), center[1] + Dec_Offset_det / pix_size]
    
    else:
        # If no rotation, use the initial estimated position
        pos_planet = pos_planet_init

    # Define bounding box around the estimated planet position for fitting
    xlim = [int(pos_planet[0] - a), int(pos_planet[0] + a)]
    ylim = [int(pos_planet[1] - a), int(pos_planet[1] + a)]
        
    # Extract the sub-image around the estimated planet position
    corono_image = data_sub[ylim[0]:ylim[1], xlim[0]:xlim[1]]

    # Find the initial peak position in the sub-image
    y0, x0 = np.where(corono_image == np.nanmax(corono_image))

    # Initialize a 2D Gaussian fit model
    gauss_init = Gaussian2D(amplitude=np.nanmax(corono_image), 
                            x_mean=x0[0], y_mean=y0[0], 
                            x_stddev=1, y_stddev=1, theta=1.) 

    # Generate coordinate grids for fitting
    shape1, shape2 = corono_image.shape
    y, x = np.mgrid[:shape1, :shape2]

    # Fit the Gaussian model to the extracted region
    fit_model = fitting.LevMarLSQFitter()
    fit = fit_model(gauss_init, x, y, corono_image)

    # Extract the best-fit position of the planet
    pos_planet = np.array([fit.x_mean.value, fit.y_mean.value])
    pos_planet_b = pos_planet[0] + xlim[0], pos_planet[1] + ylim[0]

    # Plot the fit results if requested
    if plot_fit:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the original extracted sub-image
        im = ax[0].imshow(corono_image, origin='lower')
        ax[0].set_title("Data")
        fig.colorbar(im, ax=ax[0], orientation="vertical")

        # Plot the best-fit Gaussian model
        im = ax[1].imshow(fit(x, y), origin='lower')
        ax[1].set_title("Fit")
        fig.colorbar(im, ax=ax[1], orientation="vertical")

        # Plot the residuals (data - model)
        im = ax[2].imshow(corono_image - fit(x, y), origin='lower')
        ax[2].set_title("Residual")
        fig.colorbar(im, ax=ax[2], orientation="vertical")

        plt.show()

    return pos_planet_b  # Return the best-fit position in pixels


def rotate_coordinates(RA_Offset, Dec_Offset, angle):
    """ 
    Rotate the planet's position by a given angle.
    
    Parameters:
    RA_Offset (float): Position of the planet in RA (arcseconds).
    Dec_Offset (float): Position of the planet in DEC (arcseconds).
    angle (float): Rotation angle of the image (in degrees).

    Returns:
    tuple: Rotated position (RA_Offset_det, Dec_Offset_det) in arcseconds.
    """

    # Pixel scale of the image (arcsec per pixel)
    pix_size = 0.11  

    # Convert the planet's position from arcseconds to pixels
    offset_pix = [-RA_Offset / pix_size, Dec_Offset / pix_size]

    # Convert the rotation angle from degrees to radians
    theta = np.radians(-angle)

    # Define the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta),  np.cos(theta)]])

    # Apply the rotation to the planet's position
    offset_planet = np.dot(rotation_matrix, offset_pix)

    # Convert the rotated position back to arcseconds
    RA_Offset_det = offset_planet[0] * pix_size
    Dec_Offset_det = offset_planet[1] * pix_size

    return RA_Offset_det, Dec_Offset_det


###################################################################################################
#   Reference star subtraction 
#
###################################################################################################
### with linear optimisation

# Optimised for HD 95086 and its background objects (Mâlin et al., 2024)  : 
def opt_1ref(data_target, data_ref, center_corono, num_ref=0, rmin=1, rmax=2.5):
    """ 
    Create an optimized reference from a single reference observation.

    Parameters:
    data_target (numpy.ndarray): Target data array (image to process).
    data_ref (list of numpy.ndarray): List of reference images.
    center_corono (tuple): Coordinates of the coronagraphic center (y, x).
    num_ref (int, optional): Index of the reference image to use (default is 0).
    rmin (float, optional): Inner radius of the optimization mask in arcseconds (default is 1).
    rmax (float, optional): Outer radius of the optimization mask in arcseconds (default is 2.5).

    Returns:
    numpy.ndarray: Optimized reference image for subtraction.
    """

    nb_refs = len(data_ref)  # Number of available reference images
    pix_size = 0.11  # Pixel scale in arcseconds per pixel

    # 1 - Create the optimization mask
    h, w = np.shape(data_target)  # Get image dimensions

    # Create an annular mask from rmin to rmax around the coronagraph center
    mask_anneau = createAnnularMask(h, w, (center_corono[1], center_corono[0]), 
                                    rmin / pix_size, rmax / pix_size)
    
    # Create a circular mask to exclude a background object (specific to HD 95086)
    mask_galaxy = create_circular_mask(h, w, (65, 55), 17)  

    # Combine both masks to create the final mask
    mask_tot = np.ma.mask_or(~mask_anneau, mask_galaxy)

    # Apply the mask to the target and reference images
    data_mask = np.ma.array(data_target, mask=mask_tot)
    ref_mask = [np.ma.array(data_ref[r], mask=mask_tot) for r in range(nb_refs)]

    # 2 - Define the function to minimize for optimal scaling
    def function_min(P):
        """ Minimize the residuals by scaling the reference image. """
        data = data_mask
        model = ref_mask[num_ref]
        return np.sum((data - P * model) ** 2.)  # Sum of squared residuals

    # Optimize the scaling factor P using a simplex algorithm (Nelder-Mead)
    P = optimize.fmin(function_min, x0=(0.3))
    
    print("Rescaled flux factor:", P)

    # Scale the selected reference image
    new_ref = data_ref[num_ref] * P

    return new_ref  # Return the optimized reference image
  
# For N references : 
def optimize_refs(data_target, data_ref, center_corono, rmin=0.3, rmax=8):
    """ 
    Create an optimized reference from the reference observations 
    
    Parameters:
    data_target (numpy.ndarray): Target data array.
    data_ref (list of numpy.ndarray): List of reference data arrays.
    center_corono (tuple): Center of the coronograph.
    rmin (float): Minimum radius for annular mask in arcsec (default is 0.3).
    rmax (float): Maximum radius for annular mask in arcsec (default is 8).

    Returns:
    numpy.ndarray: Data subtracted from optimized reference.
    """
    nb_refs = len(data_ref)
    
    h, w = np.shape(data_target)
    pixscale = 0.11 # MIRI pixel scale

    # Mask where to optimize the subtraction from rmin to rmax
    mask_planets = createAnnularMask(h, w, center=center_corono, small_radius=rmin/pixscale, big_radius=rmax/pixscale)
    data_mask = np.ma.array(data_target, mask=~mask_planets)
    data_ref_mask = [np.ma.array(data_ref[i], mask=~mask_planets) for i in range(nb_refs)]

    # Function to minimize
    def function_min(P):
        return np.sum((data_mask - np.sum(P[i] * data_ref_mask[i] for i in range(nb_refs)))**2)

    # Initialize parameters
    b = np.ones(nb_refs) / nb_refs
    # Optimize
    P = optimize.fmin(function_min, x0=b, disp=False)

    # Compute optimized reference
    new_ref = sum(P[i] * data_ref[i] for i in range(nb_refs))

    return data_target - new_ref

def optimize_refs_4Q(data_target, data_ref, center_corono,rmin,rmax):
    """ 
    Create an optimized reference from the reference observations :
    -> optimize_refs apply quadrant by quadrant 
    
    Parameters:
    data_target (numpy.ndarray): Target data array.
    data_ref (list of numpy.ndarray): List of reference data arrays.
    center_corono (tuple): Center of the coronograph.
    rmin (float): Minimum radius for annular mask in arcsec (default is 0.3).
    rmax (float): Maximum radius for annular mask in arcsec (default is 8).
    
    Returns:
    numpy.ndarray: Data subtracted from optimized reference.
    """
    
    cut = np.around(center_corono,0)
    nb_refs = len(data_ref)
    
    data_cut1 = data_target[:int(cut[0]), :int(cut[1])]
    data_cut2 = data_target[:int(cut[0]), int(cut[1]):]
    data_cut3 = data_target[int(cut[0]):, int(cut[1]):]
    data_cut4 = data_target[int(cut[0]):, :int(cut[1])]
    
    refs_cut1 = [data_ref[r][:int(cut[0]), :int(cut[1])]  for r in range(nb_refs)]
    refs_cut2 = [data_ref[r][:int(cut[0]), int(cut[1]):]  for r in range(nb_refs)]
    refs_cut3 = [data_ref[r][int(cut[0]):, int(cut[1]):]  for r in range(nb_refs)]
    refs_cut4 = [data_ref[r][int(cut[0]):, :int(cut[1])]  for r in range(nb_refs)]
    
    data_sub_1 = optimize_refs(data_cut1, refs_cut1,center_corono,rmin,rmax)
    data_sub_2 = optimize_refs(data_cut2, refs_cut2,center_corono,rmin,rmax)
    data_sub_3 = optimize_refs(data_cut3, refs_cut3,center_corono,rmin,rmax)
    data_sub_4 = optimize_refs(data_cut4, refs_cut4,center_corono,rmin,rmax)
    
    lower = np.concatenate([data_sub_1,data_sub_2], axis=1)
    upper = np.concatenate([data_sub_4,data_sub_3], axis=1)
    data_sub = np.concatenate([lower,upper], axis=0)

    return data_sub

### with PCA
def sub_PCA(data_red, data_red_ref,center_corono=None, Ktrunc=None, mask_center=False, mask_size=None,):

    """ 
    Create an optimize reference from the reference observations 
    - using classical PCA method removing Ktrunc components (equal to the number of reference observations by default)
    
    Parameters:
    (data have to be the same shape + center at the center of the images)
    data_red (numpy.ndarray): Target data array.
    data_red_ref (list of numpy.ndarray): List of reference data arrays.
    Ktrunc : Component to be removed

    Option to mask the center part of the image : 
    mask_center = False by default (no mask)
    mask_size = size in arcsec of the mask
    
    Returns:
    numpy.ndarray: Data subtracted from optimized reference.
    """
    
    nb_ref = len(data_red_ref)
    if Ktrunc==None :
        Ktrunc = nb_ref
    pix_size=0.11
    
    # option to mask the center :
    
    if mask_center == True :
        data_red_ref_mask = np.copy(data_red_ref)
        data_red_mask = np.copy(data_red)
        
        h,w = np.shape(data_red)
        mask = create_circular_mask(h,w,(center_corono[0],center_corono[1]), mask_size/pix_size)
        data_red_mask = np.ma.array(data_red_mask, mask = mask)
        data_red_mask.data[data_red_mask.mask==True] = np.nan
        
        data_red_ref_mask = [np.ma.array(data_red_ref_mask[r], mask = mask) for r in range(nb_ref)]
        for r in range(nb_ref):
            data_red_ref_mask[r].data[data_red_ref_mask[r].mask==True] = np.nan

    #Ref in 2D : nb ref x nb pixels
        ref_flat = [np.ndarray.flatten(data_red_ref_mask[r]) for r in range(nb_ref)]
        data_flat = np.ndarray.flatten(data_red_mask)
    else :
        ref_flat = [np.ndarray.flatten(data_red_ref[r]) for r in range(nb_ref)]
        data_flat = np.ndarray.flatten(data_red)
    
    ref_mean = [np.nanmean(ref_flat[r])  for r in range(nb_ref)]
    Xref = [ref_flat[r] - ref_mean[r] for r in range(nb_ref)]
    Xref = np.array(Xref)

    #Data in 2D : nb_frame (always 1) x nb pixels 
    
    data_mean = np.nanmean(data_flat)
    Xdata = data_flat - data_mean
    Xdata = np.nan_to_num(Xdata)
    Xref = np.nan_to_num(Xref)
    #print(np.shape(Xdata))

    Npix = len(Xdata)

    #nb of modes that we want to remove
    Ktrunc = int(Ktrunc)

    #covariance matrix measurement
    V = np.dot(Xref,np.transpose(Xref))/Npix #useless ici
    cov_matrix = np.cov(Xref)
    #measuring eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(cov_matrix)
    # modes from the ref
    modes = np.dot(np.transpose(eigenvectors),Xref)
    # mode normalization
    normv = np.sqrt(np.sum(modes**2,1))
    modes = [modes[r,:]/normv[r] for r in range(nb_ref)]
    coef = np.dot(Xdata, np.transpose(modes[:Ktrunc-1]))
    # projection of ref modes onto image coefficients
    X_data_projected = np.dot(np.transpose(modes[:Ktrunc-1]),coef)  
    ref_reconstructed = X_data_projected + data_mean #juste data mean ?
    # Reshape the new 2D ref
    if mask_center == True : # err here 
        print()
        print(data_red_mask)
        print(np.shape(data_red_mask))
        h,w = np.shape(data_red_mask)
        ref_reshape = np.reshape(ref_reconstructed, (h,w))
        data_sub = data_red_mask - ref_reshape
    else : 
        h,w = np.shape(data_red)
        ref_reshape = np.reshape(ref_reconstructed, (h,w))
        data_sub = data_red - ref_reshape
    
    return data_sub

# with mask on a region (in rings)
def sub_PCA_rings(data_target, data_ref, center_corono, Ktrunc, width = 2, rmin=0, rmax=10): #a,mask_size
    """ 
    Create an optimize reference from the reference observations 
    - using classical PCA method removing Ktrunc components (equal to the number of reference observations by default)
    - apply the method rings by rings : width = 2 by default

    Parameters: (data have to be the same shape + center at the center of the images)
    data_target (numpy.ndarray): Target data array.
    data_ref (list of numpy.ndarray): List of reference data arrays.
    center_corono (tuple): Center of the coronograph.
    Ktrunc : Component to be removed.
    width,rmin,rmax : size of the rings, min, max in arcsec
    
    Returns:
    numpy.ndarray: Data subtracted from optimized reference.
    """
    
    pix_size = 0.11
    nb_ref = len(data_ref)
    
    regions = np.arange(rmin,rmax,width) #in arcsec : rmin, rmax, and width r
    new_ref_list = []
    
    for i in regions:

        h,w=np.shape(data_target)
        center = center_corono[1], center_corono[0] 
        small_radius = i/pix_size
        big_radius = (i+width) /pix_size
        region_pca = createAnnularMask(h, w, center, small_radius, big_radius)

        #Ref en 2D : nb_ref x nb pixels in the ring
        Xref = []
        for r in range(nb_ref):
            data_ref_mask = np.ma.array(data_ref[r], mask = ~region_pca) 
            x_region, y_region = np.where(data_ref_mask.mask == False)
            array_ref = [data_ref_mask[x_region[i], y_region[i]] for i in range(len(x_region))]
            ref_flat = np.ndarray.flatten(np.array(array_ref)) 
            ref_mean = np.nanmean(ref_flat) 
            Xref.append(np.array(ref_flat - ref_mean))

        data_mask = np.ma.array(data_target, mask = ~region_pca)
        x_region, y_region = np.where(data_mask.mask == False)
        array_data = [data_mask[x_region[i], y_region[i]] for i in range(len(x_region))]
        data_flat = np.ndarray.flatten(np.array(array_data))
        data_mean = np.nanmean(data_flat)
        Xdata = data_flat - data_mean

        Npix = len(Xdata)
        Ktrunc = int(Ktrunc)
        V = np.dot(Xref,np.transpose(Xref))/Npix 
        Xref = np.nan_to_num(Xref)
        cov_matrix = np.cov(Xref)
        eigenvalues, eigenvectors = eig(cov_matrix)
        modes = np.dot(np.transpose(eigenvectors),Xref)
        normv = np.sqrt(np.sum(modes**2,1))
        modes = [modes[r,:]/normv[r] for r in range(nb_ref)]
        coef = np.dot(Xdata, np.transpose(modes[:Ktrunc-1]))

        X_data_projected = np.dot(np.transpose(modes[:Ktrunc-1]),coef)  
        ref_reconstructed = X_data_projected + data_mean #juste data mean ??
        len(ref_reconstructed)

        new_ref = np.zeros(np.shape(data_target))
        new_ref[x_region, y_region] = ref_reconstructed
        new_ref_list.append(new_ref)

    new_ref_sum = np.sum(new_ref_list, axis =0)
    data_sub = data_target - new_ref_sum
    
    return data_sub

# using PCA separately in each of the four quadrants.
def sub_4Q_PCA(data_target, data_ref, center_corono, Ktrunc, mask_center=False, mask_size=None):

    """ 
    Create an optimize reference from the reference observations 
    - call sub_PCA : using classical PCA method removing Ktrunc components (equal to the number of reference observations by default)
    - apply the method quadrant by quadrant
    
    Parameters: (data have to be the same shape + center at the center of the images)
    data_target (numpy.ndarray): Target data array.
    data_ref (list of numpy.ndarray): List of reference data arrays.
    center_corono (tuple): Center of the coronograph.
    
    Returns:
    numpy.ndarray: Data subtracted from optimized reference.
    """
    
    cut = np.around(center_corono,0)
    nb_refs = len(data_ref)
    
    data_cut1 = data_target[:int(cut[0]), :int(cut[1])]
    data_cut2 = data_target[:int(cut[0]), int(cut[1]):]
    data_cut3 = data_target[int(cut[0]):, int(cut[1]):]
    data_cut4 = data_target[int(cut[0]):, :int(cut[1])]
    
    refs_cut1 = [data_ref[r][:int(cut[0]), :int(cut[1])]  for r in range(nb_refs)]
    refs_cut2 = [data_ref[r][:int(cut[0]), int(cut[1]):]  for r in range(nb_refs)]
    refs_cut3 = [data_ref[r][int(cut[0]):, int(cut[1]):]  for r in range(nb_refs)]
    refs_cut4 = [data_ref[r][int(cut[0]):, :int(cut[1])]  for r in range(nb_refs)]

    data_sub_PCA_1 = sub_PCA(data_cut1, refs_cut1,center_corono,Ktrunc,mask_center, mask_size)
    data_sub_PCA_2 = sub_PCA(data_cut2, refs_cut2,center_corono,Ktrunc,mask_center, mask_size)
    data_sub_PCA_3 = sub_PCA(data_cut3, refs_cut3,center_corono,Ktrunc,mask_center, mask_size)
    data_sub_PCA_4 = sub_PCA(data_cut4, refs_cut4,center_corono,Ktrunc,mask_center, mask_size)
    
    lower = np.concatenate([data_sub_PCA_1,data_sub_PCA_2], axis=1)
    upper = np.concatenate([data_sub_PCA_4,data_sub_PCA_3], axis=1)
    data_sub_PCA_4Q = np.concatenate([lower,upper], axis=0)

    return data_sub_PCA_4Q

###################################################################################################
## Plotting functions 

def plot_planet(data_sub, center_corono, vmin=-2, vmax=10, a=30, plot_pos_planet=False, pos_planet_b=None, savefig=False, output_dir=None):
    """ 
    Plot the data centered on the coronagraph's center.

    Parameters:
    - data_sub (numpy.ndarray): Image data to plot.
    - center_corono (list of tuples): Coordinates of the coronagraph's center.
    - vmin, vmax (float, optional): Color scale limits.
    - a (int, optional): Field of view (in pixels) from the center.
    - plot_pos_planet (bool, optional): Whether to overlay the expected planet position.
    - pos_planet_b (list of tuples, optional): Expected planet positions.
    - savefig (bool, optional): Whether to save the figure.
    - output_dir (str, optional): Directory to save the figure.

    """

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for ft in range(2):  # Loop over two filters/views
        xlim = [int(center_corono[ft][1] - a), int(center_corono[ft][1] + a)]
        ylim = [int(center_corono[ft][0] - a), int(center_corono[ft][0] + a)]
        
        # Display image
        im = ax[ft].imshow(data_sub[ft], origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        
        # Mark coronagraph center
        ax[ft].scatter(center_corono[ft][1], center_corono[ft][0], color="lime", marker="*", label="Corono center")
        
        # Mark expected planet position if enabled
        if plot_pos_planet:
            ax[ft].scatter(pos_planet_b[ft][0], pos_planet_b[ft][1], color="blue", alpha=0.5, marker="P", label="Expected planet")
        
        # Colorbar and axis limits
        fig.colorbar(im, ax=ax[ft], orientation="vertical")
        ax[ft].set_xlim(xlim)
        ax[ft].set_ylim(ylim)

    # Save figure if requested
    if savefig:
        plt.savefig(output_dir + ".png", bbox_inches="tight", pad_inches=0)

    plt.show()

# same for 3 filters :
def plot_planet_3(data_sub, center_corono, a=None, vmin=-2, vmax=12, 
                  plot_pos_planet=False, pos_planet_b=None, 
                  savefig=False, output_dir=None):
    """ 
    Plot the data centered on the coronagraph's centers for three different filters.

    Parameters:
    - data_sub (numpy.ndarray): Image data for each filter.
    - center_corono (list of tuples): Coordinates of the coronagraph center.
    - a (int, optional): Field of view (in pixels) from the center.
    - vmin, vmax (float, optional): Color scale limits.
    - plot_pos_planet (bool, optional): Whether to overlay the expected planet position.
    - pos_planet_b (list of tuples, optional): Expected planet positions.
    - savefig (bool, optional): Whether to save the figure.
    - output_dir (str, optional): Directory to save the figure.
    """

    # Set default field of view if not specified
    if a is None:
        a = 28

    fig, ax = plt.subplots(1, 3, figsize=(17, 4))  # Create 3 subplots for three filters

    for ft in range(3):  # Loop over three filters
        xlim = [int(center_corono[ft][1] - a), int(center_corono[ft][1] + a)]
        ylim = [int(center_corono[ft][0] - a), int(center_corono[ft][0] + a)]

        # Adjust color scale for third filter
        vmin_ft, vmax_ft = (vmin, vmax) if ft < 2 else (vmin / 2, vmax / 2)

        # Display image
        im = ax[ft].imshow(data_sub[ft], origin="lower", cmap="inferno", vmin=vmin_ft, vmax=vmax_ft)

        # Mark coronagraph center
        ax[ft].scatter(center_corono[ft][1], center_corono[ft][0], color="lime", marker="*", label="Corono center")

        # Mark expected planet position if enabled
        if plot_pos_planet:
            ax[ft].scatter(pos_planet_b[ft][0], pos_planet_b[ft][1], color="blue", alpha=0.5, marker="P", label="Expected planet")

        # Add colorbar
        fig.colorbar(im, ax=ax[ft], orientation="vertical")

        # Set axis limits
        ax[ft].set_xlim(xlim)
        ax[ft].set_ylim(ylim)

        # Style axes
        ax[ft].tick_params(direction='in', width=1.2, size=5, color='w')
        ax[ft].set_xticklabels([], color='w')
        ax[ft].set_yticklabels([], color='w')

        # Scale annotation (1 arcsecond)
        ax[ft].arrow(xlim[0] + 5, ylim[0] + 5, (1 / pix_size), 0, color="white")
        ax[ft].annotate('1" ', xy=(xlim[0] + 5, ylim[0] + 5), 
                        xytext=(xlim[0] + 5, ylim[0] + 5 + 2), color="white", size=14)

    # Save figure if requested
    if savefig:
        plt.savefig(output_dir + ".pdf", bbox_inches="tight", pad_inches=0)

    plt.show()

# Showing axis in arcsec - adapted for GJ 504 b (Mâlin et al, 2025)
def plot_planet_offset(data_sub, vmin=-2, vmax=12,
                       plot_pos_planet=False, savefig=False, output_dir=None):
    """ 
    Plot the data centered on the coronagraphs' centers for three different filters.

    Parameters:
    - data_sub (numpy.ndarray): Image data for each filter.
    - vmin, vmax (float, optional): Color scale limits.
    - fov (int, optional): Field of view size in pixels.
    - plot_pos_planet (bool, optional): Whether to overlay the expected planet position.
    - savefig (bool, optional): Whether to save the figure.
    - output_dir (str, optional): Directory to save the figure.
    """

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))  # Create 3 subplots for three filters
    fov_lim = 4.0  # Field of view in arcseconds

    for ft, title in enumerate(["F1065C", "F1140C", "F1550C"]):
        fov_x = np.shape(data_sub[ft])[1] * pix_size / 2
        fov_y = np.shape(data_sub[ft])[0] * pix_size / 2

        # Adjust color scale for the third filter
        vmin_ft, vmax_ft = (vmin, vmax) if ft < 2 else (vmin + 0.5, vmax - 7)

        # Display image
        im = ax[ft].imshow(data_sub[ft], extent=(-fov_x, fov_x, -fov_y, fov_y),
                           origin="lower", cmap="inferno", vmin=vmin_ft, vmax=vmax_ft)

        # Mark coronagraph center
        ax[ft].scatter(0, 0, color="lime", marker="*", s=80)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax[ft], orientation="vertical")
        cbar.set_label("MJy/sr", rotation=270, labelpad=12, fontsize=12)

        # Set axis limits
        ax[ft].set_xlim(-fov_lim, fov_lim)
        ax[ft].set_ylim(-fov_lim, fov_lim)

        # Style axes
        ax[ft].tick_params(direction='in', width=1.2, size=5, color='w')
        ax[ft].set_xlabel("arcsec")
        ax[ft].set_yticks([-4, -2, 0, 2, 4])

        # Remove y-axis labels for the second and third filters
        if ft == 0:
            ax[ft].set_ylabel("arcsec")
            ax[ft].set_yticklabels([-4, -2, 0, 2, 4], color="k")
        else:
            ax[ft].set_ylabel("")
            ax[ft].set_yticklabels([])

        ax[ft].set_title(title, fontsize=16)

        # Orientation indicators (N & E)
        ax[ft].arrow(3.5, -3.5, 0, 0.8, color="white")
        ax[ft].text(3.2, -2.5, "N", color="w", fontweight="bold")
        ax[ft].arrow(3.5, -3.5, -0.8, 0, color="white")
        ax[ft].text(2.3, -3.6, "E", color="w", fontweight="bold")

        # Scale annotation (10 au)
        ax[ft].arrow(-fov_lim + 1.5, -fov_lim + 1, 10 / 17.59, 0, color="white")
        ax[ft].annotate("10 au", xy=(-fov_lim + 1, -fov_lim + 1),
                        xytext=(-fov_lim + 1, -fov_lim + 1 + 0.3), color="white", size=10)

        # Mark expected planet position if enabled (only for first filter)
        if plot_pos_planet and ft == 0:
            ax[ft].text(0, 3, "GJ 504 b", color="w", fontweight="bold")
            ax[ft].arrow(0.7, 2.8, 0.5, -0.5, color="white", head_width=0.15)

    # Save figure if requested
    if savefig:
        plt.savefig(output_dir + ".pdf", bbox_inches="tight", pad_inches=0)

    plt.show()


def plot(data, vmin , vmax, plot_planet = False, pos_planet=None, title=None, savefig=False, output_dir=None):
    plt.imshow(data, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    cbar=plt.colorbar()
    plt.title(title)
    cbar.set_label('MJy/sr', rotation=270,  labelpad = 12, fontsize=12)
    #plt.scatter(center[1],center[0], color="lime", marker="*",s=80)
    if plot_planet == True:
        plt.scatter(pos_planet[0],pos_planet[1], color="lime", marker="*",s=80)
    if savefig == True:
        plt.savefig(output_dir+".pdf",bbox_inches='tight', pad_inches=0)
    plt.show()



###########
def save_fits(data_sub, files_cal, name, other_things_to_add):
    """
    Save processed data as a FITS file with updated headers.

    Parameters:
    data_sub (numpy.ndarray): The processed image data to be saved.
    files_cal (list): List of calibration FITS files (first file is used for headers).
    name (str): Name of the output FITS file (without extension).
    other_things_to_add (str): Additional information to include in the header.

    Returns:
    None
    """
    
    # Extract the primary header from the first calibration file
    hdr1 = getheader(files_cal[0], 0)  
    # Add additional reduction information to the header
    hdr1['REDUCTION'] = other_things_to_add  
    hdr2 = getheader(files_cal[0], 1)  
    hdu1 = fits.PrimaryHDU(header=hdr1)  
    hdu2 = fits.ImageHDU(header=hdr2, data=data_sub)  
    hdul = fits.HDUList([hdu1, hdu2])  
    hdul.writeto(name + '.fits', overwrite=True)  