import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling import models, fitting



###################################################################################################
# Masks

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image size
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask

def createAnnularMask(h, w, center, small_radius, big_radius):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))

    Y, X = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = (small_radius <= distance_from_center) & (distance_from_center <= big_radius)
    return mask


def create_cone_mask(nx, ny, center, cone_axis_angle, angle_deg, length):
    """
    Create a 2D cone-shaped mask.
    
    Parameters
    ----------
    nx, ny : int
        Dimensions of the 2D array.
    center : tuple of float
        (x0, y0) coordinates of the cone apex.
	cone_axis_angle : float
		Define the cone axis (pointing upward, along +y)
    angle_deg : float
        Full opening angle of the cone in degrees.
    length : float
        Maximum radius/length of the cone.
        
    Returns
    -------
    mask : 2D numpy array
        Boolean mask (True inside the cone, False outside).
    """
    y, x = np.indices((ny, nx))
    x0, y0 = center
    dx = x - x0
    dy = y - y0
    r = np.sqrt(dx**2 + dy**2)
    
    # Angle of each pixel relative to the cone axis (assumed along +y by default)
    pixel_angle = np.degrees(np.arctan2(dy, dx))
    
    # Check if pixel is within angle and length
    angle_mask = np.abs(pixel_angle - cone_axis_angle) <= (angle_deg / 2)
    radius_mask = r <= length
    
    mask = angle_mask & radius_mask
    return mask


###################################################################################################
### Provided by N. Skaf and J. Mazoyer

def cart2polar(xx, yy):
    """
    Convert cartesian coordinates into polar coordinates
    :param xx: 2D-square array of x linear coordinates
    :param yy: 2D-square array of y linear coordinates - should have same dimensions as xx
    :return: r, xx/yy like array, normalized radius
             theta, xx/yy like array, angle in radian
    """
    phi = np.arctan2(-yy, -xx)
    theta = phi - np.min(phi)
    r = np.sqrt(xx**2 + yy**2)
    return r, theta

def mean_full_ring(im, pos_center, width_ring, r_min, r_max):
    """
    Calculates the robust mean of the full ring - like specal does.
    :param width_ring: width of the ring in pixels (can be 1 pixel)
    :param r_min: starting radius of the rings in pixels (can be 0)
    :param r_max: end radius of the rings in pixels
    :return: mean, separation arrays
    """
    x = np.arange(np.shape(im)[1]) - pos_center[0]
    y = np.arange(np.shape(im)[0]) - pos_center[1]

    xx, yy = np.meshgrid(x, y)
    r, theta = cart2polar(xx, yy)

    mean = np.zeros(r_max)
    separation = np.zeros(r_max)
    circle = np.zeros(r.shape)

    for i in range(r_min, r_max):
        circle[r < i + width_ring] = 1
        circle[r < i] = 0
        im_ring = im * circle
        separation[i] = r_min + i
        im_nonan = np.nan_to_num(im_ring)
        index = np.where(im_nonan != 0)
        im_ring_values = im_nonan[index]
        mean[i] = np.mean(im_ring_values)

    return mean, separation

def std_full_ring(im, pos_center, sigma, width_ring, r_min, r_max):
    """
    Calculates the robust standard deviation of the full ring - like specal does.
    :param width_ring: width of the ring in pixels (can be 1 pixel)
    :param r_min: starting radius of the rings in pixels (can be 0)
    :param r_max: end radius of the rings in pixels
    :param sigma: multiplicative factor
    :return: std, separation arrays
    """
    x = np.arange(np.shape(im)[1]) - pos_center[0]
    y = np.arange(np.shape(im)[0]) - pos_center[1]

    xx, yy = np.meshgrid(x, y)
    r, theta = cart2polar(xx, yy)

    std = np.zeros(r_max)
    separation = np.zeros(r_max)
    circle = np.zeros(r.shape)

    for i in range(r_min, r_max):
        circle[r < i + width_ring] = 1
        circle[r < i] = 0
        im_ring = im * circle
        separation[i] = r_min + i
        im_nonan = np.nan_to_num(im_ring)
        index = np.where(im_nonan != 0)
        im_ring_values = im_nonan[index]
        std[i] = np.std(im_ring_values) * sigma

    return std, separation


#########
def rotate_images(img, center_pix, window_size, ang, reshape=False):
    """
    Rotates an image around a specified center pixel with a given angle.
    Same method as implemented in spaceKLIP

    Parameters:
        img (ndarray): The input image array.
        center_pix (tuple): The (y, x) coordinates of the center of rotation.
        window_size (float): The size of the window (in the same units as image scale).
        ang (float): The rotation angle in degrees.
        reshape (bool, optional): If True, expands the output to fit the entire rotated image.

    Returns:
        ndarray: The rotated image.
    """
    
    # Convert window size to pixels (assuming a pixel scale of 0.11 per unit)
    window_pix = int(np.rint(window_size / 0.11 / 2))
    
    # Compute offset for centering
    offset = (window_pix - window_size / 0.11 / 2) * 0.11
    
    # Generate coordinate grids
    x = np.linspace(0, img.shape[0], img.shape[0])
    y = np.linspace(0, img.shape[1], img.shape[1])
    xmesh, ymesh = np.meshgrid(x, y)
    
    # Determine the old image center (y, x)
    old_center = [img.shape[1] / 2, img.shape[0] / 2]
    
    # Adjust mesh coordinates relative to the new center
    xmesh += center_pix[1]
    ymesh += center_pix[0]
    xmesh -= old_center[1]
    ymesh -= old_center[0]
    
    # Map new coordinates onto the original image
    new_data = ndimage.map_coordinates(img, [ymesh, xmesh])
    
    # Rotate the image by the specified angle
    rot_img = ndimage.rotate(new_data, ang, reshape=reshape)
    
    return rot_img



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
