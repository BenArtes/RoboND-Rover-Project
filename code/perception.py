import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_low_thresh=(160, 160, 160), rgb_high_thresh=(255, 255, 255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = ((img[:,:,0] >= rgb_low_thresh[0]) & (img[:,:,0] <= rgb_high_thresh[0])) \
                & ((img[:,:,1] >= rgb_low_thresh[1]) & (img[:,:,1] <= rgb_high_thresh[1])) \
                & ((img[:,:,2] >= rgb_low_thresh[2]) & (img[:,:,2] <= rgb_high_thresh[2]))
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


def find_navigable(img):
    return color_thresh(img)


def find_obstacles(img):
    return np.absolute(np.float32(find_navigable(img)) - 1)


def find_rocks(img):
    return color_thresh(img, (110, 110, 0), (256, 256, 50))


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1], img.shape[0]))
    return warped, mask


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    nav_pix_found_thresh = 60

    # NOTE: camera image is coming to you in Rover.img
    Rover.flat = (((Rover.pitch >= 0) and (Rover.pitch < 0.4)) or ((Rover.pitch < 0) and (Rover.pitch > -0.4))) and ((Rover.pitch > 359.6) or (Rover.pitch < 0.4))
    image = Rover.img
    dst_size = 5
    bottom_offset = 6
    # 1) Define source and destination points for perspective transform
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              ])

    # Get Thresholded Base Image for Debugging
    base_img_navigable = find_navigable(image)
    base_img_obstacles = find_obstacles(image)
    base_img_rocks = find_rocks(image)

    # 2) Apply perspective transform
    # Get Warped Base Image and Perspective Mask
    warped, warped_mask = perspect_transform(image, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    warped_navigable = find_navigable(warped) * warped_mask
    warped_obstacles = find_obstacles(warped) * warped_mask
    warped_rocks = find_rocks(warped) * warped_mask

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    #Rover.vision_image[:,:,0] = base_img_obstacles * 255
    #Rover.vision_image[:,:,2] = base_img_navigable * 255
    Rover.vision_image[:,:,0] = warped_obstacles * 255
    Rover.vision_image[:,:,2] = warped_navigable * 255


    # Localize Images and Update World Map
    ## Navigables
    nav_x, nav_y = rover_coords(warped_navigable)
    # Navigable pixels are the most suscetible to errors around the horizon
    # We should be driving over them anyway so only use pixels close to the rover for map making
    warped_navigable[:60, :, ] = 0
    x, y = rover_coords(warped_navigable)
    abs_x, abs_y = pix_to_world(nav_x, nav_y, Rover.pos[0], Rover.pos[1],
                                Rover.yaw, Rover.worldmap.shape[0], 10)
    if Rover.flat:
        Rover.worldmap[abs_y, abs_x, 2] += 5
    ## Obstacles
    x, y = rover_coords(warped_obstacles)
    abs_x, abs_y = pix_to_world(x, y, Rover.pos[0], Rover.pos[1],
                                Rover.yaw, Rover.worldmap.shape[0], 10)
    if Rover.flat:
        Rover.worldmap[abs_y, abs_x, 0] += 1
    ## Rocks
    x, y = rover_coords(warped_rocks)
    abs_x, abs_y = pix_to_world(x, y, Rover.pos[0], Rover.pos[1],
                                Rover.yaw, Rover.worldmap.shape[0], 10)
    if Rover.flat:
        Rover.worldmap[abs_y, abs_x, :] = 243

        # Nav Pix Found, so clear Obs and set Nav
        nav_pix = Rover.worldmap[:, :, 2] > nav_pix_found_thresh
        Rover.worldmap[nav_pix, 0] = 0
        Rover.worldmap[nav_pix, 2] = 254

    # Pixels roll over and I don't know how to clamp inline so we'll set pixels to 254 and check if they get to 255
    # If they do, we'll clamp
    # Have to check of Color Channel individually so we know which to set to 254
    clamp_pixels = (Rover.worldmap[:,:,0] > 243)
    Rover.worldmap[clamp_pixels, 0] = 243

    clamp_pixels = (Rover.worldmap[:, :, 1] > 243)
    Rover.worldmap[clamp_pixels, 1] = 243

    clamp_pixels = (Rover.worldmap[:, :, 2] > 243)
    Rover.worldmap[clamp_pixels, 2] = 243

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(nav_x, nav_y)

    return Rover