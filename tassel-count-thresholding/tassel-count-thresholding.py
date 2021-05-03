import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import matplotlib.pyplot as plt
import argparse



def hsv_threshold(img, min_hsv_val, max_hsv_val):
    """Return a binary mask with the pixels within the specified HSV color range."""
    # remove background dark area
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_init = np.expand_dims(np.prod(img[:, :, :] > 40, axis=2).astype(np.uint8), 2)
    mask_hsv = mask_init * img_hsv
    
    # create hsv range
    min_val = np.array(min_hsv_val)
    max_val = np.array(max_hsv_val)
    
    # creat mask of values fall inside
    mask = cv2.inRange(mask_hsv, min_val, max_val)
    mask = (mask == 255).astype(np.uint8)
    return mask


def opening_morph(mask, kernel_size = 5):
    """Perform opening morphological operation on binary mask."""

    ker_size_erode = kernel_size
    ker_size_dilate = kernel_size
    
    # create kernels for erosion & dilation
    kernel_erode = np.ones((ker_size_erode, ker_size_erode), np.uint8)
    kernel_dilate = np.ones((ker_size_dilate, ker_size_dilate), np.uint8)
    
    # perform opening morphological operation
    mask_eroded = cv2.erode(mask, kernel_erode)
    mask_eroded_dilated = cv2.dilate(mask_eroded, kernel_dilate)
    return mask_eroded_dilated
    
def watershed_segmentation(thresh_mask):
    """Perform watershed segmentation on a thresholded mask. Returns 2D array of segment labels, each with a unique integer."""
    
    # compute EDT distance map
    distance_map = ndimage.distance_transform_edt(thresh_mask)
    
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    local_maximas = peak_local_max(distance_map, indices=False, min_distance=2,
     	labels=thresh_mask)
    
    markers, num_features = ndimage.label(local_maximas, structure=np.ones((3, 3)))
    labels = watershed(-distance_map,markers,mask=thresh_mask)
    return labels


def filter_contours(contours, min_area = 20):
    """Filter out contours with insignificant area."""
    
    if not isinstance(contours, np.ndarray):
        new_contours = np.array(contours)
    
    # find area of each contour
    areas = list(map(cv2.contourArea,contours))
    areas = np.array(areas)
    
    # remove contours with small area
    proper_cnts = new_contours[areas > min_area]
    return proper_cnts


def center_of_mass(M):
    """Return center of mass of an area."""
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return [cx,cy]


def plot_centers(centers, img):
    """Plot center points on the original image"""
    
    x,y = np.transpose(centers)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,10))
    plt.imshow(img_rgb)
    plt.scatter(x,y,s=15, c='red', marker='o')
    
    
def write_centers_to_file(centers, filepath):
    """Write center coordinates to txt file."""
    
    with open(filepath, 'w') as f:
        for item in centers:
            f.write(f"{item[0]} {item[1]}\n")
            
def main(img_path, verbosity=2):
    """Find corn plants in aerial image, create annotation file."""
    
    # load image
    img = cv2.imread(img_path)
    
    # remove background
    min_val = [30, 0, 100] # hsv value
    max_val = [65, 255, 255] # hsv value
    thresh_mask = hsv_threshold(img, min_val, max_val)
    if verbosity>1:
        cv2.imshow("Thresholded mask",cv2.resize(thresh_mask*255,(0,0),fx=0.5,fy=0.5))
    
    # remove noise from the thresholded mask
    opened_mask = opening_morph(thresh_mask, kernel_size = 5)
    if verbosity>1:
        cv2.imshow("Mask after opening operation",cv2.resize(opened_mask*255,(0,0),fx=0.5,fy=0.5))
    
    # find contours of tassel blobs
    contours, hierarchy = cv2.findContours(image=opened_mask.copy(), mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)
    if verbosity>0:
        print("Identified number of crops:{}".format(len(contours)))
    
    # perform watershed segmentation to separate combined tassel blobs
    watershed_labels = watershed_segmentation(opened_mask*255)
    
    # loop over the unique labels returned by the Watershed algorithm
    new_contours= []
    for label in np.unique(watershed_labels):
     	# if the label is zero, we are examining the 'background'
     	# so simply ignore it
     	if label == 0:
             continue
     	# otherwise, allocate memory for the label region and draw
     	# it on the mask
     	mask = np.zeros(img.shape[:2], dtype="uint8")
     	mask[watershed_labels == label] = 255
     	# detect contours in the mask and grab the largest one
     	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    		cv2.CHAIN_APPROX_SIMPLE)[0]
     	new_contours.extend(cnts)

    # display all contours
    if verbosity>1:
        cnt_img = cv2.drawContours(img.copy(),new_contours,-1,(0,0,255),1)
        cv2.imshow("Found contours", cnt_img)
    
    # find filtered contours
    filtered_cnts = filter_contours(new_contours, min_area=5)
    
    # display filtered contours
    if verbosity>1:
        cnt_img = cv2.drawContours(img.copy(),filtered_cnts,-1,(0,0,255),1)
        cv2.imshow("Filtered contours",cnt_img)
    if verbosity>0:
        print('number of corn plants identified using watershed: ', len(filtered_cnts))
    
    # find center of mass of contours
    moments = map(cv2.moments,filtered_cnts)
    centers = np.array(list(map(center_of_mass,moments)))
    if verbosity>1:
        plot_centers(centers, img)
    
    # create txt files for corresponding images and include centers' coordinates
    annotation_path = ".".join(img_path.split(".")[:-1])+".txt"
    write_centers_to_file(centers, annotation_path)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image")
    parser.add_argument("-v","--verbose", action="count", help="verbosity level")
    args = parser.parse_args()
    
    main(parser.image, parser.verbose)
    


    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    