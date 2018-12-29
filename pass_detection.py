"""
Authors: Sarah Mallepalle and Kostas Pelechrinis

For every pass chart image in 'Cleaned_Pass_Charts', extract the locations of complete passes, 
incomplete passes, interceptions, and touchdowns relative to the line of scrimmage.
"""


import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pandas as pd
import math
import os
import scipy.misc

def map_pass_locations(centers, col, pass_type):
    """
    Function to map pixel location of passes to real field location of passes,
    with the y-axis at the center of the field, and the x-axis at the line of scrimmage.
    All images show either 55 yards or 75 yards in front of the line of scrimmage,
    10 yards behind the line of scrimmage, and a standard field width of 53.33 yards.
    
    Input:
        centers: list of pass locations in pixels
        col: width of image from which the pass locations were extracted
        pass_type: "COMPLETE", "INCOMPLETE", "INTERCEPTION", or "TOUCHDOWN"
    Return:
        pass_locations: Pandas DataFrame of all pass locations on the field and pass type
    """
    sideline = 40 # pixels
    width = 53.33
    center_x = col/2

    if col > 1370:
        _75_yd_line = 0
        LOS = 596

        _1_yd_x = float(col - sideline*2)/width
        _1_yd_y = float(LOS - _75_yd_line)/75

    else:
        _55_yd_line = 5
        LOS = 572
        _1_yd_x = float(col - sideline*2)/width
        _1_yd_y = float(LOS - _55_yd_line)/55

    col_names = ["pass_type", "x", "y"]
    pass_locations = pd.DataFrame(columns = col_names)

    for c in centers:
        y = c[0]
        x = c[1]
        y_loc = float(LOS - y)/_1_yd_y
        x_loc = float(x - center_x)/_1_yd_x
        df = pd.DataFrame([[pass_type, x_loc, y_loc]], columns = col_names)
        pass_locations = pass_locations.append(df, ignore_index=True)
    return pass_locations

def completions(image, n):
    """
    Function to obtain the locations of the complete passes from the image
    of the pass chart using k-means.
    
    Input: 
        image: image from the folder 'Cleaned_Pass_Charts'
        n: number of incompletions, from the corresponding data of the image
    Return:
        call to map_pass_locations:
            centers: list of pass locations in pixels
            col: width of image from which the pass locations were extracted
            pass_type: "COMPLETE"
    """

    image = cv2.imread(image)
    row, col = image.shape[0:2]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_green = np.array([40,100, 100])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    x = np.where(res != 0)[0]
    y = np.where(res != 0)[1]
    pairs = zip(x,y)
    X = map(list, pairs)

    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
    centers = kmeans.cluster_centers_

    return map_pass_locations(centers, col, "COMPLETE")

def incompletions(image, n):
    """
    Function to obtain the locations of the incomplete passes from the image
    of the pass chart using k-means, and DBSCAN to account for discrepancies
    in given number of incompletions from the data vs. number of incompletions
    shown on the field.
    
    Input: 
        image: image from the folder 'Cleaned_Pass_Charts'
        n: number of incompletions, from the corresponding data of the image
    Return:
        call to map_pass_locations:
            centers: list of pass locations in pixels
            col: width of image from which the pass locations were extracted
            pass_type: "INCOMPLETE"
    """

    image = cv2.imread(image)
    row, col = image.shape[0:2]

    lower_white = np.array([230, 230, 230])
    upper_white = np.array([255, 255, 255])

    mask = cv2.inRange(image, lower_white, upper_white)
    res = cv2.bitwise_and(image, image, mask=mask)

    x = np.where(res != 0)[0]
    y = np.where(res != 0)[1]
    pairs = zip(x,y)
    X = map(list, pairs)

    db = DBSCAN(eps=10, min_samples=0).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique, counts = np.unique(labels, return_counts=True)    
    avg = np.median(counts)
    thresh = math.floor(1.4*avg)
    for count in counts:
      if (count > thresh): n_clusters_ +=1

    #print 'Actual number of clusters: %d' % n_clusters_
    #print "Given number of interceptions:", n

    kmeans = KMeans(n_clusters=min(n_clusters_, n), random_state=0).fit(X)
    centers = kmeans.cluster_centers_

    return map_pass_locations(centers, col, "INCOMPLETE")
 
def interceptions(image, n):
    """
    Function to obtain the locations of the intercepted passes from the image
    of the pass chart using k-means.
    
    Input: 
        image: image from the folder 'Cleaned_Pass_Charts'
        n: number of interceptions, from the corresponding data of the image
    Return:
        call to map_pass_locations:
            centers: list of pass locations in pixels
            col: width of image from which the pass locations were extracted
            pass_type: "INTERCEPTION"
    """

    image = cv2.imread(image)
    row, col = image.shape[0:2]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of red in HSV with two masks
    mask1 = cv2.inRange(hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(hsv, (175,50,20), (180,255,255))

    # Threshold the HSV image to get only red colors
    mask = cv2.bitwise_or(mask1, mask2)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    x = np.where(res != 0)[0]
    y = np.where(res != 0)[1]
    pairs = zip(x,y)
    X = map(list, pairs)

    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
    centers = kmeans.cluster_centers_

    return map_pass_locations(centers, col, "INTERCEPTION")

def touchdowns(image, n):
    """
    Function to obtain the locations of the touchdown passes from the image
    of the pass chart using k-means, and DBSCAN to account for difficulties in 
    extracting touchdown passes, since they have the are the same color as both the line of 
    scrimmage and the attached touchdown trajectory lines. 
    
    Input: 
        image: image from the folder 'Cleaned_Pass_Charts'
        n: number of toucndowns, from the corresponding data of the image
    Return:
        call to map_pass_locations:
            centers: list of pass locations in pixels
            col: width of image from which the pass locations were extracted
            pass_type: "TOUCHDOWN"
    """

    im = Image.open(image)
    pix = im.load()
    col, row = im.size

    img = Image.new('RGB', (col, row), 'black') 
    p = img.load() 

    for i in range(col):
        for j in range(row):
            r = pix[i,j][0]
            g = pix[i,j][1]
            b = pix[i,j][2]
            if (col < 1370) and (j < row-105) and (j > row-111):
                if (b > 2*g) and (b > 60): 
                    p[i,j] = (0,0,0)
            elif (col > 1370) and (j < row-81) and (j > row-86):
                if (b > 2*g) and (b > 60): 
                    p[i,j] = (0,0,0)
            else: p[i,j] = pix[i,j]
            r = p[i,j][0]
            g = p[i,j][1]
            b = p[i,j][2]
            f = ((r-20)**2 + (g-80)**2+ (b-200)**2)**0.5
            if f < 32 and b > 100: 
                p[i,j] = (255, 255,0)

    scipy.misc.imsave('temp.jpg', img)
    imag = cv2.imread('temp.jpg')
    os.remove('temp.jpg')
    hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)
    lower = np.array([20,100, 100])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(imag, imag, mask=mask)
    res = cv2.fastNlMeansDenoisingColored(res)
    x = np.where(res != 0)[0]
    y = np.where(res != 0)[1]
    pairs = zip(x,y)
    X = map(list, pairs)

    if (len(pairs) != 0):
        db = DBSCAN(eps=10, min_samples=n).fit(X)
        labels = db.labels_
        coords = pd.DataFrame([x, y, labels]).T
        coords.columns = ['x', 'y','label']
        clusters = Counter(labels).most_common(n)
        td_labels = np.array([clust[0] for clust in clusters])
        km_coords = coords.loc[coords['label'].isin(td_labels)]
        km = map(list, zip(km_coords.iloc[:,0], km_coords.iloc[:,1]))

        kmeans = KMeans(n_clusters=n, random_state=0).fit(km)
        centers = kmeans.cluster_centers_ 

        return map_pass_locations(centers, col, "TOUCHDOWN")
    else:
        pass_cols = ["pass_type", "x", "y"]
        rows_td = pd.DataFrame([["TOUCHDOWN", np.NaN, np.NaN]]*n, 
            columns = pass_cols)
        return rows_td


