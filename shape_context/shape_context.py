# kwame porter robinson
# kwame@kwamata.com
#
# Implemention of the log-polar shape context, from
# Shape Matching and Object Recognition Using Shape Contexts,
# by S Belongie, J Malik, J Puzicha (2001)
# 
# see:[1] http://www.cs.berkeley.edu/~malik/papers/BMP-shape.pdf
# see:[2] https://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html

from itertools import cycle
from multiprocessing import Pool
from sklearn.metrics.pairwise import euclidean_distances # fast
from skimage import measure
import pandas as pd
import numpy as np

def _shape_context(args):
    point, keypoints = args

    # see [1], Section 3.1 compute relative vector magnitudes ...
    pts = keypoints - point
    rho = euclidean_distances(pts, [[0,0]]) # ... relative to centered point
    rho = rho/rho.mean() # note: shape is (X, 1)

    # Do [1] Section 3.1, construct log-polar histogram
    # ... first construct the uniform in logpolar bin thresholds
    #       note: [1] does not parameters for their logpolar binning
    rho_bins = np.logspace(start= -1.5 * np.e, 
                           stop = np.log(rho.max()),
                           num  = 6,
                           base=  np.e, 
                           dtype= np.float)

    # ... similarly for theta, construct uniform linear bin thresholds
    theta = np.arctan2(pts[:,0], pts[:,1])
    theta_bins = np.linspace(-np.pi, np.pi, num=12)

    # ... histgram2d is our normed log-polar histogram [1] 3.1
    context_hist = np.histogram2d(rho.reshape(-1),
                                  theta,
                                  [rho_bins, theta_bins],
                                  normed=True)

    return context_hist[0] # just return normed 2d histogram

def shape_context(keypoints, pool_size=2):
    my_pool = Pool(processes=pool_size)
    args = zip(keypoints, cycle([keypoints])) 
    result = my_pool.map(_shape_context, args)
    return result

if __name__ == "__main__":
    # %run -i shape_context.py # to access interpreter name space

    # to peek at an image
    # %pylab # exposes numpy, matplotlib into namespace
    # index = 6 # index of image you want to view
    # imshow(images.iloc[index,1:].reshape((28,28)))
    try:
        images
    except NameError:
        # if not in namespace, load train data, normalize
        path = '../data/'
        fname = 'train.csv'
        images = pd.read_csv(path+fname, dtype=np.float64)
        images.values[:, 1:] = images.values[:, 1:]/255.0

    index = 6
    image = images.iloc[index, 1:].reshape((28,28))

    # extract contour points from an image, include all found contours
    # note: as level --> 0 then number of keypoints --> |non zero contour points|
    level = 0.8
    keypoints = measure.find_contours(image, level=level)
    keypoints = np.concatenate(keypoints, axis=0)

    print(shape_context(keypoints))
