# kwame porter robinson
# kwame@kwamata.com
#
# Rough implemention of the log-polar shape context, from
# Shape Matching and Object Recognition Using Shape Contexts,
# by S Belongie, J Malik, J Puzicha (2001)
# 
# see: http://www.cs.berkeley.edu/~malik/papers/BMP-shape.pdf
# see https://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html

# todo: read in train.csv with pandas
#       grab set of keypoints 
#       generate shape context for each point
#       return as data fram
#
#       then we have a bipartite matching problem
# ?
# http://code.activestate.com/recipes/123641-hopcroft-karp-bipartite-matching/

from skimage.feature import BRIEF, corner_harris, corner_peaks
from skimage import io, measure
import pandas as pd
import numpy as np

# super simple data flow 'pipeline'
# open, normalize
path = '../data/'
fname = 'train.csv'
images = pd.read_csv(path+fname, dtype=np.float64)
images.values[:, 1:] = images.values[:, 1:]/255.0

# to peek at an image
# %pylab # exposes numpy, matplotlib into namespace
# index = 6 # index of image you want to view
# imshow(images.iloc[index,1:].reshape((28,28)))

index = 6
image = images.iloc[index, 1:].reshape((28,28))

# extract contour points from an image, include all found contours
level = 0.8
keypoints = measure.find_contours(image, level=level)
keypoints = np.concatenate(keypoints, axis=0)

for point in keypoints:
    # remove point from set of keypoints
    pts = keypoints - point

    rho = np.sqrt(pts[:,0]**2+pts[:,1]**2)
    # rho = rho/rho.max() for scale invariance?
    rho_bins = np.logspace(0, np.log(rho.max()+1), num=6, base=np.e) # diagram looks to have 6
    rho_bins[0] = 0

    theta = np.arctan2(pts[:,0], pts[:,1])
    theta_bins = np.linspace(-np.pi, np.pi, num=12)

    # could sort along theta for rotational invariance but I think
    # the learner can learn representations ... there are only 6 angles to chose from
    context_hist = np.histogram2d(rho, theta, [rho_bins, theta_bins], normed=False)
    print(context_hist)
