import cv2
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default=0)
parser.add_argument("--image1", type=str, default='fmdata/myleft.jpg')
parser.add_argument("--image2", type=str, default='fmdata/myright.jpg')
# parser.add_argument("--image1", type=str, default='fmdata/old_building_1.jpg')
# parser.add_argument("--image2", type=str, default='fmdata/old_building_2.jpg')
# parser.add_argument("--image1", type=str, default='fmdata/rushmore_1.jpg')
# parser.add_argument("--image2", type=str, default='fmdata/rushmore_2.jpg')
args = parser.parse_args()

print(args)


# function to normalize the points
# Normalization as specified in Hartley's Normalized Eight-Point Algorithm"
def normalizePoints(pts):
    num_pts = np.shape(pts)[0]
    result = np.hstack((pts, np.ones((num_pts, 1))))
    # shifting the coords to the centroid: (1/sqrt(2))* difference between points & mean val/total num points (std dev)
    pts_shift = np.sqrt(2) / np.std(pts)
    # constructing the normalization matrix norm_mat row by row
    row1 = [pts_shift, 0, -pts_shift * np.mean(pts[:, 0])]
    row2 = [0, pts_shift, -pts_shift * np.mean(pts[:, 1])]
    row3 = [0, 0, 1]
    norm_mat = np.vstack((row1, row2, row3))
    result = np.matmul(norm_mat, result.T).T
    return result, norm_mat


def FM_by_normalized_8_point(pts1, pts2):
    # F, _ = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_8POINT )
    # num of pts
    n = np.shape(pts1)[0]
    # Normalizing the point coordinates
    norm_pts1, norm_mat1 = normalizePoints(pts1)
    norm_pts2, norm_mat2 = normalizePoints(pts2)

    # Initialize the N x 9 matrix A
    A = np.zeros((n, 9))
    # building the matrix equation
    for i in range(n):
        # coefficients in the equation
        x = norm_pts1[i, 0]
        x_prime = norm_pts2[i, 0]
        y = norm_pts1[i, 1]
        y_prime = norm_pts2[i, 1]
        A[i] = [x * x_prime,
                x * y_prime,
                x,
                y * x_prime,
                y * y_prime,
                y,
                x_prime,
                y_prime,
                1]
    ua, sa, va = np.linalg.svd(A)
    E = va[-1].reshape(3, 3)
    ue, se, ve = np.linalg.svd(E)
    # Enforce Rank 2
    se[2] = 0
    # Calculating Fundamental Matrix F
    F = np.matmul(ue, np.matmul(np.diag(se), ve))
    # Un-Normalizing F
    F = np.matmul(norm_mat1.T, np.matmul(F, norm_mat2))
    F = F.T
    # to check with built-in implementation
    # F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    return F


def FM_by_RANSAC(pts1, pts2):
    # F:  fundmental matrix
    # mask:   whetheter the points are inliers
    # np.random.seed(39)
    # total number of inliers
    inlier_total = 0
    # num of pairs of points
    num_pairs = 8
    total_points = np.shape(pts1)[0]
    threshold = 0.05
    # number of iterations for ransac
    N = 2000
    # fundamental matrix
    F = np.zeros((3, 3))
    # initialize inlier mask
    mask = np.zeros(total_points)
    for i in np.arange(N):
        rand_pts = np.random.permutation(total_points)

        # two sets of points x and x'
        x = np.zeros((num_pairs, 2))
        x_prime = np.zeros((num_pairs, 2))

        # inlier count
        inlier_mask = np.zeros(total_points)
        num_of_inliers = 0

        # assigning 8 point pairs
        for m in np.arange(0, num_pairs):
            x[m] = pts1[rand_pts[m]]
            x_prime[m] = pts2[rand_pts[m]]

        # get fundamental matrix using these 8 pairs with the normalized 8pt algorithm above
        F_ransac = FM_by_normalized_8_point(x, x_prime)
        F_ransac = F_ransac.T

        # adding 1s to maintain dimensionality
        x = np.hstack((pts1, np.ones((total_points, 1))))
        x_prime = np.hstack((pts2, np.ones((total_points, 1))))

        # error calculations, x_prime.T * F * x = 0
        err = np.multiply(x_prime.T, np.matmul(F_ransac, x.T))
        # find absolute value of errors
        errors = np.fabs(np.sum(err, axis=0))

        for k in np.arange(total_points):
            if errors[k] < threshold:
                inlier_mask[k] = 1
                num_of_inliers += 1

        if num_of_inliers > inlier_total:
            inlier_total = num_of_inliers
            F = F_ransac
            mask = inlier_mask

    # to check with built in implementation
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F, mask


img1 = cv2.imread(args.image1, 0)
img2 = cv2.imread(args.image2, 0)

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F = None
if args.UseRANSAC:
    F, mask = FM_by_RANSAC(pts1, pts2)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
else:
    F = FM_by_normalized_8_point(pts1, pts2)


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img6)
plt.show()

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
plt.subplot(121), plt.imshow(img4)
plt.subplot(122), plt.imshow(img3)
plt.show()
