import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
import numpy as np
import cv2
import piexif
from scipy.optimize import least_squares


def plot_pcd(pcd, title="", marker=".", s=3):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d", aspect="equal")
    plt.title(title)
    ax.scatter(
        pcd[:, 0], pcd[:, 1], pcd[:, 2], c=pcd[:, 3:], marker=marker, s=s
    )
    plt.show()


def extract_pixel_colors(img, uv):
    return np.array([img[int(uvi[1]), int(uvi[0])] for uvi in uv]) / 255


def create_point_cloud(Xs, imgs, uvs, colors=None):
    m = np.sum([x.shape[0] for x in Xs])
    n = 6
    pcd = np.empty((m, n))
    mlast = 0
    for i, X in enumerate(Xs):
        mi = X.shape[0] + mlast
        pcd[mlast:mi, :3] = X
        if colors is None:
            pcd[mlast:mi, 3:] = extract_pixel_colors(imgs[i], uvs[i])
        else:
            pcd[mlast:mi, 3:] = colors[i]
        mlast = mi
    return pcd


def triangulate(P0, P1, u1, u2):
    X = [_triangulate(P0, P1, u1[i], u2[i]) for i in range(len(u1))]
    X = np.array([(xi / xi[-1])[:3] for xi in X])
    return X


def _triangulate(P0, P1, x1, x2):
    # P0,P1: projection matrices for each of two cameras/images
    # x1,x1: corresponding points in each of two images (If using P that
    # has been scaled by K, then use camera coordinates, otherwise use
    # generalized coordinates)
    A = np.array(
        [
            [
                P0[2, 0] * x1[0] - P0[0, 0],
                P0[2, 1] * x1[0] - P0[0, 1],
                P0[2, 2] * x1[0] - P0[0, 2],
                P0[2, 3] * x1[0] - P0[0, 3],
            ],
            [
                P0[2, 0] * x1[1] - P0[1, 0],
                P0[2, 1] * x1[1] - P0[1, 1],
                P0[2, 2] * x1[1] - P0[1, 2],
                P0[2, 3] * x1[1] - P0[1, 3],
            ],
            [
                P1[2, 0] * x2[0] - P1[0, 0],
                P1[2, 1] * x2[0] - P1[0, 1],
                P1[2, 2] * x2[0] - P1[0, 2],
                P1[2, 3] * x2[0] - P1[0, 3],
            ],
            [
                P1[2, 0] * x2[1] - P1[1, 0],
                P1[2, 1] * x2[1] - P1[1, 1],
                P1[2, 2] * x2[1] - P1[1, 2],
                P1[2, 3] * x2[1] - P1[1, 3],
            ],
        ]
    )
    u, s, vt = np.linalg.svd(A)
    return vt[-1]


def get_focal_length(path, w):
    exif = piexif.load(path)
    return exif["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] / 36 * w


ImgPair = namedtuple(
    "ImgPair",
    (
        "img1",
        "img2",
        "matched_kps",
        "u1",
        "u2",
        "x1",
        "x2",
        "E",
        "K",
        "P_1",
        "P_2",
    ),
)
KeyPoint = namedtuple("KeyPoint", ("kp", "des"))
SiftImage = namedtuple("SiftImage", ("img", "kp", "des"))


def get_common_kps(pair1, pair2):
    """Return the common keypoints and indices for each pair"""
    # Use hash table instead of set to keep track of indices
    # Don't rely on dict order as that is very new feature
    hash1 = {k[1].kp: i for i, k in enumerate(pair1.matched_kps)}
    hash2 = {k[0].kp: i for i, k in enumerate(pair2.matched_kps)}

    common = []
    idx1 = []
    idx2 = []
    for k in hash1:
        if k in hash2:
            common.append(pair1.matched_kps[hash1[k]][1])
            idx1.append(hash1[k])
            idx2.append(hash2[k])
    return common, idx1, idx2


def affine_mult(P1, P2):
    res = np.vstack((P1, [0, 0, 0, 1])) @ np.vstack((P2, [0, 0, 0, 1]))
    return res[:-1]


def estimate_pose(pair1, pair2, cidx1, cidx2, X1, P3_est):
    R = P3_est[:, :-1]
    t0 = P3_est[:, -1].reshape((3, 1))
    P2 = pair1.P_2
    P2c = pair1.K @ P2

    targets = X1[cidx1]
    r0 = cv2.Rodrigues(R)[0]
    p0 = list(r0.ravel())
    p0.extend(t0.ravel())
    u1 = pair2.u1[cidx2]
    u2 = pair2.u2[cidx2]

    def residuals(p):
        R = cv2.Rodrigues(p[:3])[0]
        t = p[3:].reshape((3, 1))
        P3 = np.hstack((R, t))
        P3c = pair2.K @ P3
        Xest = triangulate(P2c, P3c, u1, u2)
        return targets.ravel() - Xest.ravel()

    res = least_squares(residuals, p0)
    p = res.x
    R = cv2.Rodrigues(p[:3])[0]
    t = p[3:].reshape((3, 1))
    P = np.hstack((R, t))
    return P


def compute_matches(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
    return good


def process_img_pair(img1, kp1, des1, img2, kp2, des2, f):
    h, w, _ = img1.shape
    good = compute_matches(des1, des2)
    u1 = []
    u2 = []
    matched_kps = []
    for m in good:
        k1 = kp1[m.queryIdx]
        d1 = des1[m.queryIdx]
        k2 = kp2[m.trainIdx]
        d2 = des2[m.trainIdx]
        matched_kps.append((KeyPoint(k1, d1), KeyPoint(k2, d2)))
        u1.append(k1.pt)
        u2.append(k2.pt)
    # u,v coords of keypoints in images
    u1 = np.array(u1)
    u2 = np.array(u2)
    # Make homogeneous
    u1 = np.c_[u1, np.ones(u1.shape[0])]
    u2 = np.c_[u2, np.ones(u2.shape[0])]

    cu = w // 2
    cv = h // 2
    # Camera matrix
    K_cam = np.array([[f, 0, cu], [0, f, cv], [0, 0, 1]])
    K_inv = np.linalg.inv(K_cam)
    # Generalized image coords
    x1 = u1 @ K_inv.T
    x2 = u2 @ K_inv.T
    # Compute essential matrix with RANSAC
    E, inliers = cv2.findEssentialMat(
        x1[:, :2], x2[:, :2], np.eye(3), method=cv2.RANSAC, threshold=1e-3
    )
    inliers = inliers.ravel().astype(bool)

    n_in, R, t, _ = cv2.recoverPose(E, x1[inliers, :2], x2[inliers, :2])
    # P_i = [R|t] with first image considered canonical
    P_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P_2 = np.hstack((R, t))
    matched_kps = [k for i, k in enumerate(matched_kps) if inliers[i]]
    return ImgPair(
        img1,
        img2,
        matched_kps,
        u1[inliers],
        u2[inliers],
        x1[inliers],
        x2[inliers],
        E,
        K_cam,
        P_1,
        P_2,
    )


def get_pt_cloud(ifnames, imgs):
    h, w, _ = imgs[0].shape
    f = get_focal_length(ifnames[0], w)
    sift = cv2.xfeatures2d.SIFT_create()
    simgs = [SiftImage(i, *sift.detectAndCompute(i, None)) for i in imgs[:3]]
    pairs = []
    for i in range(len(simgs) - 1):
        p = process_img_pair(*simgs[i], *simgs[i + 1], f)
        pairs.append(p)
    pair12, pair23 = pairs[:2]

    common_kps, idx1, idx2 = get_common_kps(*pairs[:2])
    print(f"Common Keypoints: {len(common_kps)}")

    P1c = pair12.K @ pair12.P_1
    P2c = pair12.K @ pair12.P_2
    X12 = triangulate(P1c, P2c, pair12.u1, pair12.u2)

    P3_est = affine_mult(pair12.P_2, pair23.P_2)
    P3 = estimate_pose(pair12, pair23, idx1, idx2, X12, P3_est)
    P3c = pair23.K @ P3
    X23 = triangulate(P2c, P3c, pair23.u1, pair23.u2)
    print(f"Estimate:\n{P3_est}")
    print(f"Optimized:\n{P3}")

    pcd = create_point_cloud((X12, X23), imgs[:2], (pair12.u1, pair23.u1))

    return pcd
