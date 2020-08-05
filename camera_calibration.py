import numpy as np
import cv2
import glob

output_path = "C://Users//mjw31//Desktop//data_DP//PNC_data//calibraion//"


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


front_image_extension = "camera_"
back_image_extension = ".png"

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
checkSize = (7, 5)
check_real_l = 0.022 # 2.2cm
objp = np.zeros((5 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
objp =  objp * check_real_l
# print("objp:\n", objp)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis = axis * check_real_l

# Arrays to store object points and image points from all the images    .
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

input_path = "C://Users//mjw31//Desktop//data_DP//stereo_test//output//mono_calibration_image//"
images = glob.glob(input_path + "*.png")
imgL_set = []

for idx in range(len(images)):
    inputL_name = input_path + front_image_extension + str(idx) + back_image_extension
    imgL = cv2.imread(inputL_name, 1)
    imgL_set.append(imgL)

for idx in range(len(images)):
    ret_L, cornersL = cv2.findChessboardCorners(imgL_set[idx], checkSize, None)

    if (ret_L):
        # cv2.cornerSubPix(imgL_set[idx], cornersL, (11, 11), (-1, -1),criteria)
        cv2.drawChessboardCorners(imgL_set[idx], checkSize, cornersL, ret_L)
    else:
        print("false_idx: " + idx)

    if (ret_L != 0):
        imgpoints.append(cornersL)
        objpoints.append(objp)

    # cv2.imshow('imgL', imgL_set[idx])
    # cv2.waitKey(0)

print("Starting Calibration\n")
gray = cv2.cvtColor(imgL_set[0],cv2.COLOR_BGR2GRAY)
ret, mtx, dist, R, T = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("R: \n", R)
print("T: \n", T)
print("mtx: \n", mtx)
print("dist: \n", dist)

print("Project Axis using Calibration Data\n")
for idx in range(len(images)):
    img = imgL_set[idx]
    on = True
    if on == True:
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objpoints[0], imgpoints[idx], mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, imgpoints[idx], imgpts)

        cv2.imshow('img',img)
        k = cv2.waitKey(0)

print("Save Calibration Data\n")
np.save(output_path + "k_data", mtx)
np.save(output_path + "dist_data", dist)


