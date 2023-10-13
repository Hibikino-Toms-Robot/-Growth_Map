import glob
import cv2
import numpy as np

def trimImg(img):
    """
    画像の空白部分を除去
    """
    # 画像の上側から非ゼロの行を検出
    top = 0
    while np.sum(img[top]) == 0:
        top += 1

    # 画像の下側から非ゼロの行を検出
    bottom = img.shape[0]
    while np.sum(img[bottom - 1]) == 0:
        bottom -= 1

    # 画像の左側から非ゼロの列を検出
    left = 0
    while np.sum(img[:, left]) == 0:
        left += 1

    # 画像の右側から非ゼロの列を検出
    right = img.shape[1]
    while np.sum(img[:, right - 1]) == 0:
        right -= 1

    # 空白を除去してクロップ
    return img[top:bottom, left:right]


img_list = glob.glob('./imgs3/*.jpg')
imgs = [cv2.imread(path) for path in img_list]

# Initialize an empty list to store transformation matrices
transformation_matrices = []

for i in range(len(imgs) - 1):
    # Detect key points and compute descriptors for the images
    detector = cv2.ORB_create()
    keypoints1, descriptors1 = detector.detectAndCompute(imgs[i], None)
    keypoints2, descriptors2 = detector.detectAndCompute(imgs[i + 1], None)

    # Create a brute-force matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = matcher.match(descriptors1, descriptors2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only the top N matches (you can adjust this number)
    num_matches_to_keep = 50
    matches = matches[:num_matches_to_keep]

    # Extract the matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate the homography matrix using RANSAC
    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Store the homography matrix
    transformation_matrices.append(homography_matrix)

pre_matrix = np.eye(3)
# Print the transformation matrices (if needed)
for i, matrix in enumerate(transformation_matrices):
    print(f"Transformation Matrix {i+1} to {i+2}:\n{matrix}\n")
    path = './homography/' + str(i+1) + 'to' + str(i+2) + '.txt'

    combine_matrix = np.dot(matrix, pre_matrix)
    # print(matrix)
    # print('  ')
    pre_matrix = combine_matrix
    
    H00, H01, H02 = matrix[0]
    H10, H11, H12 = matrix[1]
    H20, H21, H22 = matrix[2]
    
    f = open(path, 'a')
    f.truncate(0)
    f.write(str(H00)+','+str(H01)+','+str(H02)+','+str(H10)+','+str(H11)+','+str(H12)+','+str(H20)+','+str(H21)+','+str(H22)+'\n')

# Stitch the images using cv2.warpPerspective with the transformation matrices
result = imgs[0]
for i in range(1, len(imgs)):
    result = cv2.warpPerspective(result, transformation_matrices[i-1], (result.shape[1] + imgs[i].shape[1], result.shape[0]))
    result[0:imgs[i].shape[0], 0:imgs[i].shape[1]] = imgs[i]

# 画像をクロップ
result = trimImg(result)

# Save the final panorama
cv2.imwrite("Panorama4.jpg", result)