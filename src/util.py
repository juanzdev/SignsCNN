import cv2
from skimage import io
from skimage import transform as tf
import numpy as np

def transform_img(img, img_width, img_height):
      #Image Resizing
      img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
      return img

def histogram_equalization(img):
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img

def mirror_img(imgpath):
    image = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    horizontal_img = cv2.flip(image, 1 )
    return horizontal_img

def shear_image(img_path,level=0.2):
	image = io.imread(img_path)
	print(image)
	afine_tf = tf.AffineTransform(shear=level)
	print(afine_tf)
	modified = tf.warp(image, afine_tf)
	modified = modified*255
	return modified

def rotate_image(image_path, angle):
  image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result

import math
import cv2
import numpy as np


def rotate_image(image, angle):
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def image_rotated_cropped(image_path,angle):
    image = cv2.imread(image_path)
    if image is not None:
	    image_height, image_width = image.shape[0:2]
	    image_orig = np.copy(image)
	    image_rotated = rotate_image(image, angle)
	    image_rotated_cropped = crop_around_center(
	        image_rotated,
	        *largest_rotated_rect(
	            image_width,
	            image_height,
	            math.radians(angle)
	        )
	    )
	    return image_rotated_cropped