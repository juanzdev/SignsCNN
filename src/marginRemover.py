from handDetection import HandDetection
import cv2
from util import shear_image
from util import rotate_image
from util import image_rotated_cropped
from util import mirror_img
import glob
import uuid
import os
import shutil
from shutil import copyfile
import numpy as np

margin_left_size_to_remove = 1
input_folder = "original_data/V"
output_folder_handregion ="original_data/V_"
input_data_set = [img for img in glob.glob(input_folder+"/"+"*jpg")]
for in_idx, img_path in enumerate(input_data_set):
	file_name = os.path.splitext(os.path.basename(img_path))[0]
	print(img_path)
	print("filename")
	print(file_name)
	path = output_folder_handregion+"/"+file_name+".jpg"
	print(path)
	img = cv2.imread(img_path)
	print(img.shape)
	sub_image = img[0:52, margin_left_size_to_remove:52]
	print(sub_image)
	resized_image = cv2.resize(sub_image, (52, 52)) 
	cv2.imwrite(path,resized_image)