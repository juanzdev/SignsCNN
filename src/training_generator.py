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


minSkin = np.array([2,50,50])
maxSkin = np.array([15,255,255])
img_size = 28

input_folder = "../original_data"
output_folder_handregion ="../original_data_handregion"
output_folder_augmented ="../original_data_augmented"
output_folder_training ="../training_data"
classes = ['A','G','V']

#purge data
shutil.rmtree(output_folder_handregion)
shutil.rmtree(output_folder_augmented)
shutil.rmtree(output_folder_training)

#create class folders in each dir
for cls in classes:
	os.makedirs(output_folder_handregion+"/"+cls)
	os.makedirs(output_folder_augmented+"/"+cls)
	os.makedirs(output_folder_training+"/"+cls)
	print("myclasses")
	print(cls)


generate_random_filename = 1

for cls in classes:
	input_data_set = [img for img in glob.glob(input_folder+"/"+cls+"/"+"*jpg")]
	print(input_data_set)
	print(input_folder)
	for in_idx, img_path in enumerate(input_data_set):
		file_name = os.path.splitext(os.path.basename(img_path))[0]
		print(file_name)
		path = output_folder_handregion+"/"+cls+"/"+file_name+".jpg"
		img = cv2.imread(img_path)
		print(img.shape)
		imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		equ_resize = cv2.resize(imgray,(img_size,img_size))
		print(path)
		cv2.imwrite(path,equ_resize)

print("PLEASE MANUALLY FILTER WRONG CLASSIFICATIONS, press any key to continue with mirror and augmentation...")
raw_input('Press enter to continue: ')

#DATA AUGMENTATION MIRROR
for cls in classes:
	input_data_set = [img for img in glob.glob(output_folder_handregion+"/"+cls+"/"+"*jpg")]
	for in_idx, img_path in enumerate(input_data_set):
		file_name = os.path.splitext(os.path.basename(img_path))[0]
		print(file_name)
		#save original too
		path = output_folder_handregion+"/"+cls+"/"+file_name+"_mirrored.jpg"
		img_mirrored = mirror_img(img_path)
		cv2.imwrite(path,img_mirrored)

#DATA AUGMENTATION ROTATION
for cls in classes:
	input_data_set = [img for img in glob.glob(output_folder_handregion+"/"+cls+"/"+"*jpg")]
	for in_idx, img_path in enumerate(input_data_set):
		file_name = os.path.splitext(os.path.basename(img_path))[0]
		print(file_name)
		augmentation_number = 4
		initial_rot = -20
		#save original too
		path = output_folder_augmented+"/"+cls+"/"+file_name+".jpg"
		copyfile(img_path, path)
		for x in range(1, augmentation_number):
			rotation_coeficient = x
			rotation_step=5
			total_rotation=initial_rot+rotation_step*rotation_coeficient
			mouth_rotated = image_rotated_cropped(img_path,total_rotation)
			mouth_rotated = cv2.resize(mouth_rotated, (img_size, img_size), interpolation = cv2.INTER_CUBIC)
			if generate_random_filename == 1:
				guid = uuid.uuid4()
				uid_str = guid.urn
				str_guid = uid_str[9:]
				path = ""
				if 'showingteeth' in img_path:
				    path = output_folder_augmented+"/"+str_guid+"_showingteeth.jpg"
				else:
				    path = output_folder_augmented+"/"+cls+"/"+str_guid+".jpg"
				cv2.imwrite(path,mouth_rotated)
			else:
				path = ""
				if 'showingteeth' in img_path:
				    path = output_folder_augmented+"/"+file_name+"_rotated"+str(x)+"_showingteeth.jpg"
				else:
				    path = output_folder_augmented+"/"+cls+"/"+file_name+"_rotated"+str(x)+".jpg"
				cv2.imwrite(path,mouth_rotated)

#COPY TO TRAINING FOLDER
for cls in classes:
	input_data_set = [img for img in glob.glob(output_folder_augmented+"/"+cls+"/"+"*jpg")]
	for in_idx, img_path in enumerate(input_data_set):
		file_name = os.path.splitext(os.path.basename(img_path))[0]
		print(file_name)
		#save original too
		path = output_folder_training+"/"+cls+"/"+file_name+".jpg"
		copyfile(img_path, path)