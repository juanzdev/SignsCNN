import cv2
import glob
import uuid
import os
import shutil
from shutil import copyfile
import numpy as np

IMAGE_WIDTH = 52
IMAGE_HEIGHT = 52

input_folder_foreground = "foregrounds"
input_folder_background = "backgrounds"
output_folder ="output"
output_folder_augmented ="output_augmented"
classes = ['A','G','V']
#BGR
maxGreen = 245
minBlue = 50
minRed = 50


#purge data
shutil.rmtree(output_folder)
shutil.rmtree(output_folder_augmented)

#create class folders in each dir
for cls in classes:
	os.makedirs(output_folder+"/"+cls)
	os.makedirs(output_folder_augmented+"/"+cls)


generate_random_filename = 1
print("TRAINING GENERATOR...")

input_data_set_background = [img for img in glob.glob(input_folder_background+"/"+"*jpg")]
for cls in classes:
	input_data_set_foreground = [img for img in glob.glob(input_folder_foreground+"/"+cls+"/"+"*jpg")]
	print(input_data_set_foreground)
	#foregrounds
	for in_idx, img_path in enumerate(input_data_set_foreground):
		file_name = os.path.splitext(os.path.basename(img_path))[0]
		print(file_name)
		#foreground , green
		
		#iterate all backgrounds
		for in_jdx, img_path_background in enumerate(input_data_set_background):
			original_img_foreground = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
			#make copy
			img_foreground = original_img_foreground[:]
			black = img_foreground[0, 0, 0]
			green = img_foreground[0, 0, 1]
			red = img_foreground[0, 0, 2]
			guid = uuid.uuid4()
			uid_str = guid.urn
			str_guid = uid_str[9:]
			path = output_folder+"/"+cls+"/"+"_"+str_guid+".jpg"
			print(path)
			print(img_path_background)
			img_background = cv2.imread(img_path_background, cv2.IMREAD_UNCHANGED)
			backgroundColor = [black,green,red]
			height, width = img_foreground.shape[:2]
			resizeBack = cv2.resize(img_background, (width, height), interpolation = cv2.INTER_CUBIC)
			for i in range(width):
				for j in range(height):
					pixel = img_foreground[j, i]
        			#print(pixel[1])
					if np.all(pixel[1] > maxGreen):
        			# and pixel[0]<minBlue and pixel[2]<minRed):#
						img_foreground[j, i] = resizeBack[j, i]
			cv2.imwrite(path,img_foreground)

