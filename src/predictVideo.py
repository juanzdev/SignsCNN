import numpy as np
import sys
import glob
import uuid
import cv2
from handDetection import HandDetection
import os
import shutil
import math
from hand_cnn import HandCnn
import Tkinter as tk
from Tkinter import *
from PIL import Image, ImageTk
import model
import numpy as np
import tensorflow as tf

img_size = 28
classes = ['G','V']
classes2 = ['G','V']
def hand_letter(pred):
	value = pred[0]
	if value == 0:
		return "G"
	elif value==1:
		return "V"
	elif value==2:
		return "V"
	elif value==3:
		return "G"
	elif value==4:
		return "V"


FRAMEWIDTH=600
FRAMEHEIGHT=600

copy_frame = None
frame = None

cap = cv2.VideoCapture(0)
cap.set(3,FRAMEWIDTH)
cap.set(4,FRAMEHEIGHT)
#hand_cnn_instance = HandCnn()

num_channels = 1
img_size = 28
img_size_flat = img_size * img_size * num_channels

root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
sess = tf.Session()
# restore trained data
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    print(x)
    y,y_conv,y_conv_cls,variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "../snapshots/snp_4770")

def convolutional(input):
    return sess.run(y, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


def show_frame():
    _, frame = cap.read()
    copy_frame = frame.copy()
    cv2.rectangle(copy_frame, (20,17), (300,455), (25,25,25),3,1)
    handRegionImg = None
    if(handRegionImg==None):
		imgray = cv2.cvtColor(copy_frame,cv2.COLOR_BGR2GRAY)
		sub_image = imgray[20:450, 25:295]
		equ = cv2.equalizeHist(sub_image)
		equ_resize = cv2.resize(equ,(img_size,img_size))
		cv2.imshow("region to process", equ_resize)
		images = []
		#print(equ_resize.shape)
		images.append(equ_resize)
		images = np.array(images)
		train_batch_size = 1
		img_size_flat = img_size * img_size * num_channels
		#print(img_size_flat)
		x_batch = images;
		x_batch = x_batch.reshape(train_batch_size, img_size_flat)
		output2 = convolutional(x_batch)
		#substract_output = np.zeros(2)
		#substract_output[0] = output2[1]
		#substract_output[1] = output2[2]
        #predictions,yconv,softmax = hand_cnn_instance.predict_single(equ_resize)
		print(output2)
		#print(softmax)
		softmax_pred = output2
		#cv2.putText(copy_frame, hand_letter(predictions),(100,100),cv2.FONT_HERSHEY_PLAIN,2.2,(0,0,0),0,2)
		
		#state vars
		#top right
		info_width = 170
		info_height = 200
		x1info_area = FRAMEWIDTH-info_width
		y1info_area = 50 #padding top
		x2info_area = FRAMEWIDTH
		y2info_area = y1info_area+info_height
		intermargin = 20
		cv2.rectangle(copy_frame, (x1info_area,y1info_area), (x2info_area,y2info_area), (25,25,25),-1,1)
		
		for i in range(0,len(softmax_pred)):
			print(softmax_pred[i])
			st= int(round(float(softmax_pred[i])*100))
			#print(st)
			cv2.rectangle(copy_frame, (x1info_area+30,y1info_area+i*intermargin), (x1info_area+30+st,y1info_area+i*intermargin+10), (255,255,255),-1,1)
			cv2.putText(copy_frame, str(st)+"%",(x2info_area-30,y1info_area+10+i*intermargin),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),0,2)
			cv2.putText(copy_frame, classes2[i],(x1info_area+0,y1info_area+10+i*intermargin),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),0,2)

    frame = cv2.flip(frame, 2)
    cv2image = cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()