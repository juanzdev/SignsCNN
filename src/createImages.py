import numpy as np
import sys
import glob
import uuid
import cv2
from handDetection import HandDetection
import os
import shutil
import math
import threading
import Tkinter as tk
from Tkinter import *
from PIL import Image, ImageTk
from hand_cnn import HandCnn

FRAMEWIDTH=900
FRAMEHEIGHT=800
img_size = 28

output_folder = "../original_data"
target_train_label="G"
classes = ['A','G','V']

cap = cv2.VideoCapture(0)
cap.set(3,FRAMEWIDTH)
cap.set(4,FRAMEHEIGHT)

frame = None
copy_frame_hand = None
hand_cnn_instance = HandCnn()

def hand_letter(pred):
    value = pred[0]
    if value == 0:
        return "A"
    elif value==1:
        return "G"
    elif value==2:
        return "V"
    elif value==3:
        return "G"
    elif value==4:
        return "V"

def callbackYES(self,event=None):
	guid = uuid.uuid4()
	uid_str = guid.urn
	str_guid = uid_str[9:]
	output_path = output_folder+"/"+target_train_label+"/"+str_guid+".jpg"
	cv2.imwrite(output_path,copy_frame_hand)
	print("SAVED")

root = tk.Tk()
root.bind('y', callbackYES)
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()

def show_frame():
    _, frame = cap.read()
    global copy_frame_hand
    copy_frame = frame.copy()
    cv2.rectangle(copy_frame, (20,17), (325,345), (25,25,25),3,1)
    imgray = cv2.cvtColor(copy_frame,cv2.COLOR_BGR2GRAY)
    sub_image = copy_frame[20:350, 25:320]
    sub_image_gray = imgray[20:350, 25:320]
    copy_frame_hand=sub_image

    equ = cv2.equalizeHist(sub_image_gray)
    equ_resize = cv2.resize(equ,(img_size,img_size))
    predictions,yconv,softmax = hand_cnn_instance.predict_single(equ_resize)
    softmax_pred = softmax[0]
    #state vars
    #top right
    info_width = 170
    info_height = 200
    x1info_area = FRAMEWIDTH-info_width
    y1info_area = 300 #padding top
    x2info_area = FRAMEWIDTH
    y2info_area = y1info_area+info_height
    intermargin = 20
    copy_frame = cv2.flip(copy_frame, 1)
    cv2.rectangle(copy_frame, (x1info_area,y1info_area), (x2info_area,y2info_area), (25,25,25),-1,1)
    
    for i in range(0,len(softmax_pred)):
        print(softmax_pred[i])
        st= int(round(float(softmax_pred[i])*100))
        print(st)     
        cv2.rectangle(copy_frame, (x1info_area+30,y1info_area+i*intermargin), (x1info_area+30+st,y1info_area+i*intermargin+10), (255,255,255),-1,1)
        cv2.putText(copy_frame, str(st)+"%",(x2info_area-30,y1info_area+10+i*intermargin),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),0,2)
        cv2.putText(copy_frame, classes[i],(x1info_area+0,y1info_area+10+i*intermargin),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),0,2)
    
    cv2image = cv2.cvtColor(copy_frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()

