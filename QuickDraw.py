import tkinter as tk
from tkinter import  *
import cv2
import numpy as np 
import time
from glob import glob
import os
import re
import ast
import cv2
import csv
import time
import ast
import urllib
from PIL import Image, ImageDraw
from tqdm import tqdm
from dask import bag, threaded
import matplotlib
import matplotlib.pyplot as pltc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from dask import bag, threaded
import keras
import pickle 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
#from keras.applications.nasnet import NASNetMobile
from keras.preprocessing import image
from keras.models import Model,model_from_json,load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications import MobileNet

BASE_SIZE = 256
size =80
def draw_cv2(raw_strokes, size=256, lw=6):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for stroke in raw_strokes:
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), 255, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img
def df_to_image_array(df, size=size, lw=6):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
        cv2.imwrite('result{}.png'.format(i),x[i])
    x = x / 255.
    x = x.reshape((len(df), size, size, 1)).astype(np.float32)
    return x
#load label 
def format_img(img,size=size):
    img = cv2.resize(img,(size,size))
    img = img/255.
    img = img.reshape((1, size, size, 1)).astype(np.float32)
    return img
with open('dictLabel.pkl', 'rb') as f:
    labels= pickle.load(f)
reverse_dict = {v: k for k, v in labels.items()}
#Load_Model
model = model_from_json(open('model.json').read(),custom_objects={'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D})
model.load_weights('my_model_shuffe_large_15.h5')
model.summary()
print("Loaded model from disk")
pred_results = []
# chunksize =10000
# INPUT_DIR = 'input/'
# sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv', index_col=['key_id'])
# reader = pd.read_csv('input/test_simplified.csv',nrows=100)#chunksize=chunksize)
# #for chunk in reader.iloc(:,::
# chunk = reader.iloc[4:5,]
# print(chunk)

# imgs = df_to_image_array(chunk)
# # print(imgs[0])
# cv2.imwrite('logimg.jpg',imgs[0])
# pred = model.predict(imgs, verbose=1) 
# top_3 =  np.argsort(-pred)[:, 0:3]  
# pred_results.append(top_3)
# # print(labels[top_3[0][0]])
# print(top_3)
# print(reverse_dict[top_3[0,0]])
    # input()
print("Finished test predictions...")
width =600
height =600
center = height//2
white = (255, 255, 255)
green = (0,128,0)
def mmove(event):
    print(event.x, event.y)
def clear():
    global txt
    cv.delete('all')
    txt= cv.create_text(300,550,fill="darkblue",font="Times 14 italic bold",text="Predict ")
    global img,pas,windows
    img = np.zeros((600, 600), np.uint8)
    pas = np.zeros(1)
    windows = np.array([600,600,0,0])
a = None
def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    if x1 < windows[0]: windows[0]=x1
    if y1 < windows[1]: windows[1]=y1
    if x2 > windows[2]: windows[2]=x2
    if y2 > windows[3]: windows[3]=y2 
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    # draw.line([x1, y1, x2, y2],fill="black",width=5)
# pas = pas +1 
# if pas %100 ==0:
    cv2.line(img, (x1, y1), (x2, y2), 255, lw)
    pas[0] = pas[0]+1
    if pas[0]%20 ==0: 
        print('ahihi')
        frame = format_img(img[windows[1]:windows[3],windows[0]:windows[2]],size=size)
        pred = model.predict(frame, verbose=1) 
        top_3 =  np.argsort(-pred)[:, 0:3]  
        # pred_results.append(top_3)
        print(reverse_dict[top_3[0][0]])
        cv.itemconfig(txt, text="Predict {}".format(reverse_dict[top_3[0][0]]))
        # img2 = cv2.bitwise_not(img)

        # _,contours,hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[1]
        # print(len(contours))
        # x,y,w,h = cv2.boundingRect(cnt)
        # # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.imwrite('imgraw.png',img)
        cv2.imwrite('test.png',img[windows[1]:windows[3],windows[0]:windows[2]])
root = tk.Tk()

print(labels)
# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()
img = np.zeros((600, 600), np.uint8)
lw=6
windows = np.zeros((2,2))
pas = np.zeros(1)
windows = np.array([600,600,0,0])
cv.pack(expand=YES, fill=BOTH)
txt= cv.create_text(300,550,fill="darkblue",font="Times 14 italic bold",text="Predict ")

button=Button(text="clear",command=clear)
button.pack()
cv.bind("<B1-Motion>", paint)
root.mainloop()