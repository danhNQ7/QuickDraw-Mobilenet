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
import pickle
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
    x = x / 255.
    x = x.reshape((len(df), size, size, 1)).astype(np.float32)
    return x
labels = {}
with open('dictLabel.pkl', 'rb') as f:
    labels= pickle.load(f)
#Load_Model
model = model_from_json(open('model.json').read(),custom_objects={'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D})
model.load_weights('my_model_15.h5')
model.summary()
print("Loaded model from disk")
pred_results = []
chunksize =10000
INPUT_DIR = 'input/'
sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv', index_col=['key_id'])
reader = pd.read_csv('input/test_simplified.csv',chunksize=chunksize)
for chunk in tqdm(reader):
    imgs = df_to_image_array(chunk)
    # print(imgs[0])
    cv2.imwrite('logimg.jpg',imgs[0])
    pred = model.predict(imgs, verbose=1) 
    top_3 =  np.argsort(-pred)[:, 0:3]  
    pred_results.append(top_3)
    print(labels[top_3[0][0]])

    # input()
print("Finished test predictions...")

startTime = time.time()

classes_path = os.listdir(INPUT_DIR + 'train_simplified/')
class_dict = {x[:-4].replace(" ", "_"):i for i, x in enumerate(classes_path)}
reverse_dict = {v: k for k, v in class_dict.items()}
pred_results = np.concatenate(pred_results)
print("Finished data prep...")
preds_df = pd.DataFrame({'first': pred_results[:,0], 'second': pred_results[:,1], 'third': pred_results[:,2]})
preds_df = preds_df.replace(reverse_dict)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']


sub['word'] = preds_df.words.values
sub.to_csv('1class_per_label_proto.csv')
sub.head()
endTime = time.time()
print(endTime - startTime)
