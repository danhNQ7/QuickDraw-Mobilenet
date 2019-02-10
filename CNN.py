from glob import glob
import re
import ast
import cv2
import csv
import time
import ast
import urllib
import json
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
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications import MobileNet, ResNet50
import pickle 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
BASE_SIZE = 256
DP_DIR = 'ShuffleData/'
INPUT_DIR = 'input/'
NCSVS = 80
NCATS = 340
import os
print(os.listdir("input"))

startTime = time.time()
#clean spaces in name
classes_path = os.listdir(INPUT_DIR + 'train_simplified/')
classes_path = sorted(classes_path, key=lambda s: s.lower())
class_dict = {x[:-4].replace(" ", "_"):i for i, x in enumerate(classes_path)}
labels = {x[:-4].replace(" ", "_") for i, x in enumerate(classes_path)}
with open('Log/dictLabel.pkl', 'wb') as f:
        pickle.dump(class_dict, f, pickle.HIGHEST_PROTOCOL)
input()
n_labels = len(labels)
print("Number of labels: {}".format(n_labels))

fileList = glob(INPUT_DIR + "train_simplified/*.csv")     

n_files = n_labels #number of csv files same as labels due to stupid structure.

#time is sacred HARDCODED FOR THE COMP
n_records = 49707919
# MobileNet
size = 80
# Resnet50
# size = 197

#for f in fileList: saving time
#    n_records += sum(1 for line in open(f))
print("Number of records: {}".format(n_records))
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
# to image from stroke
def drawing_to_np(drawing, shape=(size, size)):
    drawing = eval(drawing)
    fig, ax = plt.subplots()
    for x,y in drawing:
        ax.plot(x, y, marker='.')
        ax.axis('off')
    fig.canvas.draw()
    # Convert images to numpy arrat
    np_drawing = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    img = cv2.resize(np_drawing, shape)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_expanded = img_gray[:, :, np.newaxis]
    return img_expanded
def draw_cv2_reshape_normalized(raw_strokes, size=size, lw=6):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for stroke in raw_strokes:
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), 255, lw)

    img = cv2.resize(img, (size, size))
    img = img / 255.
    img = img[:, :, np.newaxis]
    return img
def draw_cv2(raw_strokes, size=256, lw=6):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for stroke in raw_strokes:
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), 255, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

#ADD DATA AUGMENTATION TO BOOST
def image_generator(size, batchsize, ks, lw=6):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
                x = x / 255.
                x = x.reshape((len(df), size, size, 1)).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y

def df_to_image_array(df, size=size, lw=6):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
    x = x / 255.
    x = x.reshape((len(df), size, size, 1)).astype(np.float32)
    return x
STEPS = 2000
batchsize = 256
epochs = 15
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)),nrows=40000)
x_valid = df_to_image_array(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))
train_datagen = image_generator(size=size, batchsize=batchsize, ks=range(NCSVS - 1))

# base_model = ResNet50(input_shape=(size, size, 1), include_top=False, weights=None, classes=n_labels)
base_model =  MobileNet(input_shape=(size, size, 1), include_top=False, weights=None, classes=n_labels)
#input('basemodel')
# add a global spatial average pooling layer
x = base_model.output
x = Flatten()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(n_labels, activation='softmax')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
#input('model')
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
#input('complie')
model.summary()
callbacks = [
    ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5,
                       mode='max', cooldown=3, verbose=1),
    ModelCheckpoint('model_shuffe_large_7.h5', monitor='val_top_3_accuracy', mode='max', save_best_only=True,
                    save_weights_only=True)
    
]

# hist = model.fit_generator(
#     train_datagen, steps_per_epoch=STEPS, epochs=epochs, verbose=1,
#     validation_data=(x_valid, y_valid),
#     callbacks = callbacks
# )
# STEPSS = int( np.ceil(len(train_datagen) / batchsize) )
hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=epochs, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('my_model_shuffe_large_15_resnet50.h5')
def gen_graph(history, title):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['top_3_accuracy'])
    plt.plot(history.history['val_top_3_accuracy'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation', 'Test top 3', 'Validation top 3'], loc='upper left')
    plt.show()
    plt.plot(history.history['categorical_crossentropy'])
    plt.plot(history.history['val_categorical_crossentropy'])
    plt.title('Loss ' + title)
    plt.ylabel('MLogLoss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()
    plt.savefig('result.jpg')
#plot
gen_graph(hist, 
             "Simple net lul")

pred_results = []
chunksize = 10000
reader = pd.read_csv(INPUT_DIR + 'test_simplified.csv', chunksize=chunksize)
for chunk in tqdm(reader):
    imgs = df_to_image_array(chunk)
    pred = model.predict(imgs, verbose=1)
    top_3 =  np.argsort(-pred)[:, 0:3]  
    pred_results.append(top_3)
print("Finished test predictions...")
#prepare data for saving
reverse_dict = {v: k for k, v in class_dict.items()}
pred_results = np.concatenate(pred_results)
print("Finished data prep...")
preds_df = pd.DataFrame({'first': pred_results[:,0], 'second': pred_results[:,1], 'third': pred_results[:,2]})
preds_df = preds_df.replace(reverse_dict)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']

sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv', index_col=['key_id'])
sub['word'] = preds_df.words.values
sub.to_csv('1class_per_label_proto.csv')
sub.head()
endTime = time.time()
print(endTime - startTime)
