import sys
import os
import numpy as np
sys.path.append('../')
from data_IO.data_reader import get_data
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
dropout_keep_prob = 0.5
class_nums = 50
LABEL_PATH = r'/home/xujingning/ocean/ocean_data/label.csv'
DATA_PATH = '/home/xujingning/ocean/ocean_data/data_img/'
MODEL_PATH = '/home/xujingning/ocean/ocean_data/VGG+weighted/'


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        print('micro:')
        print('f1: ', f1_score(val_targ, val_predict, average='micro'))
        print('precision: ', precision_score(val_targ, val_predict, average='micro'))
        print('recall: ', recall_score(val_targ, val_predict, average='micro'))
        print('macro:')
        print('f1: ', f1_score(val_targ, val_predict, average='macro'))
        print('precision: ', precision_score(val_targ, val_predict, average='macro'))
        print('recall: ', recall_score(val_targ, val_predict, average='macro'))
        return


_metrics = Metrics()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
labels, imgs, _ = get_data(DATA_PATH, LABEL_PATH)
total_imgs_train, total_imgs_test, total_labels_train, total_labels_test = train_test_split(imgs, labels, test_size=0.3, random_state=16)

total_labels_train, total_labels_test = to_categorical(total_labels_train, class_nums), to_categorical(total_labels_test, class_nums)

input_shape = (224, 224, 3)

model = Sequential(name='vgg 16-sequential')

model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block1_conv1'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block1_conv2'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block1_pool'))

model.add(Conv2D(128,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block2_conv1'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block2_conv2'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block2_pool'))

model.add(Conv2D(256,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block3_conv1'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block3_conv2'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block3_conv3'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block3_pool'))

model.add(Conv2D(512,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block4_conv1'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block4_conv2'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block4_conv3'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block4_pool'))

model.add(Conv2D(512,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block5_conv1'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block5_conv2'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block5_conv3'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block5_pool'))

model.add(Flatten(name='flatten'))
model.add(Dense(4096,activation='relu',name='fc1'))
model.add(Dropout(dropout_keep_prob))
model.add(Dense(4096,activation='relu',name='fc2'))
model.add(Dropout(dropout_keep_prob))
model.add(Dense(class_nums,activation='softmax',name='predictions'))

model.load_weights('/home/xujingning/ocean/ocean_data/VGG+weighted/my_model_weights.h5')
import keras.optimizers
opt = keras.optimizers.Adadelta()
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
for k in range(0, 10):
    H = model.fit(total_imgs_train,total_labels_train,validation_data=(total_imgs_test, total_labels_test),batch_size=64,epochs=3,verbose=2, class_weight='auto',callbacks=[_metrics])
    model.save_weights('/home/xujingning/ocean/ocean_data/VGG+weighted/my_model_weights.h5')