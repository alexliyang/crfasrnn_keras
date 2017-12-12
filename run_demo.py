"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os

import keras
import cv2
import numpy as np
import tensorflow as tf

from crfrnn_model import get_crfrnn_model_def, get_crfrnn_model_def_v2
import util


def main():
    input_file = "image.jpg"
    output_file = "labels.png"

    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = "/data/download/crfrnn_keras_model.h5"

    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    img_data, img_h, img_w = util.get_preprocessed_image(input_file)
    probs = model.predict(img_data, verbose=False)[0, :, :, :]
    segmentation = util.get_label_image(probs, img_h, img_w)
    segmentation.save(output_file)


def train():
    train_img_dir = '/data/dataset/traffic_line/train'
    val_img_dir = '/data/dataset/traffic_line/verify'
    train_data = []
    train_label_data = []
    for file_name in os.listdir(train_img_dir):
        file_path = os.path.join(train_img_dir, file_name)

        if 'mask'  not in file_name:
            tmp_img = cv2.imread(file_path)
            tmp_img = cv2.resize(tmp_img, (500, 500),
                                 interpolation=cv2.INTER_CUBIC)
            tmp_img = tmp_img.astype('float32') / 255.0
            train_data.append(tmp_img)
            mask_file_path = file_path[:-4] + '_mask.jpg'
            tmp_img = cv2.imread(mask_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            tmp_img = cv2.resize(tmp_img, (500, 500),
                                 interpolation=cv2.INTER_CUBIC)
            tmp_img = tmp_img.astype('float32') / 255.0
            train_label_data.append(tmp_img)
    train_data = np.asarray(train_data)
    train_label_data = np.expand_dims(train_label_data, axis=3)
    train_label_data = np.asarray(train_label_data)

    val_data = []
    val_label_data = []
    for file_name in os.listdir(val_img_dir):
        file_path = os.path.join(val_img_dir, file_name)

        if 'mask' not in file_name:
            tmp_img = cv2.imread(file_path)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img = cv2.resize(tmp_img, (500, 500),
                                 interpolation=cv2.INTER_CUBIC)
            tmp_img = tmp_img.astype('float32') / 255.0
            val_data.append(tmp_img)
            mask_file_path = file_path[:-4] + '_mask.jpg'
            tmp_img = cv2.imread(mask_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            tmp_img = cv2.resize(tmp_img, (500, 500),
                                 interpolation=cv2.INTER_CUBIC)
            tmp_img = tmp_img.astype('float32') / 255.0
            val_label_data.append(tmp_img)
    val_data = np.asarray(val_data)
    val_label_data = np.expand_dims(val_label_data, axis=3)
    val_label_data = np.asarray(val_label_data)

    model = get_crfrnn_model_def_v2()
    model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.99),
                  loss=keras.losses.binary_crossentropy)
    tb = keras.callbacks.TensorBoard('./tb', histogram_freq=1, batch_size=4,
                                     write_grads=True)
    save_model = keras.callbacks.ModelCheckpoint(
        './model/ckp_{epoch}_{val_loss:.4f}.hdf5', monitor='val_loss', verbose=0,
        save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit(x=train_data, y=train_label_data, batch_size=4, epochs=10000,
              callbacks=[tb, save_model],
              validation_data=(val_data, val_label_data))


if __name__ == "__main__":
    train()
