import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import keras
from sklearn import preprocessing

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics.pairwise import cosine_similarity

path = Path("path_to_file")
df = pd.read_excel(path,encoding='utf-8')

x = df['text'].astype(str).tolist()
y = df['label'].astype(str).tolist()

le = preprocessing.LabelEncoder()
le.fit(y)

def encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

x_enc = x
y_enc = encode(le, y)


test_count = round(len(x)*0.02)

x_train = np.asarray(x_enc[test_count:])
y_train = np.asarray(y_enc[test_count:])

x_test = np.asarray(x_enc[:test_count])
y_test = np.asarray(y_enc[:test_count])

#create class weight dict
class_weights = compute_class_weight('balanced', np.unique(decode(le,y_train)), decode(le,y_train))
class_weights_d = dict(enumerate(class_weights))

module_url_ML = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
#module_url_ML = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"

hub_layer = hub.KerasLayer(module_url_ML, input_shape=[], dtype=tf.string, trainable=False)

d_out_rate = 0.35

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(d_out_rate))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(d_out_rate))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, 
                    class_weight = class_weights_d,
                    epochs=3,
                    batch_size=512, verbose=1)
loss, accuracy = model.evaluate(x_test, y_test)
print("Training Accuracy: {:.4f}".format(accuracy))
