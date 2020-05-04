import os
import pathlib

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image

model = tf.keras.models.load_model('my_model.h5')
model.summary()

print(model.inputs)

class_names = ['junior', 'senior']

IMG_HEIGHT = 150
IMG_WIDTH = 150

PATH = os.path.join(os.getcwd(), 'dataset')
test_dir = os.path.join(PATH, 'test')
test_datagen = ImageDataGenerator(rescale=1./255) # Generator for our validation data
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict(test_generator,
	steps=STEP_SIZE_TEST,
	verbose=1)

print(pred)
predicted_class_indices=np.argmax(pred,axis=1)
filenames=test_generator.filenames
print(predicted_class_indices)
print(filenames)
