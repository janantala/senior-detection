import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.join(os.getcwd(), 'dataset')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_senior_dir = os.path.join(train_dir, 'senior')  # directory with our training senior pictures
train_junior_dir = os.path.join(train_dir, 'junior')  # directory with our training junior pictures
validation_senior_dir = os.path.join(validation_dir, 'senior')  # directory with our validation senior pictures
validation_junior_dir = os.path.join(validation_dir, 'junior')  # directory with our validation junior pictures

num_senior_tr = len(os.listdir(train_senior_dir))
num_junior_tr = len(os.listdir(train_junior_dir))

num_senior_val = len(os.listdir(validation_senior_dir))
num_junior_val = len(os.listdir(validation_junior_dir))

total_train = num_senior_tr + num_junior_tr
total_val = num_senior_val + num_junior_val

print('total training senior images:', num_senior_tr)
print('total training junior images:', num_junior_tr)

print('total validation senior images:', num_senior_val)
print('total validation junior images:', num_junior_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 128
epochs = 120
IMG_HEIGHT = 150
IMG_WIDTH = 150

NUM_CLASSES = 2

# Data preparation

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
# Here, you applied rescale, 45 degree rotation, width shift, height shift, horizontal flip and zoom augmentation to the training images

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

# Create the model

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Save checkpoints during training

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model

history = model.fit(
    train_data_gen,
    steps_per_epoch=(total_train / batch_size),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=(total_val / batch_size),
    callbacks=[cp_callback]
)

# Save model

model.save('my_model.h5') 

# Visualize training results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

