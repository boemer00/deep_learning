from __future__ import print_function, division
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Image configuration
IMAGE_SIZE = [100, 100]
epochs = 5
batch_size = 32

# Paths
train_path = '../large_files/fruits-360/Training'
valid_path = '../large_files/fruits-360/Validation'
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# Display a random image
sample_image = np.random.choice(image_files)
plt.imshow(image.img_to_array(image.load_img(sample_image)).astype('uint8'))
plt.show()

# Model building
res = ResNet50(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False
)

for layer in res.layers:
    layer.trainable = False

x = Flatten()(res.output)
prediction = Dense(len(glob(train_path + '/*')), activation='softmax')(x)
model = Model(inputs=res.input, outputs=prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
model.summary()

# Data augmentation
gen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.1,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

# Data generators
train_generator = gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)

valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)

# Training
r = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files) // batch_size,
    validation_steps=len(valid_image_files) // batch_size
)

# Confusion matrix
labels = [None] * len(train_generator.class_indices)
for k, v in train_generator.class_indices.items():
    labels[v] = k

def get_confusion_matrix(data_path, N):
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break
    return confusion_matrix(targets, predictions)

cm = get_confusion_matrix(train_path, len(image_files))
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))

# Plotting results
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

from util import plot_confusion_matrix
plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')
