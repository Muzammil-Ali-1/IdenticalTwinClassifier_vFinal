#Importing the librariws
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import optimizers

#initializing the CNN
classifier = Sequential()

#adding a convolution with corresponding max pooling step
classifier.add(Convolution2D(32, (3,3), input_shape=(128,128,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding a second convolutional layer
classifier.add(Convolution2D(64, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding a third convolutional layer
classifier.add(Convolution2D(64, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattening step
classifier.add(Flatten())

#adding hidden layer(s) and output layer
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))

#compiling the CNN
#optimizer = optimizers.SGD(momentum=0.9, nesterov=True)
classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

#citting the CNN to the twin images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=8,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=8,
        class_mode='binary')


classifier.fit_generator(training_set,
         steps_per_epoch = (498/8),
         epochs = 100,
         validation_data = test_set,
         validation_steps = 2)

#Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/md_or_mz_8.jpg', target_size=(128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0);
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'MZ (me)'
else:
    prediction = 'MD (my twin)'
    
classifier.save('twin_classifier_version1_8631.h5')
classifier = load_model('twin_classifier_version1_8631.h5')

    