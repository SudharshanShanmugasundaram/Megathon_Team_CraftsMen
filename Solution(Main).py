from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

classifier = Sequential()

classifier.add(Conv2D(32,(3,3),input_shape = (128,128,3),activation = 'relu'))
classifier.add(Conv2D(64,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(128,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(128,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128,activation = 'relu',))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 128,activation = 'relu',))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 4,activation = 'softmax',))
classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',target_size=(128, 128),batch_size=8,class_mode='categorical')

test_set = test_datagen.flow_from_directory("data/test",
                                            target_size=(128, 128),
                                            batch_size=8,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                        steps_per_epoch=144,
                        epochs=1,
                        validation_data=test_set,
                        validation_steps=20)

classifier.save('softmax.h5')

from keras.models import load_model
m = load_model('softmax.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('data/single_prediction/bottle.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = m.predict(test_image)
#print(training_set.class_indices)
print(result)
if(max(result[0][1],result[0][2],result[0][3])<0.5):
    print("Not a breeding area")
else:
    print("It is a breeding area")