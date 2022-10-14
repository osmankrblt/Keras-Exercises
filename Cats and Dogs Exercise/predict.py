from keras.applications.vgg16 import preprocess_input
import numpy as np


def  predictImage(image,model):

    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
   
    result = model.predict(np.array(image))
   
    result = np.argmax(result)

    return   "Cat" if result == 0 else "Dog"