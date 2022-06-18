from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator




def flip_data(image_array):

    sample_as_array = expand_dims(image_array,0)

    datagen = ImageDataGenerator(horizontal_flip=True)

    it = datagen.flow(sample_as_array, batch_size=1)

    batch = it.next()

    new_image = batch[0].astype('uint8')

    return new_image
