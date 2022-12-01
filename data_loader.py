from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(height, width, batch_size, train_set_dir, test_set_dir):
    '''
      Function that loads the dataset using image data generator from directory
      Args:
        height (int): Required height of the image
        width (int): Required width of the image
        batch_size (int): Required batch size for training
        train_set_dir (str): The path for training set
        test_set_dir (str): The path for testing set
      Returns the train, validation, testing generator objects
    '''
    train_datagen = ImageDataGenerator(rescale=1./255, 
                                       shear_range=0.2, 
                                       zoom_range=0.2, 
                                       horizontal_flip=True, 
                                       validation_split=0.1)
    
    if train_set_dir!=None:
        train_generator = train_datagen.flow_from_directory(train_set_dir, 
                                                            target_size=(height, width), 
                                                            batch_size=batch_size, 
                                                            shuffle=True, 
                                                            class_mode='categorical', 
                                                            subset='training') # set as training data

        validation_generator = train_datagen.flow_from_directory(train_set_dir, 
                                                                 target_size=(height, width), 
                                                                 batch_size=batch_size, 
                                                                 shuffle=True, 
                                                                 class_mode='categorical', 
                                                                 subset='validation') 

        test_generator = train_datagen.flow_from_directory(test_set_dir, 
                                                           target_size=(height, width), 
                                                           batch_size=batch_size, 
                                                           shuffle=True, 
                                                           class_mode='categorical') 

        return train_generator, validation_generator, test_generator
    
    else:
        test_generator = train_datagen.flow_from_directory(test_set_dir, 
                                                           target_size=(height, width), 
                                                           batch_size=batch_size, 
                                                           shuffle=True, 
                                                           class_mode='categorical') 
        
        return test_generator