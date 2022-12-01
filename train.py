import os
import yaml
import argparse

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from data_loader import load_data
from models import Models 

def train(train_generator, validation_generator, test_generator,
          model, save_path, learning_rate, epochs):
    '''
      Training function for the model
      Args:
        train_generator (obj): Image Generator object for training data
        validation_generator (obj): Image Generator object for validation data
        test_generator (obj): Image Generator object for testing data
        model (obj): The randomly initialized tensorflow model
        save_path (str): Path to save model checkpoints and logs
        learning_rate (float): The learning rate for optimizer
        epochs (int): Number of epochs to train the model
    '''
    filepath = os.path.join(save_path, 'model')
    os.makedirs(filepath, exist_ok = True) 
    checkpoint_save_path = filepath + '/model{epoch:02d}-{val_loss:.2f}.h5'
    tensorboard_path = os.path.join(save_path, 'logs')
    cp = ModelCheckpoint(checkpoint_save_path, save_best_only=True, 
                         monitor = 'val_loss', mode = 'min', verbose=0)
    es = EarlyStopping(patience=10,monitor = 'val_loss')
    tb = TensorBoard(log_dir=tensorboard_path,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
    call = [cp,es,tb]
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    history = model.fit_generator(train_generator, 
                                  epochs=epochs, 
                                  validation_data=validation_generator, 
                                  verbose=1,
                                  callbacks=call)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Neural Network for Image Classification')

    parser.add_argument('--config_path', type = str, default = './config.yaml',
                        help='Model configuration file path')

    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_data= yaml.safe_load(file)

    dataset_config = config_data['dataset']
    height, width = dataset_config['height'], dataset_config['width']
    train_dir, test_dir = dataset_config['train_dir'], dataset_config['test_dir']

    training_config = config_data['training']
    batch_size = training_config['batch_size']
    epochs = training_config['epochs'] 
    learning_rate = training_config['learning_rate']

    train_generator, validation_generator, test_generator = load_data(height=height, 
                                                                      width=width, 
                                                                      batch_size=batch_size, 
                                                                      train_set_dir=train_dir, 
                                                                      test_set_dir=test_dir)

    model_config = config_data['model']
    model_obj = Models(name=model_config['name'], height=height, width=width)
    model = model_obj.select_model()

    train(train_generator=train_generator, 
          validation_generator=validation_generator, 
          test_generator=test_generator, 
          model=model, 
          save_path=model_config['save_path'], 
          learning_rate=learning_rate, 
          epochs=epochs)
