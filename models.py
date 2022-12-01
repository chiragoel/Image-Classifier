from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,Dropout




class Models:
    def __init__(self, name, height=224, width=224, weights='imagenet'):
        self.name = name
        self.shape = (height, width, 3)
        self.weights = weights
        
    def _vgg_model(self):
        
        model = VGG16(weights=self.weights, include_top=False, input_shape=self.shape)
        # Freeze weights
        for layer in model.layers[:15]:
            layer.trainable = False
        # Check frozen layers
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)
        
        x = model.output
        x = Flatten()(x) # Flatten dimensions to for use in FC layers
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
        x = Dense(256, activation='relu')(x)
        x = Dense(3, activation='softmax')(x) # Softmax for multiclass
        transfer_model = Model(inputs=model.input, outputs=x)
        return transfer_model


    def _resnet_model(self):
        model = ResNet50(weights=self.weights, include_top=False, input_shape=self.shape)
        # Freeze weights
        for layer in model.layers[:143]:
            layer.trainable = False
        # Check frozen layers
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)
        
        x = model.output
        x = Flatten()(x) # Flatten dimensions to for use in FC layers
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
        x = Dense(3, activation='softmax')(x) # Softmax for multiclass
        transfer_model = Model(inputs=model.input, outputs=x)
        return transfer_model

    def _incept_resnet(self):
        model = InceptionResNetV2(weights=self.weights, include_top=False, input_shape=self.shape)
        # Freeze weights
        for layer in model.layers[:762]:
            layer.trainable = False
        # Check frozen layers
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)
        
        x = model.output
        x = Flatten()(x) # Flatten dimensions to for use in FC layers
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
        x = Dense(3, activation='softmax')(x) # Softmax for multiclass
        transfer_model = Model(inputs=model.input, outputs=x)
        return transfer_model

    def _mobile_net(self):
        model = MobileNetV2(weights=self.weights, include_top=False, input_shape=self.shape)
        # Freeze weights
        for layer in model.layers[:143]:
            layer.trainable = False
        # Check frozen layers
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)
        
        x = model.output
        x = Flatten()(x) # Flatten dimensions to for use in FC layers
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
        x = Dense(256, activation='relu')(x)
        x = Dense(3, activation='softmax')(x) # Softmax for multiclass
        transfer_model = Model(inputs=model.input, outputs=x)
        return transfer_model
    
    def select_model(self):
        if self.name == 'vgg':
            model = self._vgg_model()
        elif self.name == 'resnet':
            model = self._resnet_model()
        elif self.name == 'inception_resnet':
            model = self._incept_resnet()
        elif self.name == 'mobilenet':
            model = self._mobile_net()
        else:
            raise ValueError("Not a valid model name choice. Your choices are: vgg, resnet, inception_resnet, mobilenet")
        return model
