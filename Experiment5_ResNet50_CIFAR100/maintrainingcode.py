import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D, Dropout,BatchNormalization,GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

import random

# Defining the L2L0 Regularizer, including the standard L2-norm regularizer and the proposed L0-norm-based regularizer
tf.keras.utils.get_custom_objects().clear()
@tf.keras.utils.register_keras_serializable(package='Custom', name='L2L0_Reg')
class L2L0_Reg(tf.keras.regularizers.Regularizer):

  def __init__(self, l0=0., beta=0, l1=0., l2=0.):  # pylint: disable=redefined-outer-name
    self.l0 = K.cast_to_floatx(l0)
    self.beta = K.cast_to_floatx(beta)
    self.l1 = K.cast_to_floatx(l1)
    self.l2 = K.cast_to_floatx(l2)

  def __call__(self, x):
    # ones_tensor = tf.ones(x.shape)
    # return self.l0 * math_ops.reduce_sum(ones_tensor-math_ops.exp(-self.beta*math_ops.abs(x)))
    if not self.l2 and not self.l0:
      return K.constant(0.)
    regularization = 0.
    if self.l0:
      ones_tensor = tf.ones(x.shape)
      regularization += self.l0 * math_ops.reduce_sum(ones_tensor-math_ops.exp(-self.beta*math_ops.abs(x)))
    if self.l2:
      regularization += self.l2 * math_ops.reduce_sum(math_ops.square(x))
    return regularization

  def get_config(self):
    return {'l0': float(self.l0), 'beta': float(self.beta), 'l2': float(self.l2), 'l1': float(self.l1)}

  @classmethod
  def from_config(cls, config):
      l0 = float(config.pop("l0"))
      beta = float(config.pop("beta"))
      l2 = float(config.pop("l2"))
      return cls(l0=l0,beta=beta,l2=l2)


# Helper function:
def l0_exp(l0=0.01, l1=0.1, l2=0.1, beta=10):
  return L2L0_Reg(l0=l0, beta=beta, l1=0, l2=l2)
  
  
# Time tracking function (from somewhere on the internet):
def timer(start_time=None):
  #function to track time 
  if not start_time:
      print(datetime.now())
      start_time = datetime.now()
      return start_time
  elif start_time:
      thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
      tmin, tsec = divmod(temp_sec, 60)
      print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
  

# Defining the considered beta and alphal0 hyperparameters
betas = [0.0,10.0,10.0,10.0,5.0,5.0,5.0]
alphas = [1e-7,1e-7,5e-7,1e-6,1e-7,5e-7,1e-6]

for alp,bta in zip(alphas,betas):


    # Loading the CIFAR100 dataset:
    tf.keras.utils.disable_interactive_logging() # This avoids excessive logging.
    cifar100 = tf.keras.datasets.cifar100
    (X_train, Y_train), (X_test,Y_test) = cifar100.load_data()
    tf.keras.utils.enable_interactive_logging() # Restoring logging.
    
    # Dataset preprocessing:
    y_train = to_categorical(Y_train, num_classes = 100)
    y_test = to_categorical(Y_test, num_classes = 100)
    
    # Normalizing the input data:
    x_train = X_train * 1.0/255    
    x_test = X_test * 1.0/255
    
    # Data generator for augmentation and etc:
    train_datagen = ImageDataGenerator(
            rotation_range = 10,
            zoom_range = 0.1,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            shear_range = 0.1,
            horizontal_flip = True,
            vertical_flip = False,
            featurewise_center=True,
            featurewise_std_normalization=True,
            validation_split = 0.2 )
    train_datagen.fit(x_train)

    # Generator for validation data:
    valid_datagen = ImageDataGenerator( featurewise_center=True,
                                        featurewise_std_normalization=True, 
                                        validation_split = 0.2 )
    valid_datagen.fit(x_train)
    
    # Callback for learning rate reduction:
    from keras.callbacks import ReduceLROnPlateau
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_accuracy', 
        patience=8, 
        verbose=1, 
        factor=0.5, 
        min_lr=1e-6)

    # Defining the ResNet50 model:
    tf.keras.utils.disable_interactive_logging()
    from tensorflow.keras.applications.resnet50 import ResNet50
    resnet_model = ResNet50(
        include_top = False,
        weights = None, # 'imagenet',
        input_shape = (224,224,3)
    )
    tf.keras.utils.enable_interactive_logging()
    
    # Adding some layers:
    model=tf.keras.models.Sequential()
    model.add(UpSampling2D(size=(7, 7),interpolation='bilinear'))
    model.add(resnet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.25))
    model.add(Dense(256, activation='relu',name="Dense1"))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='softmax',name="Dense2"))
    
    # Adding regularization to the model:
    # regularization = "L2"
    regularization = "L2L0"
    # regularization = None
    beta = bta
    alphal0d = alp
    alphal0c = alp
    alphal2 = 1e-4
    
    NORM = True
    
    prods = []
    
    for layer in model.submodules:
      if isinstance(layer, Dense):
        if regularization == "L2":
          layer.kernel_regularizer = tf.keras.regularizers.L2(alphal2)
        elif regularization == "L2L0":
            if NORM:
                const_l0 =  np.sqrt( ((2048*256) if (layer.name == "Dense1") else (256*100)  ) / 524288  )
                layer.kernel_regularizer = l0_exp(l0=const_l0*alphal0d, l1=0., l2=alphal2, beta=beta)
            else:
                layer.kernel_regularizer = l0_exp(l0=alphal0d, l1=0., l2=alphal2, beta=beta)
    #     print(layer.get_config())
      if isinstance(layer, Conv2D):
        layer.kernel_initializer = tf.keras.initializers.HeNormal()
        if regularization == "L2":
          layer.kernel_regularizer = tf.keras.regularizers.L2(alphal2)
        elif regularization == "L2L0":
          x = layer.get_weights()[0]
          prods.append(np.prod(x.shape[1:]))
          if NORM:
            const_l0 = np.sqrt( np.prod(x.shape[1:])/2097152 )
            layer.kernel_regularizer = l0_exp(l0=const_l0*alphal0c, l1=0., l2=alphal2, beta=beta)
          else: 
            layer.kernel_regularizer = l0_exp(l0=alphal0c, l1=0., l2=alphal2, beta=beta)
    #     print(layer.get_config())
    
    # This part is needed for the added regularizations to take effect:
    model_json = model.to_json()
    model = tf.keras.models.model_from_json(model_json)
    for layer in model.submodules:
        if isinstance(layer,Conv2D) and layer.kernel_regularizer:
            print(layer.kernel_regularizer.get_config())
    print(model.losses)
    
    # Compiling the model:
    # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer = optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setting a periodic model saving as a precaution:
    cpointcallback = tf.keras.callbacks.ModelCheckpoint(f"resnet50_comregl2l0_beta{beta}_a{alphal0d}_a{alphal0c}_a2{alphal2}_NORM{NORM}_incrbatch_new10.tf", 
                                                        verbose = 1, 
                                                        save_weights_only = True, 
                                                        period = 5,
                                                        save_format = "tf")
    
    rseed = random.randint(1,1000) # Random seed for train/validation split
    start_time=timer(None)
    # Training the model:
    try:
      result = model.fit(
          train_datagen.flow(x_train, y_train, batch_size = 128,subset='training',seed=rseed,shuffle=True),
          validation_data=valid_datagen.flow(x_train, y_train, batch_size = 128,subset='validation',seed=rseed,shuffle=True),
          epochs = 100,
          verbose = 2,
          callbacks = [learning_rate_reduction,cpointcallback]
      )
    except KeyboardInterrupt:
      print("\n\nInterrupted!\n\n")
    timer(start_time)
    
    model.save_weights(f"resnet50_comregl2l0_beta{beta}_a{alphal0d}_a{alphal0c}_a2{alphal2}_NORM{NORM}_incrbatch_final_10.tf",save_format="tf")