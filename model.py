from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam

def build_model(im_shape, vocab_size, num_answers, big_model):
#   # The CNN
#   im_input = Input(shape=im_shape)
#   x1 = Conv2D(8, 3, padding='same')(im_input)
#   x1 = MaxPooling2D()(x1)
#   x1 = Conv2D(16, 3, padding='same')(x1)
#   x1 = MaxPooling2D()(x1)
#   if big_model:
#     x1 = Conv2D(32, 3, padding='same')(x1)
#     x1 = MaxPooling2D()(x1)
#   x1 = Flatten()(x1)
  #Load model wothout classifier/fully connected layers
  VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))#256,256,3

  #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
  for layer in VGG_model.layers:
	  layer.trainable = False
    
  #VGG_model.summary()  #Trainable parameters will be 0

  feature_extractor=VGG_model.predict(x_train)

  x1 = feature_extractor.reshape(feature_extractor.shape[0],-1)
  x1 = Dense(32, activation='tanh')(x1)
  print("shape of image vector=",x1.shape)

  # The question network
  q_input = Input(shape=(vocab_size,))
  x2 = Dense(32, activation='tanh')(q_input)
  x2 = Dense(32, activation='tanh')(x2)

  # Merge -> output
  out = Multiply()([x1, x2])
  out = Dense(32, activation='tanh')(out)
  out = Dense(num_answers, activation='softmax')(out)

  model = Model(inputs=[im_input, q_input], outputs=out)
  model.compile(Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

  return model
