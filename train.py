from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
from model import build_model
# from prepare_data import setup
import matplotlib.pyplot as plt
import numpy as np

# Support command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--big-model', action='store_true', help='Use the bigger model with more conv layers')
parser.add_argument('--use-data-dir', action='store_true', help='Use custom data directory, at /data')
args = parser.parse_args()

if args.big_model:
  print('Using big model')
if args.use_data_dir:
  print('Using data directory')

# # Prepare data
# train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, _, _, _ = setup(args.use_data_dir)
with np.load('temp_arra.npz') as data:
    train_X_ims = data['train_X_ims'] 
    train_X_seqs = data['train_X_seqs'] 
    train_Y = data['train_Y'] 
    test_X_ims = data['test_X_ims'] 
    test_X_seqs = data['test_X_seqs']
    test_Y = data['test_Y'] 
    im_shape = data['im_shape'] 
    vocab_size = data['vocab_size'] 
    num_answers = data['num_answers']

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers, args.big_model)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
# early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

print('\n--- Training model...')
print("checkpoint==",checkpoint)
history = model.fit(
  [train_X_ims, train_X_seqs],
  train_Y,
  validation_data=([test_X_ims, test_X_seqs], test_Y),
  shuffle=True,
  epochs=20,
  callbacks=[checkpoint],
)

print("acc = ", history.history['accuracy'])
print("val_acc = ", history.history['val_accuracy'])
print("loss = ", history.history['loss'])
print("val_loss = ", history.history['val_loss'])

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# from sklearn.metrics import accuracy_score
# import numpy as np
# from sklearn.metrics import confusion_matrix

# y_pred = model.predict([test_X_ims, test_X_seqs])
# y_pred=np.argmax(y_pred, axis=1)
# y_test=np.argmax(test_Y, axis=1)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# # evaluate predictions
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy: %.3f' % (accuracy * 100))
