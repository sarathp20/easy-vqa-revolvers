from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
from model import build_model
from prepare_data import setup

# Support command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--big-model', action='store_true', help='Use the bigger model with more conv layers')
parser.add_argument('--use-data-dir', action='store_true', help='Use custom data directory, at /data')
args = parser.parse_args()

if args.big_model:
  print('Using big model')
if args.use_data_dir:
  print('Using data directory')

# Prepare data
train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, _, _, _ = setup(args.use_data_dir)

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers, args.big_model)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

print('\n--- Training model...')
print("checkpoint==",checkpoint)
model.fit(
  [train_X_ims, train_X_seqs],
  train_Y,
  validation_data=([test_X_ims, test_X_seqs], test_Y),
  shuffle=True,
  epochs=8,
  callbacks=[checkpoint],
)

from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix

y_pred = model.predict([test_X_ims, test_X_seqs])
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(test_Y, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % (accuracy * 100))
