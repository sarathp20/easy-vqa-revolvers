import json
import argparse
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

# instantiate an empty dict
# team = {}

# add a team member
# team['train_X_ims'] = train_X_ims
# team['train_X_seqs'] = train_X_seqs
# team['train_Y'] = train_Y
# team['test_X_ims'] = test_X_ims
# team['test_X_seqs'] = test_X_seqs
# team['test_Y'] = test_Y
# team['im_shape'] = im_shape
# team['vocab_size'] = vocab_size
# team['num_answers'] = num_answers

# with open('mydata.json', 'w') as f:
#     json.dump(team, f)
import numpy as np
import os
np.savez('temp_arra.npz', train_X_ims=train_X_ims,train_X_seqs = train_X_seqs, train_Y = train_Y, test_X_ims = test_X_ims, test_X_seqs = test_X_seqs, test_Y = test_Y, im_shape = im_shape, vocab_size = vocab_size, num_answers= num_answers)

