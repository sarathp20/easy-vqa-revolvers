from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import json
import os
from text_extraction_using_bert_no_output import text_feature_extraction
import numpy as np

def setup(use_data_dir):
  print('\n--- Reading questions...')
  # Read data from json file
  def read_questions(path):
    with open(path, 'r') as file:
        qs = json.load(file)
    texts = [q[0] for q in qs]
    answers = [q[1] for q in qs]
    image_ids = [int(q[2]) for q in qs]
    return (texts, answers, image_ids)
  train_qs, train_answers, train_image_ids = read_questions('qstn_ans_id_dataset.json')
  test_qs, test_answers, test_image_ids = read_questions('test.json')
  print(f'Read {len(train_qs)} training questions and {len(test_qs)} testing questions.')
	
  print('\n--- Reading answers...')
  # Read answers from answers.txt
  with open('answers1.txt', 'r') as file:
    all_answers = [a.strip() for a in file]
  num_answers = len(all_answers)
  print(f'Found {num_answers} total answers:')
  print(all_answers)
	
  print('\n--- Reading/processing images...')
  def load_and_proccess_image(image_path):
    # Load image, then scale and shift pixel values to [-0.5, 0.5]
    im = img_to_array(load_img(image_path))
    return im / 255 - 0.5
	
  def read_images(paths):
    # paths is a dict mapping image ID to image path
    # Returns a dict mapping image ID to the processed image
    ims = {}
    for image_id, image_path in paths.items():
      ims[image_id] = load_and_proccess_image(image_path)
    return ims
	
    # Read images from data/ folder
  def extract_paths(dir):
    paths = {}
    for filename in os.listdir(dir):
        if filename.endswith('.png'):
            image_id = int(filename[:-4])
            paths[image_id] = os.path.join(dir, filename)
    return paths

  train_ims = read_images(extract_paths('Img/S7ProjectDataset'))
  test_ims  = read_images(extract_paths('Img/testImage'))
  im_shape = train_ims[101].shape
  print(f'Read {len(train_ims)} training images and {len(test_ims)} testing images.')
  print(f'Each image has shape {im_shape}.')
	
  print('\n--- Fitting question tokenizer...')
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(train_qs)
"""
The fit_on_texts method is a part of Keras tokenizer class which is used to update the internal vocabulary for the texts list. We need to call be before using other methods of texts_to_sequences or texts_to_matrix.

The object returned by fit_on_texts can be used to derive more information by using the following attributes-
word_counts : It is a dictionary of words along with the counts.
word_docs : Again a dictionary of words, this tells us how many documents contain this word
word_index : In this dictionary, we have unique integers assigned to each word.
document_count : This integer count will tell us the total number of documents used for fitting the tokenizer.

"""

    # We add one because the Keras Tokenizer reserves index 0 and never uses it.
  vocab_size = len(tokenizer.word_index) + 1
  print(f'Vocab Size: {vocab_size}')
  print(tokenizer.word_index)
	
#   print('\n--- Converting questions to bags of words...')
#   train_X_seqs = tokenizer.texts_to_matrix(train_qs)
#   test_X_seqs = tokenizer.texts_to_matrix(test_qs)
#   print(f'Example question bag of words: {train_X_seqs[0]}')

  print("Performing feature extraction from text:")
  train_X_seqs = text_feature_extraction(train_qs)
  test_X_seqs = text_feature_extraction(test_qs)

  print('\n--- Creating model input images...')
  train_X_ims = np.array([train_ims[id] for id in train_image_ids])
  test_X_ims = np.array([test_ims[id] for id in test_image_ids])

  print('\n--- Creating model outputs...')
  train_answer_indices = [all_answers.index(a) for a in train_answers]
  test_answer_indices = [all_answers.index(a) for a in test_answers]
  print("check==",len(train_answer_indices),len(test_answer_indices))
  train_Y = to_categorical(train_answer_indices)
   #train_Y.reshape(len(train_X_seqs),768)
  test_Y = to_categorical(test_answer_indices,num_classes=323)
  #test_Y.reshape(len(test_X_seqs),768)
  print(train_Y.shape,test_Y.shape)	
  print(f'Example model output: {train_Y[0]}')


  return (train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs,
          test_Y, im_shape, vocab_size, num_answers,
          all_answers, test_qs, test_answer_indices)  # for the analyze script
