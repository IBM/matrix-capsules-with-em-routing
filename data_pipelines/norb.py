"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com
"""

import tensorflow as tf
import numpy as np

import os
import re

from config import FLAGS


def _parser(serialized_example):
  """Parse smallNORB example from tfrecord.
  
  Author:
    Ashley Gritzman 15/11/2018
  Args: 
    serialized_example: serialized example from tfrecord  
  Returns:
    img: image
    lab: label
    cat: 
      category
      the instance in the category (0 to 9)
    elv: 
      elevation
      the elevation (0 to 8, which mean cameras are 30, 
      35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
    azi: 
      azimuth
      the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in 
      degrees)
    lit: 
      lighting
      the lighting condition (0 to 5)
  """

  features = tf.parse_single_example(
    serialized_example, 
    features={
      'img_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
      'category': tf.FixedLenFeature([], tf.int64), 
      'elevation': tf.FixedLenFeature([], tf.int64), 
      'azimuth': tf.FixedLenFeature([], tf.int64), 
      'lighting': tf.FixedLenFeature([], tf.int64),
     })

  img = tf.decode_raw(features['img_raw'], tf.float64)
  img = tf.reshape(img, [96, 96, 1])
  img = tf.cast(img, tf.float32)  # * (1. / 255) # left unnormalized

  lab = tf.cast(features['label'], tf.int32)
  cat = tf.cast(features['category'], tf.int32)
  elv = tf.cast(features['elevation'], tf.int32)
  azi = tf.cast(features['azimuth'], tf.int32)
  lit = tf.cast(features['lighting'], tf.int32)

  return img, lab, cat, elv, azi, lit


def _train_preprocess(img, lab, cat, elv, azi, lit):
  """Preprocessing for training.
  
  Preprocessing from Hinton et al. (2018) "Matrix capsules with EM routing."
  Hinton2018: "We downsample smallNORB to 48 × 48 pixels and normalize each 
  image to have zero mean and unit variance. During training, we randomly crop 
  32 × 32 patches and add random brightness and contrast to the cropped images.
  During test, we crop a 32 × 32 patch from the center of the image and 
  achieve..."
  
  Author:
    Ashley Gritzman 15/11/2018
  Args: 
    img: this fn only works on the image
    lab, cat, elv, azi, lit: allow these to pass through  
  Returns:
    img: image processed
    lab, cat, elv, azi, lit: allow these to pass through   
  """
  
  img = img / 255.
  img = tf.image.resize_images(img, [48, 48])
  img = tf.image.per_image_standardization(img)
  img = tf.random_crop(img, [32, 32, 1])
  img = tf.image.random_brightness(img, max_delta = 2.0)
  #original 0.5, 1.5
  img = tf.image.random_contrast(img, lower=0.5, upper=1.5) 
  
  # Original
  # image = tf.image.random_brightness(image, max_delta=32. / 255.)
  # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
  # image = tf.image.resize_images(image, [48, 48])
  # image = tf.random_crop(image, [32, 32, 1])

  return img, lab, cat, elv, azi, lit


def _val_preprocess(img, lab, cat, elv, azi, lit):
  """Preprocessing for validation/testing.
  
  Preprocessing from Hinton et al. (2018) "Matrix capsules with EM routing." 
  Hinton2018: "We downsample smallNORB to 48 × 48 pixels and normalize each 
  image to have zero mean and unit variance. During training, we randomly crop 
  32 × 32 patches and add random brightness and contrast to the cropped 
  images. During test, we crop a 32 × 32 patch from the center of the image 
  and achieve..."
  
  Author:
    Ashley Gritzman 15/11/2018
  Args: 
    img: this fn only works on the image
    lab, cat, elv, azi, lit: allow these to pass through  
  Returns:
    img: image processed
    lab, cat, elv, azi, lit: allow these to pass through   
  """
  
  img = img / 255.
  img = tf.image.resize_images(img, [48, 48])
  img = tf.image.per_image_standardization(img)
  img = tf.slice(img, [8, 8, 0], [32, 32, 1])
  
  # Original
  # image = tf.image.resize_images(image, [48, 48])
  # image = tf.slice(image, [8, 8, 0], [32, 32, 1])

  return img, lab, cat, elv, azi, lit
  

def input_fn(path, is_train: bool, force_set=None):
  """Input pipeline for smallNORB using tf.data.
  
  Author:
    Ashley Gritzman 15/11/2018
  Args: 
    is_train:  
  Returns:
    dataset: image tf.data.Dataset 
  """

  import re
  split = "train" if is_train else "test"
  if force_set is not None:
    split = force_set
  CHUNK_RE = re.compile(r"%s.*\.tfrecords"%split)

  chunk_files = [os.path.join(path, fname)
           for fname in os.listdir(path)
           if CHUNK_RE.match(fname)]
  
  # 1. create the dataset
  dataset = tf.data.TFRecordDataset(chunk_files)
  
  # 2. map with the actual work (preprocessing, augmentation…) using multiple 
  # parallel calls
  dataset = dataset.map(_parser, num_parallel_calls=FLAGS.num_threads)
  if is_train:
    dataset = dataset.map(_train_preprocess, 
                          num_parallel_calls=FLAGS.num_threads)
  else:
    dataset = dataset.map(_val_preprocess, 
                          num_parallel_calls=FLAGS.num_threads)
  
  # 3. shuffle (with a big enough buffer size)
  # In response to a question on OpenReview, Hinton et al. wrote the 
  # following:
  # https://openreview.net/forum?id=HJWLfGWRb&noteId=rJgxonoNnm
  # "We did not have any special ordering of training batches and we random 
  # shuffle. In terms of TF batch:
  # capacity=2000 + 3 * batch_size, ensures a minimum amount of shuffling of 
  # examples. min_after_dequeue=2000."
  capacity = 2000 + 3 * FLAGS.batch_size
  dataset = dataset.shuffle(buffer_size = capacity)
    
  # 4. batch
  dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

  # 5. repeat
  dataset = dataset.repeat()
  
  # 6. prefetch
  dataset = dataset.prefetch(1)
  
  return dataset


def create_inputs_norb(path, is_train: bool, force_set=None):
  """Get a batch from the input pipeline.
  
  Author:
    Ashley Gritzman 15/11/2018
  Args: 
    is_train:  
  Returns:
    img, lab, cat, elv, azi, lit: 
  """
  
  # Create batched dataset
  dataset = input_fn(path, is_train, force_set)
  
  # Create one-shot iterator
  iterator = dataset.make_one_shot_iterator()
  
  img, lab, cat, elv, azi, lit = iterator.get_next()
  
  output_dict = {'image': img,
           'label': lab,
           'category': cat,
           'elevation': elv,
           'azimuth': azi,
           'lighting': lit}
  
  return output_dict


def plot_smallnorb(is_train=True, samples_per_class=5):
  """Plot examples from the smallNORB dataset.
  
  Execute this command in a Jupyter Notebook.
  
  Author:
    Ashley Gritzman 18/04/2019
  Args: 
    is_train: True for the training dataset, False for the test dataset
    samples_per_class: number of samples images per class
  Returns:
    None
  """
  
  # To plot pretty figures
  import matplotlib.pyplot as plt
  plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
  plt.rcParams['image.interpolation'] = 'nearest'
  plt.rcParams['image.cmap'] = 'gray'
  
  from config import get_dataset_path
  path = get_dataset_path("smallNORB")
  
  CLASSES = ['animal', 'human', 'airplane', 'truck', 'car']

  # Get batch from data queue. Batch size is FLAGS.batch_size, which is then 
  # divided across multiple GPUs
  input_dict = create_inputs_norb(path, is_train=is_train)
  with tf.Session() as sess:
    input_dict = sess.run(input_dict)
    
  img_bch = input_dict['image']
  lab_bch = input_dict['label']
  cat_bch = input_dict['category']
  elv_bch = input_dict['elevation']
  azi_bch = input_dict['azimuth']
  lit_bch = input_dict['lighting']
  
  num_classes = len(CLASSES)

  fig = plt.figure(figsize=(num_classes * 2, samples_per_class * 2))
  fig.suptitle("category, elevation, azimuth, lighting")  
  for y, cls in enumerate(CLASSES):
    idxs = np.flatnonzero(lab_bch == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
      plt_idx = i * num_classes + y + 1
      plt.subplot(samples_per_class, num_classes, plt_idx)
      #plt.imshow(img_bch[idx].astype('uint8').squeeze())
      plt.imshow(np.squeeze(img_bch[idx]))
      plt.xticks([], [])
      plt.yticks([], [])
      plt.xlabel("{}, {}, {},{}".format(cat_bch[idx], elv_bch[idx], 
                        azi_bch[idx], lit_bch[idx]))
      # plt.axis('off')

      if i == 0:
        plt.title(cls)
  plt.show()
