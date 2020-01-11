"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com

Credits: 
  Suofei Zhang & Hang Yu, "Matrix-Capsules-EM-Tensorflow" 
  https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
"""

import tensorflow as tf
import os, sys, time
import tensorflow.contrib.slim as slim
import datetime   # date stamp the log directory
import shutil     # to remove a directory
# to sort files in directory by date
from stat import S_ISREG, ST_CTIME, ST_MODE   
import re   # for regular expressions
import sklearn.metrics as skm
import numpy as np

# Get logger that has already been created in config.py
import daiquiri
logger = daiquiri.getLogger(__name__)

from config import FLAGS
import config as conf
import models as mod


from adv_patch_train_val import patch_inputs

# for outputting sample to csv
import pandas as pd

# load and save png patch
from PIL import Image
import re

def main(args):
  
  # Set reproduciable random seed
  tf.set_random_seed(1234)
  
  # Directories
  # Get name
  split = FLAGS.load_dir.split('/')
  if split[-1]:
    name = split[-1]
  else:
    name = split[-2]
    
  # Get parent directory
  split = FLAGS.load_dir.split("/" + name)
  parent_dir = split[0]

  test_dir = '{}/{}/test'.format(parent_dir, name)
  test_summary_dir = test_dir + '/summary'

  # Clear the test log directory
  if (FLAGS.reset is True) and os.path.exists(test_dir):
    shutil.rmtree(test_dir) 
  if not os.path.exists(test_summary_dir):
    os.makedirs(test_summary_dir)
  
  # Logger
  conf.setup_logger(logger_dir=test_dir, name="logger_test.txt")
  logger.info("name: " + name)
  logger.info("parent_dir: " + parent_dir)
  logger.info("test_dir: " + test_dir)
  
  # Load hyperparameters from train run
  conf.load_or_save_hyperparams()
  
  # Get dataset hyperparameters
  logger.info('Using dataset: {}'.format(FLAGS.dataset))
  
  # Dataset
  dataset_size_test  = conf.get_dataset_size_test(FLAGS.dataset) if FLAGS.partition == "test" else conf.get_dataset_size_train(FLAGS.dataset)
  num_classes        = conf.get_num_classes(FLAGS.dataset)
  create_inputs_test = conf.get_create_inputs(FLAGS.dataset, mode=FLAGS.partition)

  
  #----------------------------------------------------------------------------
  # GRAPH - TEST
  #----------------------------------------------------------------------------
  logger.info('BUILD TEST GRAPH')
  g_test = tf.Graph()
  with g_test.as_default():
    # Get global_step
    global_step = tf.train.get_or_create_global_step()

    num_batches_test = int(dataset_size_test / FLAGS.batch_size)

    # Get data
    input_dict = create_inputs_test()
    batch_x = input_dict['image']
    batch_labels = input_dict['label']
    
    # AG 10/12/2018: Split batch for multi gpu implementation
    # Each split is of size FLAGS.batch_size / FLAGS.num_gpus
    # See: https://github.com/naturomics/CapsNet-
    # Tensorflow/blob/master/dist_version/distributed_train.py
    splits_x = tf.split(
        axis=0, 
        num_or_size_splits=FLAGS.num_gpus, 
        value=batch_x)
    splits_labels = tf.split(
        axis=0, 
        num_or_size_splits=FLAGS.num_gpus, 
        value=batch_labels)
    
    # Build architecture
    build_arch = conf.get_dataset_architecture(FLAGS.dataset)
    # for baseline
    #build_arch = conf.get_dataset_architecture('baseline')
    
    #--------------------------------------------------------------------------
    # MULTI GPU - TEST
    #--------------------------------------------------------------------------
    # Calculate the logits for each model tower
    tower_logits = []
    tower_recon_losses = []
    reuse_variables = None
    with tf.device("/cpu:0"):
      scale_min_feed = tf.placeholder(tf.float32, shape=[], name="scale_min_feed")
      scale_max_feed = tf.placeholder(tf.float32, shape=[], name="scale_max_feed")
    patch_feed = None
    if FLAGS.patch_path:
      patch_feed = tf.placeholder(tf.float32,
                                  shape=batch_x.get_shape().as_list()[-3:],
                                  name="patch_feed")
    for i in range(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('tower_%d' % i) as scope:
          with slim.arg_scope([slim.variable], device='/cpu:0'):
            logits, recon_losses, patch_node = tower_fn(
                build_arch,
                splits_x[i],
                scale_min_feed,
                scale_max_feed,
                patch_feed,
                scope,
                num_classes,
                reuse_variables=reuse_variables, 
                is_train=False)

          # Don't reuse variable for first GPU, but do reuse for others
          reuse_variables = True
          # Keep track of losses and logits across for each tower
          tower_logits.append(logits)
          tower_recon_losses.append(recon_losses)
    # Combine logits from all towers
    test_metrics = {}
    if not FLAGS.save_patch:
      test_logits = tf.concat(tower_logits, axis=0)
      test_preds = tf.argmax(test_logits, axis=-1)
      test_recon_losses = tf.concat(tower_recon_losses, axis=0)
      test_metrics = {'preds': test_preds,
                     'labels': batch_labels,
                     'recon_losses': test_recon_losses
                     }
    if FLAGS.adv_patch:
      test_metrics['patch'] = patch_node
    
    # Reset and read operations for streaming metrics go here
    test_reset = {}
    test_read = {}
    
    # Saver
    saver = tf.train.Saver(max_to_keep=None)
    
    # Set summary op
    
  
    #--------------------------------------------------------------------------
    # SESSION - TEST
    #--------------------------------------------------------------------------
    #sess_test = tf.Session(
    #    config=tf.ConfigProto(allow_soft_placement=True, 
    #                          log_device_placement=False), 
    #    graph=g_test)
    # Perry: added in for RTX 2070 incompatibility workaround
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess_test = tf.Session(config=config, graph=g_test)

   
    
    #sess_test.run(tf.local_variables_initializer())
    #sess_test.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(
        test_summary_dir, 
        graph=sess_test.graph)


    ckpts_to_test = []
    load_dir_chechpoint = os.path.join(FLAGS.load_dir, "train", "checkpoint")
    
    # Evaluate the latest ckpt in dir
    if FLAGS.ckpt_name is None:
      latest_ckpt = tf.train.latest_checkpoint(load_dir_chechpoint)
      ckpts_to_test.append(latest_ckpt)

    # Evaluate all ckpts in dir  
    elif FLAGS.ckpt_name == "all":
      # Get list of files in firectory and sort by date created
      filenames = os.listdir(load_dir_chechpoint)
      regex = re.compile(r'.*.index')
      filenames = filter(regex.search, filenames)
      data_ckpts = (os.path.join(load_dir_chechpoint, fn) for fn in filenames)
      data_ckpts = ((os.stat(path), path) for path in data_ckpts)

      # regular files, insert creation date
      data_ckpts = ((stat[ST_CTIME], path) for stat, path in data_ckpts 
                    if S_ISREG(stat[ST_MODE]))
      data_ckpts= sorted(data_ckpts)
      # remove ".index"
      ckpts_to_test = [path[:-6] for ctime, path in data_ckpts]
        
    # Evaluate ckpt specified by name
    else:
      ckpt_name = os.path.join(load_dir_chechpoint, FLAGS.ckpt_name)
      ckpts_to_test.append(ckpt_name)    
      
      
    #--------------------------------------------------------------------------
    # MAIN LOOP
    #--------------------------------------------------------------------------
    # Run testing on checkpoints
    for ckpt in ckpts_to_test:
      saver.restore(sess_test, ckpt)

      if FLAGS.save_patch:
        out = sess_test.run(test_metrics['patch'])
        patch = out
        if patch.shape[-1] == 1:
          patch = np.squeeze(patch, axis=-1)
        formatted = (patch * 255).astype('uint8')
        img = Image.fromarray(formatted)
        img.save(os.path.join(FLAGS.load_dir, "test", "saved_patch.png"))
        return
          
      # Reset accumulators
      sess_test.run(test_reset)
      test_preds_vals = []
      test_labels_vals = []
      test_recon_losses_vals = []
      test_scales = []

      interval = 0.1 if FLAGS.adv_patch else 1
      for scale in np.arange(0, 1, interval):
        for i in range(num_batches_test):
          feed_dict = {scale_min_feed:scale, scale_max_feed:scale}
          if FLAGS.patch_path:
            patch_dims = patch_feed.get_shape()
            patch = np.asarray(Image.open(FLAGS.patch_path), dtype=np.float32)
            if len(patch.shape) < 3:
              patch = np.expand_dims(patch, axis=-1)
          if patch_dims[-1] == 1:
            patch = np.mean(patch, axis=-1, keepdims=True)
          patch = patch/255
          feed_dict[patch_feed] = patch
          out = sess_test.run([test_metrics], feed_dict=feed_dict)
          test_metrics_v = out[0]
          #ckpt_num = re.split('-', ckpt)[-1]
          #logger.info('TEST ckpt-{}'.format(ckpt_num)
          #    + ' bch-{:d}'.format(i)
          #    )
          test_preds_vals.append(test_metrics_v['preds'])
          test_labels_vals.append(test_metrics_v['labels'])
          test_recon_losses_vals.append(test_metrics_v['recon_losses'])
          test_scales.append(np.full(test_metrics_v['preds'].shape, fill_value=scale))

    logger.info('writing to csv')
    test_preds_vals = np.concatenate(test_preds_vals)
    test_labels_vals = np.concatenate(test_labels_vals)
    test_recon_losses_vals = np.concatenate(test_recon_losses_vals)
    test_scales = np.concatenate(test_scales)

    data = {'predictions': test_preds_vals,
            'labels': test_labels_vals,
            'reconstruction_losses': test_recon_losses_vals,
            'scales': test_scales
           }
    filename = "recon_losses.csv"
    if FLAGS.patch_path:
      filename = re.sub('[^\w\-_]', '_', FLAGS.patch_path) + "_" + FLAGS.partition + ".csv"
    csv_save_path = os.path.join(FLAGS.load_dir, FLAGS.partition, filename)
    pd.DataFrame(data).to_csv(csv_save_path, index=False)
    logger.info('csv saved at ' + csv_save_path)

      
def tower_fn(build_arch, 
             x,
             scale_min_feed,
             scale_max_feed,
             patch_feed,
             scope,
             num_classes, 
             is_train=False, 
             reuse_variables=None): 
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    patch = None
    if FLAGS.adv_patch:
      x, patch = patch_inputs(x, is_train=is_train, reuse=reuse_variables, 
                              scale_min=scale_min_feed, scale_max=scale_max_feed,
                              patch_feed=patch_feed)
    output = build_arch(x, is_train, num_classes=num_classes)
  if FLAGS.save_patch:
    return None, None, patch
  decoder_out = output["decoder_out"]
  recon_loss = mod.reconstruction_loss(x, decoder_out, batch_reduce=False)
  return output['scores'], recon_loss, patch
 

if __name__ == "__main__":
  tf.app.run()
