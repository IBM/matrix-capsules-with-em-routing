"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com
"""

import tensorflow as tf
from datetime import datetime   # date stamp the log directory
import json  # for saving and loading hyperparameters
import os, sys, re
import time

import daiquiri
import logging
logger = daiquiri.getLogger(__name__)

flags = tf.app.flags

# Need this line for flags to work with Jupyter
# https://github.com/tensorflow/tensorflow/issues/17702
flags.DEFINE_string('f', '', 'kernel')


#------------------------------------------------------------------------------
# HYPERPARAMETERS
#------------------------------------------------------------------------------
# set to 64 according to authors (https://openreview.net/forum?id=HJWLfGWRb)
flags.DEFINE_integer('batch_size', 64, 'batch size in total across all gpus') 
flags.DEFINE_integer('epoch', 100, 'epoch')
flags.DEFINE_integer('iter_routing', 2, 'number of iterations')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')
flags.DEFINE_float('lrn_rate', 3e-3, 'learning rate to use in Adam optimiser')
flags.DEFINE_boolean('weight_reg', False,
                     'train with regularization of weights')
flags.DEFINE_float('nn_weight_reg_lambda', 2e-7, '''lagrange multiplier for
                    l2 weight regularization constraint of non-capsule weights''')
flags.DEFINE_float('capsule_weight_reg_lambda', 0, '''lagrange multiplier for
                    l2 weight regularization constraint of capsule weights''')
flags.DEFINE_float('recon_loss_lambda', 1, '''lagrange multiplier for
                    reconstruction loss constraint''')
flags.DEFINE_string('norm', 'norm2', 'norm type')
flags.DEFINE_float('final_temp', 0.01, '''final temperature used in
                    EM routing activations''')
flags.DEFINE_boolean('affine_voting', True, '''whether to use affine instead
                     of linear transformations to calculate votes''')
flags.DEFINE_float('drop_rate', 0.5, 'proportion of routes or capsules dropped')
flags.DEFINE_boolean('dropout', False, '''whether to apply dropout''')
flags.DEFINE_boolean('dropconnect', False, '''whether to apply dropconnect''')
flags.DEFINE_boolean('dropout_extra', False, '''whether to apply extra dropout''')
#------------------------------------------------------------------------------
# ARCHITECTURE PARAMETERS
#------------------------------------------------------------------------------
flags.DEFINE_string('dataset', 'smallNORB',
                    '''dataset name: currently only "smallNORB, mnist,
                     cifar10, svhn, and imagenet64" supported,
                     feel free to add your own''')
flags.DEFINE_integer('A', 64, 'number of channels in output from ReLU Conv1')
flags.DEFINE_integer('B', 8, 'number of capsules in output from PrimaryCaps')
flags.DEFINE_integer('C', 16, 'number of channels in output from ConvCaps1')
flags.DEFINE_integer('D', 16, 'number of channels in output from ConvCaps2')
flags.DEFINE_boolean('deeper', False, '''whether or not to go deeper''')
flags.DEFINE_boolean('rescap', False, '''whether or not to add residual
                      capsule routes to the final class layer''') # not supported yet
flags.DEFINE_integer('E', 8, 'number of channels in output from ConvCaps3')
flags.DEFINE_integer('F', 16, 'number of channels in output from ConvCaps4')
flags.DEFINE_integer('G', 16, 'number of channels in output from ConvCaps5')
flags.DEFINE_boolean('recon_loss', False, '''whether to apply reconstruction
                      loss''')
flags.DEFINE_boolean('multi_weighted_pred_recon', False, '''whether to use multiple
                      weighted predicted classes instead of single label for decoder
                      input''')
flags.DEFINE_integer('num_bg_classes', 0, '''number of background
                      classes for decoder''')
flags.DEFINE_integer('X', 512, 'number of neurons in reconstructive layer 1')
flags.DEFINE_integer('Y', 1024, 'number of neurons in reconstructive layer 2')
flags.DEFINE_boolean('zeroed_bg_reconstruction', False, '''whether to return
                      counter factual reconstruction output on zeroed bg''')
#------------------------------------------------------------------------------
# ADVERSARIAL PATCH PARAMETERS
#------------------------------------------------------------------------------
# also modify recon_loss and recon_loss_lambda to adjust patch optimization parameters
flags.DEFINE_boolean('train_on_test', True, '''whether to train patch on the test dataset
                                           for stronger performance''')
flags.DEFINE_boolean('new_patch', False, '''whether to start training a new patch from ckpt,
                                         which excludes restoring of certain variables''')
flags.DEFINE_float('max_rotation', 22.5, '''max degree of rotation in random
                                         patch transformations''')
# train scale values from https://github.com/tensorflow/cleverhans/blob/master/examples/adversarial_patch/AdversarialPatch.ipynb
flags.DEFINE_float('scale_min', 0.3, '''patch scaling minimum''')
flags.DEFINE_float('scale_max', 1.5, '''patch scaling maximum''')
flags.DEFINE_integer('target_class', 0, '''the targeted class for adversarial patch''')
flags.DEFINE_boolean('carliniwagner', True, '''whether to use carlini's adversarial loss''')
flags.DEFINE_float('adv_conf_thres', 20, '''logit confidence of the adversarial example,
                                            default to 20 per C&W 17 for best transferability''')

# for sampling reconstruction losses
flags.DEFINE_boolean('adv_patch', True, '''whether to sample reconstruction losses with
                                        adversarial patch at different scales''')
flags.DEFINE_boolean('save_patch', False, '''whether to save the patch''')
flags.DEFINE_string('partition', "train", '''dataset partition to sample reconstruction losses from''')
flags.DEFINE_string('patch_path', None, '''filepath of the patch to be loaded''')
#------------------------------------------------------------------------------
# ENVIRONMENT SETTINGS
#------------------------------------------------------------------------------
flags.DEFINE_integer('num_gpus', 1, 'number of GPUs')
flags.DEFINE_integer('num_threads', 8, 
                     'number of parallel calls in the input pipeline')
flags.DEFINE_string('mode', 'train', 'train, validate, or test')
flags.DEFINE_string('name', '', 'name of experiment in log directory')
flags.DEFINE_boolean('reset', False, 'clear the train or test log directory')
flags.DEFINE_string('debugger', None, 
                    '''set to host of TensorBoard debugger e.g. "dccxc180:8886 
                    or dccxl015:8770"''')
flags.DEFINE_boolean('profile', False, 
                     '''get runtime statistics to display inTensorboard e.g. 
                     compute time''')
flags.DEFINE_string('load_dir', None, 
                    '''directory containing train or test checkpoints to 
                    continue from''')
flags.DEFINE_string('ckpt_name', None, 
                    '''None to load the latest ckpt; all to load all ckpts in 
                      dir; name to load specific ckpt''')
flags.DEFINE_string('params_path', None, 'path to JSON containing parameters')
flags.DEFINE_string('logdir', 'default', 'subdirectory in which logs are saved')

LOCAL_STORAGE = './'
flags.DEFINE_string('storage', LOCAL_STORAGE, 
                    'directory where logs and data are stored')
flags.DEFINE_string('db_name', 'capsules_ex1', 
                    'Name of the DB for mongo for sacred')

# Parse flags
FLAGS = flags.FLAGS


#------------------------------------------------------------------------------
# DIRECTORIES
#------------------------------------------------------------------------------
def setup_train_directories():
  
  # Set log directory
  date_stamp = datetime.now().strftime('%Y%m%d_%H:%M:%S:%f')
  save_dir = os.path.join(tf.app.flags.FLAGS.storage, 'logs/',
              tf.app.flags.FLAGS.dataset, tf.app.flags.FLAGS.logdir)
  train_dir = '{}/{}_{}/train'.format(save_dir, date_stamp, FLAGS.name)

  # Clear the train log directory
  if FLAGS.reset is True and tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)

  # Create train directory
  if not tf.gfile.Exists(train_dir):
    tf.gfile.MakeDirs(train_dir)

  # Set summary directory
  train_summary_dir = os.path.join(train_dir, 'summary')

  # Create summary directory
  if not tf.gfile.Exists(train_summary_dir):
    tf.gfile.MakeDirs(train_summary_dir)
    
  return train_dir, train_summary_dir


#------------------------------------------------------------------------------
# SETUP LOGGER
#------------------------------------------------------------------------------
def setup_logger(logger_dir, name="logger"):
  os.environ['TZ'] = 'US/Eastern'
  time.tzset()
  daiquiri_formatter = daiquiri.formatter.ColorFormatter(
      fmt= "%(asctime)s %(color)s%(levelname)s: %(message)s%(color_stop)s",
      datefmt="%Y-%m-%d %H:%M:%S")
  logger_path = os.path.join(logger_dir, name)
  daiquiri.setup(level=logging.INFO, outputs=(
      daiquiri.output.Stream(formatter=daiquiri_formatter),
      daiquiri.output.File(logger_path,formatter=daiquiri_formatter),
     ))
  # To access the logger from other files, just put this line at the top:
  # logger = daiquiri.getLogger(__name__)

  
#------------------------------------------------------------------------------
# LOAD OR SAVE HYPERPARAMETERS
#------------------------------------------------------------------------------
def load_or_save_hyperparams(train_dir=None):
     
  # Load parameters from file
  # params_path is given in the case that run a new training using existing 
  # parameters
  # load_dir is given in the case of testing or continuing training 
  if FLAGS.params_path or FLAGS.load_dir:

    if FLAGS.params_path:
      params_path = os.path.abspath(FLAGS.params_path)
    elif FLAGS.load_dir:
      params_path = os.path.join(FLAGS.load_dir, "train", 
                     "params", "params.json")
      params_path = os.path.abspath(params_path)

    with open(params_path, 'r') as params_file:
      params = json.load(params_file)
      
      # Get list of flags that were specifically set in command line
      cl_args = sys.argv[1:]
      specified_flags = [re.search('--(.*)=', s).group(1) for s in cl_args]
      
      for name, value in params.items():
        # ignore flags that were specifically set./run in command line
        if name in specified_flags:
          pass
        else:
          FLAGS.__flags[name].value = value 
    logger.info("Loaded parameters from file: {}".format(params_path))

  # Save parameters to file
  if FLAGS.mode == 'train' and train_dir is not None: 
    params_dir_path = os.path.join(train_dir, "params")
    os.makedirs(params_dir_path, exist_ok=True)
    params_file_path = os.path.join(params_dir_path, "params.json")
    params = FLAGS.flag_values_dict()
    params_json = json.dumps(params, indent=4, separators=(',', ':'))
    with open(params_file_path, 'w') as params_file:
      params_file.write(params_json)
    logger.info("Parameters saved to file: {}".format(params_file_path))


#------------------------------------------------------------------------------
# FACTORIES FOR DATASET
#------------------------------------------------------------------------------
def get_dataset_path(dataset_name: str):
  # dataset does not return path if using tensorflow_datasets
  # those are actually saved under ~/tensorflow_datasets/
  options = {'smallNORB': 'data/smallNORB/tfrecord',
             'mnist': '',
             'cifar10': '',
             'svhn': '',
             'imagenet64': ''}
  path = FLAGS.storage + options[dataset_name]
  return path


def get_dataset_size_train(dataset_name: str):
  options = {'mnist': 55000, 
             'smallNORB': 23400 * 2,
             'fashion_mnist': 55000, 
             'cifar10': 50000, 
             'cifar100': 50000,
             'svhn': 73257,
             'imagenet64': 1281167}
  return options[dataset_name]


def get_dataset_size_test(dataset_name: str):
  if dataset_name is 'imagenet64':
    logger.info("%s pipeline is not set up for testing, using validation set for testing instead"%dataset_name)
    return get_dataset_size_validate(dataset_name)
  options = {'mnist': 10000, 
             'smallNORB': 23400 * 2,
             'fashion_mnist': 10000, 
             'cifar10': 10000, 
             'cifar100': 10000,
             'svhn': 26032}
  return options[dataset_name]


def get_dataset_size_validate(dataset_name: str):
  if dataset_name == 'smallNORB' or dataset_name == 'mnist' or dataset_name == 'cifar10' or dataset_name == 'svhn':
    logger.info("%s pipeline is not set up for validation, using test set for validation instead"%dataset_name)
    return get_dataset_size_test(dataset_name)
  options = {'imagenet64': 50000}
  return options[dataset_name]


def get_num_classes(dataset_name: str):
  options = {'mnist': 10, 
             'smallNORB': 5, 
             'fashion_mnist': 10, 
             'cifar10': 10, 
             'cifar100': 100,
             'svhn': 10,
             'imagenet64': 1000}
  return options[dataset_name]


from data_pipelines import norb as data_norb
from data_pipelines import mnist as data_mnist
from data_pipelines import cifar10 as data_cifar10
from data_pipelines import svhn as data_svhn
from data_pipelines import imagenet64 as data_imagenet64
def get_create_inputs(dataset_name: str, mode="train"):
  
  force_set = None
  if mode == "train":
    is_train = True
  else:
    # for dataset pipelines that don't have validation set up
    is_train = False
  if mode == "train_whole":
    force_set = "train"
  elif mode == "train_on_test":
    force_set = "test"
   
  path = get_dataset_path(dataset_name)
  
  options = {'smallNORB':
                 lambda: data_norb.create_inputs_norb(path, is_train, force_set),
             'mnist':
                 lambda: data_mnist.create_inputs(is_train, force_set),
             'cifar10':
                 lambda: data_cifar10.create_inputs(is_train, force_set),
             'svhn':
                 lambda: data_svhn.create_inputs(is_train, force_set),
             'imagenet64':
                 lambda: data_imagenet64.create_inputs(is_train, force_set)}
  return options[dataset_name]


import models as mod
def get_dataset_architecture(dataset_name: str):
  # options = {'smallNORB': mod.build_arch_smallnorb,
  #            'baseline': mod.build_arch_baseline,
  #            'mnist': mod.build_arch_smallnorb,
  #            'cifar10': mod.build_arch_smallnorb}
  # return options[dataset_name]
  if FLAGS.deeper:
    return mod.build_arch_deepcap
  return mod.build_arch_smallnorb 

