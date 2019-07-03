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
flags.DEFINE_integer('epoch', 2000, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations')
flags.DEFINE_integer('num_gpus', 1, 'number of GPUs')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')
flags.DEFINE_float('lrn_rate', 3e-3, 'learning rate to use in Adam optimiser')
flags.DEFINE_float('val_prop', 0.1, 
                   'proportion of test dataset to use for validation')
flags.DEFINE_boolean('weight_reg', False, 
                     'train with regularization of weights')
flags.DEFINE_string('norm', 'norm2', 'norm type')
flags.DEFINE_integer('num_threads', 8, 
                     'number of parallel calls in the input pipeline')
flags.DEFINE_string('dataset', 'smallNORB', 
                    '''dataset name: currently only "smallNORB" supported, feel
                    free to add your own''')
flags.DEFINE_float('final_lambda', 0.01, 'final lambda in EM routing')


#------------------------------------------------------------------------------
# ARCHITECTURE PARAMETERS
#------------------------------------------------------------------------------
flags.DEFINE_integer('A', 64, 'number of channels in output from ReLU Conv1')
flags.DEFINE_integer('B', 8, 'number of capsules in output from PrimaryCaps')
flags.DEFINE_integer('C', 16, 'number of channels in output from ConvCaps1')
flags.DEFINE_integer('D', 16, 'number of channels in output from ConvCaps2')


#------------------------------------------------------------------------------
# ENVIRONMENT SETTINGS
#------------------------------------------------------------------------------
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

CCC_STORAGE = '/dccstor/astro/ashley/'
LOCAL_STORAGE = './'
flags.DEFINE_string('storage', CCC_STORAGE, 
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
  date_stamp = datetime.now().strftime('%Y%m%d')
  save_dir = os.path.join(tf.app.flags.FLAGS.storage, 'logs/',
              tf.app.flags.FLAGS.dataset)
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
  os.environ['TZ'] = 'Africa/Johannesburg'
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
  elif FLAGS.mode == 'train': 
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
  options = {'smallNORB': 'data/smallNORB/tfrecord'}
  path = FLAGS.storage + options[dataset_name]
  return path


def get_dataset_size_train(dataset_name: str):
  options = {'mnist': 55000, 
             'smallNORB': 23400 * 2,
             'fashion_mnist': 55000, 
             'cifar10': 50000, 
             'cifar100': 50000}
  return options[dataset_name]


def get_dataset_size_test(dataset_name: str):
  options = {'mnist': 10000, 
             'smallNORB': 23400 * 2,
             'fashion_mnist': 10000, 
             'cifar10': 10000, 
             'cifar100': 10000}
  return options[dataset_name]


def get_dataset_size_validate(dataset_name: str):
  options = {'smallNORB': 23400 * 2}
  return options[dataset_name]


def get_num_classes(dataset_name: str):
  options = {'mnist': 10, 
             'smallNORB': 5, 
             'fashion_mnist': 10, 
             'cifar10': 10, 
             'cifar100': 100}
  return options[dataset_name]


import data_pipeline_norb as data_norb
def get_create_inputs(dataset_name: str, mode="train"):
  
  if mode == "train":
    is_train = True
  else:
    is_train = False
    
  path = get_dataset_path(dataset_name)
  
  options = {'smallNORB': 
         lambda: data_norb.create_inputs_norb(path, is_train)}
  return options[dataset_name]


import models as mod
def get_dataset_architecture(dataset_name: str):
  options = {'smallNORB': mod.build_arch_smallnorb,
             'baseline': mod.build_arch_baseline}
  return options[dataset_name]