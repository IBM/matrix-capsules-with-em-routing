"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com
"""

# Public modules
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# My modules
from config import FLAGS
import utils as utl
import layers as lyr
import em_routing as em

# Get logger that has already been created in config.py
import daiquiri
logger = daiquiri.getLogger(__name__)


#------------------------------------------------------------------------------
# CAPSNET FOR SMALLNORB
#------------------------------------------------------------------------------
def build_arch_smallnorb(inp, is_train: bool, num_classes: int):
  
  logger.info('input shape: {}'.format(inp.get_shape()))
  batch_size = int(inp.get_shape()[0])
  spatial_size = int(inp.get_shape()[1])

  # xavier initialization is necessary here to provide higher stability
  # initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
  # instead of initializing bias with constant 0, a truncated normal 
  # initializer is exploited here for higher stability
  bias_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01) 

  # AG 13/11/2018
  # In response to a question on OpenReview, Hinton et al. wrote the 
  # following:
  # "We use a weight decay loss with a small factor of .0000002 rather than 
  # the reconstruction loss."
  # https://openreview.net/forum?id=HJWLfGWRb&noteId=rJeQnSsE3X
  weights_regularizer = tf.contrib.layers.l2_regularizer(0.0000002)

  # for drop connect during em routing
  drop_rate = FLAGS.drop_rate if is_train else 0

  # weights_initializer=initializer,
  with slim.arg_scope([slim.conv2d], 
    trainable = is_train, 
    biases_initializer = bias_initializer,
    weights_regularizer = weights_regularizer):
    
    #----- Batch Norm -----#
    output = slim.batch_norm(
        inp,
        center=False, 
        is_training=is_train, 
        trainable=is_train)
    
    #----- Convolutional Layer 1 -----#
    with tf.variable_scope('relu_conv1') as scope:
      output = slim.conv2d(output, 
      num_outputs=FLAGS.A, 
      kernel_size=[5, 5], 
      stride=2, 
      padding='SAME', 
      scope=scope, 
      activation_fn=tf.nn.relu)
      
      spatial_size = int(output.get_shape()[1])
      assert output.get_shape() == [batch_size, spatial_size, spatial_size, 
                                    FLAGS.A]
      logger.info('relu_conv1 output shape: {}'.format(output.get_shape()))
    
    #----- Primary Capsules -----#
    with tf.variable_scope('primary_caps') as scope:
      pose = slim.conv2d(output, 
      num_outputs=FLAGS.B * 16, 
      kernel_size=[1, 1], 
      stride=1, 
      padding='VALID', 
      scope='pose', 
      activation_fn=None)
      activation = slim.conv2d(
          output, 
          num_outputs=FLAGS.B, 
          kernel_size=[1, 1], 
          stride=1, 
          padding='VALID', 
          scope='activation', 
          activation_fn=tf.nn.sigmoid)

      spatial_size = int(pose.get_shape()[1])
      pose = tf.reshape(pose, shape=[batch_size, spatial_size, spatial_size, 
                                     FLAGS.B, 16], name='pose')
      activation = tf.reshape(
          activation, 
          shape=[batch_size, spatial_size, spatial_size, FLAGS.B, 1], 
          name="activation")
      
      assert pose.get_shape() == [batch_size, spatial_size, spatial_size, 
                                  FLAGS.B, 16]
      assert activation.get_shape() == [batch_size, spatial_size, spatial_size,
                                        FLAGS.B, 1]
      logger.info('primary_caps pose shape: {}'.format(pose.get_shape()))
      logger.info('primary_caps activation shape {}'
                  .format(activation.get_shape()))
      
      tf.summary.histogram("activation", activation)
       
    #----- Conv Caps 1 -----#
    # activation_in: (64, 7, 7, 8, 1) 
    # pose_in: (64, 7, 7, 16, 16) 
    # activation_out: (64, 5, 5, 32, 1)
    # pose_out: (64, 5, 5, 32, 16)
    activation, pose = lyr.conv_caps(
        activation_in = activation, 
        pose_in = pose, 
        kernel = 3, 
        stride = 2, 
        ncaps_out = FLAGS.C, 
        name = 'lyr.conv_caps1', 
        weights_regularizer = weights_regularizer)
    
    #----- Conv Caps 2 -----#
    # activation_in: (64, 7, 7, 8, 1) 
    # pose_in: (64, 7, 7, 16, 1) 
    # activation_out: (64, 5, 5, 32, 1)
    # pose_out: (64, 5, 5, 32, 16)
    activation, pose = lyr.conv_caps(
        activation_in = activation, 
        pose_in = pose, 
        kernel = 3, 
        stride = 1, 
        ncaps_out = FLAGS.D, 
        name = 'lyr.conv_caps2',
        weights_regularizer = weights_regularizer,
        drop_rate = drop_rate)
    
    #----- Class Caps -----#
    # activation_in: (64, 5, 5, 32, 1)
    # pose_in: (64, 5, 5, 32, 16)
    # activation_out: (64, 5)
    # pose_out: (64, 5, 16) 
    activation_out, pose_out = lyr.fc_caps(
        activation_in = activation,
        pose_in = pose,
        ncaps_out = num_classes,
        name = 'class_caps',
        weights_regularizer = weights_regularizer)
    
  return {'scores': activation_out, 'pose_out': pose_out}


#------------------------------------------------------------------------------
# BASELINE CNN FOR SMALLNORB
#------------------------------------------------------------------------------
def build_arch_baseline(input, is_train: bool, num_classes: int):
  """Spread loss.
  
  "As the baseline for our experiments on generalization to novel viewpoints 
  we train a CNN which has two convolutional layers with 32 and 64 channels
  respectively. Both layers have a kernel size of 5 and a stride of 1 with a 
  2 × 2 max pooling. The third layer is a 1024 unit fully connected layer 
  with dropout and connects to the 5-way softmax output layer. All hidden units
  use the ReLU non-linearity. We use the same image preparation for the CNN 
  baseline as described above for the capsule network. Our baseline CNN was the
  result of an extensive hyperparameter search over filter sizes, numbers of 
  channels and learning rates.
  
  See Hinton et al. "Matrix Capsules with EM Routing" equation (3).
  
  Author:
    Ashley Gritzman 19/10/2018  
  Credit:
    Adapted from Suofei Zhang's implementation on GitHub, "Matrix-Capsules-
    EM-Tensorflow"
    https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow  
  Args: 
    input: 
    is_train: 
    num_classes: 
  Returns:
    output: 
      mean loss for entire batch
      (scalar)
  """

  bias_initializer = tf.truncated_normal_initializer(
      mean=0.0, stddev=0.01)
  
  # The paper didnot mention any regularization, a common l2 regularizer to 
  # weights is added here
  weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

  logger.info('input shape: {}'.format(input.get_shape()))

  # weights_initializer=initializer,
  with slim.arg_scope([slim.conv2d, slim.fully_connected], 
    trainable=is_train, 
    biases_initializer=bias_initializer, 
    weights_regularizer=weights_regularizer):
    
    #----- Conv1 -----#
    with tf.variable_scope('relu_conv1') as scope:
      output = slim.conv2d(
          input, 
          num_outputs=32, 
          kernel_size=[5, 5], 
          stride=1, 
          padding='SAME', 
          scope=scope, 
          activation_fn=tf.nn.relu)
      output = slim.max_pool2d(output, [2, 2], scope='max_2d_layer1')
      logger.info('output shape: {}'.format(output.get_shape()))
  
    #----- Conv2 -----#
    with tf.variable_scope('relu_conv2') as scope:
      output = slim.conv2d(
          output, 
          num_outputs=64, 
          kernel_size=[5, 5], 
          stride=1, 
          padding='SAME', 
          scope=scope, 
          activation_fn=tf.nn.relu)
      output = slim.max_pool2d(output, [2, 2], scope='max_2d_layer2')
      logger.info('output shape: {}'.format(output.get_shape()))
    
    #----- FC with Dropout -----#
    output = slim.flatten(output)
    output = slim.fully_connected(
        output, 1024, 
        scope='relu_fc3', 
        activation_fn=tf.nn.relu)
    logger.info('output shape: {}'.format(output.get_shape()))
    output = slim.dropout(output, 0.5, scope='dropout')
    
    #----- FC final layer -----#
    logits = slim.fully_connected(
        output, 
        num_classes, 
        scope='final_layer', 
        activation_fn=None)
    logger.info('output shape: {}'.format(output.get_shape()))
    
    return {'scores': logits}


#------------------------------------------------------------------------------
# LOSS FUNCTIONS
#------------------------------------------------------------------------------
def spread_loss(scores, y):
  """Spread loss.
  
  "In order to make the training less sensitive to the initialization and 
  hyper-parameters of the model, we use “spread loss” to directly maximize the 
  gap between the activation of the target class (a_t) and the activation of the 
  other classes. If the activation of a wrong class, a_i, is closer than the 
  margin, m, to at then it is penalized by the squared distance to the margin."
  
  See Hinton et al. "Matrix Capsules with EM Routing" equation (3).
  
  Author:
    Ashley Gritzman 19/10/2018  
  Credit:
    Adapted from Suofei Zhang's implementation on GitHub, "Matrix-Capsules-
    EM-Tensorflow"
    https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow  
  Args: 
    scores: 
      scores for each class 
      (batch_size, num_class)
    y: 
      index of true class 
      (batch_size, 1)  
  Returns:
    loss: 
      mean loss for entire batch
      (scalar)
  """
  
  with tf.variable_scope('spread_loss') as scope:
    batch_size = int(scores.get_shape()[0])

    # AG 17/09/2018: modified margin schedule based on response of authors to 
    # questions on OpenReview.net: 
    # https://openreview.net/forum?id=HJWLfGWRb
    # "The margin that we set is: 
    # margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, step / 50000.0 - 4))
    # where step is the training step. We trained with batch size of 64."
    global_step = tf.to_float(tf.train.get_global_step())
    m_min = 0.2
    m_delta = 0.79
    m = (m_min 
         + m_delta * tf.sigmoid(tf.minimum(10.0, global_step / 50000.0 - 4)))

    num_class = int(scores.get_shape()[-1])

    y = tf.one_hot(y, num_class, dtype=tf.float32)
    
    # Get the score of the target class
    # (64, 1, 5)
    scores = tf.reshape(scores, shape=[batch_size, 1, num_class])
    # (64, 5, 1)
    y = tf.expand_dims(y, axis=2)
    # (64, 1, 5)*(64, 5, 1) = (64, 1, 1)
    at = tf.matmul(scores, y)
    
    # Compute spread loss, paper eq (3)
    loss = tf.square(tf.maximum(0., m - (at - scores)))
    
    # Sum losses for all classes
    # (64, 1, 5)*(64, 5, 1) = (64, 1, 1)
    # e.g loss*[1 0 1 1 1]
    loss = tf.matmul(loss, 1. - y)
    
    # Compute mean
    loss = tf.reduce_mean(loss)

  return loss


def cross_ent_loss(logits, y):
  """Cross entropy loss.
  
  Author:
    Ashley Gritzman 06/05/2019  
  Args: 
    logits: 
      logits for each class 
      (batch_size, num_class)
    y: 
      index of true class 
      (batch_size, 1)  
  Returns:
    loss: 
      mean loss for entire batch
      (scalar)
  """
  loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
  loss = tf.reduce_mean(loss)

  return loss



def total_loss(scores, y):
  """total_loss = spread_loss + regularization_loss.
  
  If the flag to regularize is set, the the total loss is the sum of the spread   loss and the regularization loss.
  
  Author:
    Ashley Gritzman 19/10/2018  
  Credit:
    Adapted from Suofei Zhang's implementation on GitHub, "Matrix-Capsules-
    EM-Tensorflow"
    https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow  
  Args: 
    scores: 
      scores for each class 
      (batch_size, num_class)
    y: 
      index of true class 
      (batch_size, 1)  
  Returns:
    total_loss: 
      mean total loss for entire batch
      (scalar)
  """
  
  with tf.variable_scope('total_loss') as scope:
    # spread loss
    sprd_loss = spread_loss(scores, y)

    if FLAGS.weight_reg:
      # Regularization
      regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      reg_loss = tf.add_n(regularization)
      total_loss = sprd_loss + reg_loss
      tf.summary.scalar('spread_loss', sprd_loss)
      tf.summary.scalar('regularization_loss', reg_loss)
    else:
      # No regularization
      total_loss = sprd_loss
      tf.summary.scalar('spread_loss', sprd_loss)

  return total_loss 



  











