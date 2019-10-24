"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com
"""

import tensorflow as tf
import numpy as np

# Get logger that has already been created in config.py
import daiquiri
logger = daiquiri.getLogger(__name__)

import utils as utl
import em_routing as em


def conv_caps(activation_in, 
              pose_in, 
              kernel, 
              stride, 
              ncaps_out, 
              name='conv_caps', 
              weights_regularizer=None,
              drop_rate=0,
              dropout=False,
              dropconnect=False,
              affine_voting=True,
              share_class_kernel=False):
  """Convolutional capsule layer.
  
  "The routing procedure is used between each adjacent pair of capsule layers. 
  For convolutional capsules, each capsule in layer L + 1 sends feedback only to 
  capsules within its receptive field in layer L. Therefore each convolutional 
  instance of a capsule in layer L receives at most kernel size X kernel size 
  feedback from each capsule type in layer L + 1. The instances closer to the 
  border of the image receive fewer feedbacks with corner ones receiving only 
  one feedback per capsule type in layer L + 1."
  
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  convolutional capsule layer.
  
  Author:
    Ashley Gritzman 27/11/2018
    
  Args: 
    activation_in:
      (batch_size, child_space, child_space, child_caps, 1)
      (64, 7, 7, 8, 1) 
    pose_in:
      (batch_size, child_space, child_space, child_caps, 16)
      (64, 7, 7, 8, 16) 
    kernel: 
    stride: 
    ncaps_out: depth dimension of parent capsules
    
  Returns:
    activation_out: 
      (batch_size, parent_space, parent_space, parent_caps, 1)
      (64, 5, 5, 32, 1)
    pose_out:
      (batch_size, parent_space, parent_space, parent_caps, 16)
      (64, 5, 5, 32, 16)
  """
  
  with tf.variable_scope(name) as scope:
    
    # Get shapes
    shape = pose_in.get_shape().as_list()
    batch_size = shape[0]
    child_space = shape[1]
    child_space_2 = int(child_space**2)
    child_caps = shape[3]
    parent_space = int(np.floor((child_space-kernel)/stride + 1))
    parent_space_2 = int(parent_space**2)
    parent_caps = ncaps_out
    kernel_2 = int(kernel**2)
    
    with tf.variable_scope('votes') as scope:
      # Tile poses and activations
      # (64, 7, 7, 8, 16)  -> (64, 5, 5, 9, 8, 16)
      pose_tiled, spatial_routing_matrix = utl.kernel_tile(
          pose_in, 
          kernel=kernel, 
          stride=stride)
      activation_tiled, _ = utl.kernel_tile(
          activation_in, 
          kernel=kernel, 
          stride=stride)

      # Check dimensions of spatial_routing_matrix
      assert spatial_routing_matrix.shape == (child_space_2, parent_space_2)

      # Unroll along batch_size and parent_space_2
      # (64, 5, 5, 9, 8, 16) -> (64*5*5, 9*8, 16)
      pose_unroll = tf.reshape(
          pose_tiled, 
          shape=[batch_size * parent_space_2, kernel_2 * child_caps, 16])
      activation_unroll = tf.reshape(
          activation_tiled, 
          shape=[batch_size * parent_space_2, kernel_2 * child_caps, 1])
      
      # (64*5*5, 9*8, 16) -> (64*5*5, 9*8, 32, 16)
      votes = utl.compute_votes(
          pose_unroll, 
          parent_caps, 
          weights_regularizer, 
          tag=True,
          affine_voting=affine_voting,
          share_kernel_weights_by_children_class=share_class_kernel,
          kernel_size=kernel_2)
      logger.info(name + ' votes shape: {}'.format(votes.get_shape()))

    with tf.variable_scope('routing') as scope:
      # votes (64*5*5, 9*8, 32, 16)
      # activations (64*5*5, 9*8, 1)
      # pose_out: (N, OH, OW, o, 4x4)
      # activation_out: (N, OH, OW, o, 1)
      pose_out, activation_out = em.em_routing(votes, 
                           activation_unroll, 
                           batch_size, 
                           spatial_routing_matrix,
                           drop_rate,
                           dropout,
                           dropconnect)
  
    logger.info(name + ' pose_out shape: {}'.format(pose_out.get_shape()))
    logger.info(name + ' activation_out shape: {}'
                .format(activation_out.get_shape()))

    tf.summary.histogram(name + "activation_out", activation_out)
  
  return activation_out, pose_out


def fc_caps(activation_in, 
            pose_in,
            ncaps_out, 
            name='class_caps', 
            weights_regularizer=None,
            drop_rate=0,
            dropout=False,
            dropconnect=False,
            affine_voting=True):
  """Fully connected capsule layer.
  
  "The last layer of convolutional capsules is connected to the final capsule 
  layer which has one capsule per output class." We call this layer 'fully 
  connected' because it fits these characteristics, although Hinton et al. do 
  not use this teminology in the paper.
  
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description.
  
  Author:
    Ashley Gritzman 27/11/2018
    
  Args: 
    activation_in:
      (batch_size, child_space, child_space, child_caps, 1)
      (64, 7, 7, 8, 1) 
    pose_in:
      (batch_size, child_space, child_space, child_caps, 16)
      (64, 7, 7, 8, 16) 
    ncaps_out: number of class capsules
    name: 
    weights_regularizer:
    
  Returns:
    activation_out: 
      score for each output class
      (batch_size, ncaps_out)
      (64, 5)
    pose_out:
      pose for each output class capsule
      (batch_size, ncaps_out, 16)
      (64, 5, 16)
  """
  
  with tf.variable_scope(name) as scope:
    
    # Get shapes
    shape = pose_in.get_shape().as_list()
    batch_size = shape[0]
    child_space = shape[1]
    child_caps = shape[3]

    with tf.variable_scope('v') as scope:
      # In the class_caps layer, we apply same multiplication to every spatial 
      # location, so we unroll along the batch and spatial dimensions
      # (64, 5, 5, 32, 16) -> (64*5*5, 32, 16)
      pose = tf.reshape(
          pose_in, 
          shape=[batch_size * child_space * child_space, child_caps, 16])
      activation = tf.reshape(
          activation_in, 
          shape=[batch_size * child_space * child_space, child_caps, 1], 
          name="activation")

      # (64*5*5, 32, 16) -> (65*5*5, 32, 5, 16)
      votes = utl.compute_votes(pose, ncaps_out, weights_regularizer,
                                affine_voting=affine_voting)

      # (65*5*5, 32, 5, 16)
      assert (
        votes.get_shape() == 
        [batch_size * child_space * child_space, child_caps, ncaps_out, 16])
      logger.info(name + ' votes original shape: {}'
                  .format(votes.get_shape()))

    with tf.variable_scope('coord_add') as scope:
      # (64*5*5, 32, 5, 16)
      votes = tf.reshape(
          votes, 
          [batch_size, child_space, child_space, child_caps, ncaps_out, 
           votes.shape[-1]])
      votes = coord_addition(votes)

    with tf.variable_scope('routing') as scope:
      # Flatten the votes:
      # Combine the 4 x 4 spacial dimensions to appear as one spacial dimension       # with many capsules.
      # [64*5*5, 16, 5, 16] -> [64, 5*5*16, 5, 16]
      votes_flat = tf.reshape(
          votes, 
          shape=[batch_size, child_space * child_space * child_caps, 
                 ncaps_out, votes.shape[-1]])
      activation_flat = tf.reshape(
          activation, 
          shape=[batch_size, child_space * child_space * child_caps, 1])
      
      spatial_routing_matrix = utl.create_routing_map(child_space=1, k=1, s=1)

      logger.info(name + ' votes in to routing shape: {}'
            .format(votes_flat.get_shape()))
      
      pose_out, activation_out = em.em_routing(votes_flat, 
                           activation_flat, 
                           batch_size, 
                           spatial_routing_matrix,
                           drop_rate,
                           dropout,
                           dropconnect)

    activation_out = tf.squeeze(activation_out, name="activation_out")
    pose_out = tf.squeeze(pose_out, name="pose_out")

    logger.info(name + ' activation shape: {}'
                .format(activation_out.get_shape()))
    logger.info(name + ' pose shape: {}'.format(pose_out.get_shape()))

    tf.summary.histogram("activation_out", activation_out)
      
  return activation_out, pose_out

  
def coord_addition(votes):
  """Coordinate addition for connecting the last convolutional capsule layer to   the final layer.
  
  "When connecting the last convolutional capsule layer to the final layer we do 
  not want to throw away information about the location of the convolutional 
  capsules but we also want to make use of the fact that all capsules of the 
  same type are extracting the same entity at different positions. We therefore   share the transformation matrices between different positions of the same 
  capsule type and add the scaled coordinate (row, column) of the center of the   receptive field of each capsule to the first two elements of the right-hand 
  column of its vote matrix. We refer to this technique as Coordinate Addition.   This should encourage the shared final transformations to produce values for 
  those two elements that represent the fine position of the entity relative to   the center of the capsule’s receptive field."
  
  In Suofei's implementation, they add x and y coordinates as two new dimensions   to the pose matrix i.e. from 16 to 18 dimensions. The paper seems to say that   the values are added to existing dimensions.
  
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  coordinate addition.  
  
  Author:
    Ashley Gritzman 27/11/2018
    
  Credit:
    Based on Jonathan Hui's implementation:
    https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-
    Capsule-Network/
    
  Args: 
    votes:
      (batch_size, child_space, child_space, child_caps, n_output_capsules, 16)
      (64, 5, 5, 32, 5, 16) 
      
  Returns:
    votes: 
      same size as input, with coordinate encoding added to first two elements 
      of right hand column of vote matrix
      (batch_size, parent_space, parent_space, parent_caps, 1)
      (64, 5, 5, 32, 16)
  """
  
  # get spacial dimension of votes
  height = votes.get_shape().as_list()[1]
  width = votes.get_shape().as_list()[2]
  dims = votes.get_shape().as_list()[-1]
  
  # Generate offset coordinates
  # The difference here is that the coordinate won't be exactly in the middle of 
  # the receptive field, but will be evenly spread out
  w_offset_vals = (np.arange(width) + 0.50)/float(width)
  h_offset_vals = (np.arange(height) + 0.50)/float(height)
  
  w_offset = np.zeros([width, dims]) # (5, 16)
  w_offset[:,3] = w_offset_vals
  # (1, 1, 5, 1, 1, 16)
  w_offset = np.reshape(w_offset, [1, 1, width, 1, 1, dims]) 
  
  h_offset = np.zeros([height, dims])
  h_offset[:,7] = h_offset_vals
  # (1, 5, 1, 1, 1, 16)
  h_offset = np.reshape(h_offset, [1, height, 1, 1, 1, dims]) 
  
  # Combine w and h offsets using broadcasting
  # w is (1, 1, 5, 1, 1, 16)
  # h is (1, 5, 1, 1, 1, 16)
  # together (1, 5, 5, 1, 1, 16)
  offset = w_offset + h_offset
  
  # Convent from numpy to tensor
  offset = tf.constant(offset, dtype=tf.float32)
    
  votes = tf.add(votes, offset, name="votes_with_coord_add")
  
  return votes
