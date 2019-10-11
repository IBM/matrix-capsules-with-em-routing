"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com

Credits: 
  1.  Jonathan Hui's blog, "Understanding Matrix capsules with EM Routing 
      (Based on Hinton's Capsule Networks)" 
      https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-
      Capsule-Network/
  2.  Questions and answers on OpenReview, "Matrix capsules with EM routing" 
      https://openreview.net/forum?id=HJWLfGWRb
  3.  Suofei Zhang's implementation on GitHub, "Matrix-Capsules-EM-Tensorflow" 
      https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
  4.  Guang Yang's implementation on GitHub, "CapsulesEM" 
      https://github.com/gyang274/capsulesEM
"""

# Public modules
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# My modules
from config import FLAGS
import utils as utl

# Get logger that has already been created in config.py
import daiquiri
logger = daiquiri.getLogger(__name__)


def em_routing(votes_ij, activations_i, batch_size, spatial_routing_matrix, drop_rate=0, dropout=False, dropconnect=False):
  """The EM routing between input capsules (i) and output capsules (j).
  
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  of EM routing.
  
  Author:
    Ashley Gritzman 19/10/2018
  Definitions:
    N -> number of samples in batch
    OH -> output height
    OW -> output width
    kh -> kernel height
    kw -> kernel width
    kk -> kh * kw
    i -> number of input capsules, also called "child_caps"
    o -> number of output capsules, also called "parent_caps"
    child_space -> spatial dimensions of input capsule layer i
    parent_space -> spatial dimensions of output capsule layer j
    n_channels -> number of channels in pose matrix (usually 4x4=16)
  Args: 
    votes_ij: 
      votes from capsules in layer i to capsules in layer j
      For conv layer:
        (N*OH*OW, kh*kw*i, o, 4x4)
        (64*6*6, 9*8, 32, 16)
      For FC layer:
        The kernel dimensions are equal to the spatial dimensions of the input 
        layer i, and the spatial dimensions of the output layer j are 1x1.
        (N*1*1, child_space*child_space*i, o, 4x4)
        (64, 4*4*16, 5, 16)
    activations_i: 
      activations of capsules in layer i (L)
      (N*OH*OW, kh*kw*i, 1)
      (64*6*6, 9*8, 1)
    batch_size: 
    spatial_routing_matrix: 
  Returns:
    poses_j: 
      poses of capsules in layer j (L+1)
      (N, OH, OW, o, 4x4) 
      (64, 6, 6, 32, 16)
    activations_j: 
      activations of capsules in layer j (L+1)
      (N, OH, OW, o, 1)
      (64, 6, 6, 32, 1)
  """
  if dropout or dropconnect:
    assert drop_rate > 0
  #----- Dimensions -----#
  
  # Get dimensions needed to do conversions
  N = batch_size
  votes_shape = votes_ij.get_shape().as_list()
  OH = np.sqrt(int(votes_shape[0]) / N)
  OH = int(OH)
  OW = np.sqrt(int(votes_shape[0]) / N)
  OW = int(OW)
  kh_kw_i = int(votes_shape[1])
  o = int(votes_shape[2])
  n_channels = int(votes_shape[3])
  
  # Calculate kernel size by adding up column of spatial routing matrix
  # Do this before conventing the spatial_routing_matrix to tf
  kk = int(np.sum(spatial_routing_matrix[:,0]))
  
  parent_caps = o
  child_caps = int(kh_kw_i/kk)
  
  rt_mat_shape = spatial_routing_matrix.shape
  child_space_2 = rt_mat_shape[0]
  child_space = int(np.sqrt(child_space_2))
  parent_space_2 = rt_mat_shape[1]
  parent_space = int(np.sqrt(parent_space_2))
   
  
  #----- Reshape Inputs -----#

  # conv: (N*OH*OW, kh*kw*i, o, 4x4) -> (N, OH, OW, kh*kw*i, o, 4x4)
  # FC: (N, child_space*child_space*i, o, 4x4) -> (N, 1, 1, child_space*child_space*i, output_classes, 4x4)
  votes_ij = tf.reshape(votes_ij, [N, OH, OW, kh_kw_i, o, n_channels]) 
  
  # (N*OH*OW, kh*kw*i, 1) -> (N, OH, OW, kh*kw*i, o, n_channels)
  #              (24, 6, 6, 288, 1, 1)
  activations_i = tf.reshape(activations_i, [N, OH, OW, kh_kw_i, 1, 1])
  

  #----- Betas -----#

  """
  # Initialization from Jonathan Hui [1]:
  beta_v_hui = tf.get_variable(
    name='beta_v', 
    shape=[1, 1, 1, o], 
    dtype=tf.float32,
    initializer=tf.contrib.layers.xavier_initializer())
  beta_a_hui = tf.get_variable(
    name='beta_a', 
    shape=[1, 1, 1, o], 
    dtype=tf.float32,
    initializer=tf.contrib.layers.xavier_initializer())
                              
  # AG 21/11/2018: 
  # Tried to find std according to Hinton's comments on OpenReview 
  # https://openreview.net/forum?id=HJWLfGWRb&noteId=r1lQjCAChm
  # Hinton: "We used truncated_normal_initializer and set the std so that at the 
  # start of training half of the capsules in each layer are active and half 
  # inactive (for the Primary Capsule layer where the activation is not computed 
  # through routing we use different std for activation convolution weights & 
  # for pose parameter convolution weights)."
  # 
  # std beta_v seems to control the spread of activations
  # To try and achieve what Hinton said about half active and half not active,
  # I change the std values and check the histogram/distributions in 
  # Tensorboard
  # to try and get a good spread across all values. I couldn't get this working
  # nicely.
  beta_v_hui = slim.model_variable(
    name='beta_v', 
    shape=[1, 1, 1, 1, o, 1], 
    dtype=tf.float32,
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=10.0))
  """
  beta_a = slim.model_variable(
    name='beta_a', 
    shape=[1, 1, 1, 1, o, 1], 
    dtype=tf.float32, 
    initializer=tf.truncated_normal_initializer(mean=-1000.0, stddev=500.0))
  
  # AG 04/10/2018: using slim.variable to create instead of tf.get_variable so 
  # that they get correctly placed on the CPU instead of GPU in the multi-gpu 
  # version.
  # One beta per output capsule type
  # (1, 1, 1, 1, 32, 1)
  # (N, OH, OH, i, o, n_channels)
  beta_v = slim.model_variable(
    name='beta_v', 
    shape=[1, 1, 1, 1, o, 1], 
    dtype=tf.float32,            
    initializer=tf.contrib.layers.xavier_initializer(),
    regularizer=None)
  """
  beta_a = slim.model_variable(
    name='beta_a', 
    shape=[1, 1, 1, 1, o, 1], 
    dtype=tf.float32, 
    initializer=tf.contrib.layers.xavier_initializer(),
    regularizer=None)
  """

  with tf.variable_scope("em_routing") as scope:
    # Initialise routing assignments
    # rr (1, 6, 6, 9, 8, 16) 
    #  (1, parent_space, parent_space, kk, child_caps, parent_caps)
    rr = utl.init_rr(spatial_routing_matrix, child_caps, parent_caps)
    
    # Need to reshape (1, 6, 6, 9, 8, 16) -> (1, 6, 6, 9*8, 16, 1)
    rr = np.reshape(
      rr, 
      [1, parent_space, parent_space, kk*child_caps, parent_caps, 1])
    
    # Convert rr from np to tf
    rr = tf.constant(rr, dtype=tf.float32)
    if dropconnect:
      logits = np.log(np.asarray([[drop_rate, 1 - drop_rate]]))
      dropconnect_mask = tf.cast(tf.random.categorical(logits,
                                 tf.size(rr)),
                                 tf.float32)
      dropconnect_mask = tf.reshape(dropconnect_mask, tf.shape(rr))
      rr = tf.multiply(dropconnect_mask, rr)
 
    for it in range(FLAGS.iter_routing):  
      # AG 17/09/2018: modified schedule for inverse_temperature (lambda) based
      # on Hinton's response to questions on OpenReview.net: 
      # https://openreview.net/forum?id=HJWLfGWRb
      # "the formula we used for lambda is:
      # lambda = final_lambda * (1 - tf.pow(0.95, tf.cast(i + 1, tf.float32)))
      # where 'i' is the routing iteration (range is 0-2). Final_lambda is set 
      # to 0.01."
      # final_lambda = 0.01
      final_lambda = FLAGS.final_temp
      inverse_temperature = (final_lambda * 
                             (1 - tf.pow(0.95, tf.cast(it + 1, tf.float32))))

      # AG 26/06/2018: added var_j
      activations_j, mean_j, stdv_j, var_j = m_step(
        rr,
        votes_ij,
        activations_i,
        beta_v, beta_a, 
        inverse_temperature=inverse_temperature)
      
      # We skip the e_step call in the last iteration because we only need to 
      # return the a_j and the mean from the m_stp in the last iteration to 
      # compute the output capsule activation and pose matrices  
      if it < FLAGS.iter_routing - 1:
        rr = e_step(votes_ij, 
                    activations_j, 
                    mean_j, 
                    stdv_j, 
                    var_j, 
                    spatial_routing_matrix)
        if dropconnect:
          rr = tf.multiply(dropconnect_mask, rr)

    # pose: (N, OH, OW, o, 4 x 4) via squeeze mean_j (24, 6, 6, 32, 16)
    poses_j = tf.squeeze(mean_j, axis=-3, name="poses")

    # activation: (N, OH, OW, o, 1) via squeeze o_activation is 
    # [24, 6, 6, 32, 1]
    activations_j = tf.squeeze(activations_j, axis=-3, name="activations")
    if dropout:
      logits = np.log(np.asarray([[drop_rate, 1 - drop_rate]]))
      dropout_mask = tf.cast(tf.random.categorical(logits,
                             tf.size(activations_j)), tf.float32)
      dropout_mask = tf.reshape(dropout_mask, tf.shape(activations_j))
      activations_j = tf.multiply(dropout_mask, activations_j)
  return poses_j, activations_j


def m_step(rr, votes, activations_i, beta_v, beta_a, inverse_temperature):
  """The m-step in EM routing between input capsules (i) and output capsules 
  (j).
  
  Compute the activations of the output capsules (j), and the Gaussians for the
  pose of the output capsules (j).
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  of m-step.
  
  Author:
    Ashley Gritzman 19/10/2018
    
  Args: 
    rr: 
      assignment weights between capsules in layer i and layer j
      (N, OH, OW, kh*kw*i, o, 1)
      (64, 6, 6, 9*8, 16, 1)
    votes_ij: 
      votes from capsules in layer i to capsules in layer j
      For conv layer:
        (N, OH, OW, kh*kw*i, o, 4x4)
        (64, 6, 6, 9*8, 32, 16)
      For FC layer:
        The kernel dimensions are equal to the spatial dimensions of the input 
        layer i, and
        the spatial dimensions of the output layer j are 1x1.
        (N, 1, 1, child_space*child_space*i, output_classes, 4x4)
        (64, 1, 1, 4*4*16, 5, 16)
    activations_i: 
      activations of capsules in layer i (L)
      (N, OH, OW, kh*kw*i, o, n_channels)
      (24, 6, 6, 288, 1, 1)
    beta_v: 
      Trainable parameters in computing cost 
      (1, 1, 1, 1, 32, 1)
    beta_a: 
      Trainable parameters in computing next level activation 
      (1, 1, 1, 1, 32, 1)
    inverse_temperature: lambda, increase over each iteration by the caller
    
  Returns:
    activations_j: 
      activations of capsules in layer j (L+1)
      (N, OH, OW, 1, o, 1)
      (64, 6, 6, 1, 32, 1)
    mean_j: 
      mean of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    stdv_j: 
      standard deviation of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    var_j: 
      variance of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
  """

  with tf.variable_scope("m_step") as scope:
    
    rr_prime = rr * activations_i
    rr_prime = tf.identity(rr_prime, name="rr_prime")

    # rr_prime_sum: sum over all input capsule i
    rr_prime_sum = tf.reduce_sum(rr_prime, 
                                 axis=-3, 
                                 keepdims=True, 
                                 name='rr_prime_sum')
    
    # AG 13/12/2018: normalise amount of information
    # The amount of information given to parent capsules is very different for 
    # the final "class-caps" layer. Since all the spatial capsules give output 
    # to just a few class caps, they receive a lot more information than the 
    # convolutional layers. So in order for lambda and beta_v/beta_a settings to 
    # apply to this layer, we must normalise the amount of information.
    # activ from convcaps1 to convcaps2 (64*5*5, 144, 16, 1) 144/16 = 9 info
    # (N*OH*OW, kh*kw*i, o, 1)
    # activ from convcaps2 to classcaps (64, 1, 1, 400, 5, 1) 400/5 = 80 info
    # (N, 1, 1, IH*IW*i, n_classes, 1)
    child_caps = float(rr_prime.get_shape().as_list()[-3])
    parent_caps = float(rr_prime.get_shape().as_list()[-2])
    ratio_child_to_parent =  child_caps/parent_caps
    layer_norm_factor = 100/ratio_child_to_parent
    # logger.info("ratio_child_to_parent: {}".format(ratio_child_to_parent))
    # rr_prime_sum = rr_prime_sum/ratio_child_to_parent

    # mean_j: (24, 6, 6, 1, 32, 16)
    mean_j_numerator = tf.reduce_sum(rr_prime * votes, 
                                     axis=-3, 
                                     keepdims=True, 
                                     name="mean_j_numerator")
    mean_j = tf.div(mean_j_numerator, 
                    rr_prime_sum + FLAGS.epsilon, 
                    name="mean_j")
    
    #----- AG 26/06/2018 START -----#
    # Use variance instead of standard deviation, because the sqrt seems to 
    # cause NaN gradients during backprop.
    # See original implementation from Suofei below
    var_j_numerator = tf.reduce_sum(rr_prime * tf.square(votes - mean_j), 
                                    axis=-3, 
                                    keepdims=True, 
                                    name="var_j_numerator")
    var_j = tf.div(var_j_numerator, 
                   rr_prime_sum + FLAGS.epsilon, 
                   name="var_j")
    
    # Set the minimum variance (note: variance should always be positive)
    # This should allow me to remove the FLAGS.epsilon safety from log and div 
    # that follow
    #var_j = tf.maximum(var_j, FLAGS.epsilon)
    #var_j = var_j + FLAGS.epsilon
    
    ###################
    #var_j = var_j + 1e-5
    var_j = tf.identity(var_j + 1e-9, name="var_j_epsilon")
    ###################
    
    # Compute the stdv, but it shouldn't actually be used anywhere
    # stdv_j = tf.sqrt(var_j)
    stdv_j = None
    
    ######## layer_norm_factor
    cost_j_h = (beta_v + 0.5*tf.log(var_j)) * rr_prime_sum * layer_norm_factor
    cost_j_h = tf.identity(cost_j_h, name="cost_j_h")
    
    # ----- END ----- #
    
    """
    # Original from Suofei (reference [3] at top)
    # stdv_j: (24, 6, 6, 1, 32, 16)
    stdv_j = tf.sqrt(
      tf.reduce_sum(
        rr_prime * tf.square(votes - mean_j), axis=-3, keepdims=True
      ) / rr_prime_sum,
      name="stdv_j"
    )
    # cost_j_h: (24, 6, 6, 1, 32, 16)
    cost_j_h = (beta_v + tf.log(stdv_j + FLAGS.epsilon)) * rr_prime_sum
    """
    
    # cost_j: (24, 6, 6, 1, 32, 1)
    # activations_j_cost = (24, 6, 6, 1, 32, 1)
    # yg: This is done for numeric stability.
    # It is the relative variance between each channel determined which one 
    # should activate.
    cost_j = tf.reduce_sum(cost_j_h, axis=-1, keepdims=True, name="cost_j")
    #cost_j_mean = tf.reduce_mean(cost_j, axis=-2, keepdims=True)
    #cost_j_stdv = tf.sqrt(
    #  tf.reduce_sum(
    #    tf.square(cost_j - cost_j_mean), axis=-2, keepdims=True
    #  ) / cost_j.get_shape().as_list()[-2]
    #)
    
    # AG 17/09/2018: trying to remove normalisation
    # activations_j_cost = beta_a + (cost_j_mean - cost_j) / (cost_j_stdv)
    activations_j_cost = tf.identity(beta_a - cost_j, 
                                     name="activations_j_cost")

    # (24, 6, 6, 1, 32, 1)
    activations_j = tf.sigmoid(inverse_temperature * activations_j_cost,
                               name="sigmoid")
    
    # AG 26/06/2018: added var_j to return
    return activations_j, mean_j, stdv_j, var_j

  
# AG 26/06/2018: added var_j
def e_step(votes_ij, activations_j, mean_j, stdv_j, var_j, spatial_routing_matrix):
  """The e-step in EM routing between input capsules (i) and output capsules (j).
  
  Update the assignment weights using in routing. The output capsules (j) 
  compete for the input capsules (i).
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  of e-step.
  
  Author:
    Ashley Gritzman 19/10/2018
    
  Args: 
    votes_ij: 
      votes from capsules in layer i to capsules in layer j
      For conv layer:
        (N, OH, OW, kh*kw*i, o, 4x4)
        (64, 6, 6, 9*8, 32, 16)
      For FC layer:
        The kernel dimensions are equal to the spatial dimensions of the input 
        layer i, and the spatial dimensions of the output layer j are 1x1.
        (N, 1, 1, child_space*child_space*i, output_classes, 4x4)
        (64, 1, 1, 4*4*16, 5, 16)
    activations_j: 
      activations of capsules in layer j (L+1)
      (N, OH, OW, 1, o, 1)
      (64, 6, 6, 1, 32, 1)
    mean_j: 
      mean of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    stdv_j: 
      standard deviation of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    var_j: 
      variance of each channel in capsules of layer j (L+1)
      (N, OH, OW, 1, o, n_channels)
      (24, 6, 6, 1, 32, 16)
    spatial_routing_matrix: ???
    
  Returns:
    rr: 
      assignment weights between capsules in layer i and layer j
      (N, OH, OW, kh*kw*i, o, 1)
      (64, 6, 6, 9*8, 16, 1)
  """
  
  with tf.variable_scope("e_step") as scope:
    
    # AG 26/06/2018: changed stdv_j to var_j
    o_p_unit0 = - tf.reduce_sum(
      tf.square(votes_ij - mean_j, name="num") / (2 * var_j), 
      axis=-1, 
      keepdims=True, 
      name="o_p_unit0")
    
    o_p_unit2 = - 0.5 * tf.reduce_sum(
      tf.log(2*np.pi * var_j), 
      axis=-1, 
      keepdims=True, 
      name="o_p_unit2"
    )

    # (24, 6, 6, 288, 32, 1)
    o_p = o_p_unit0 + o_p_unit2
    zz = tf.log(activations_j + FLAGS.epsilon) + o_p
    
    # AG 13/11/2018: New implementation of normalising across parents
    #----- Start -----#
    zz_shape = zz.get_shape().as_list()
    batch_size = zz_shape[0]
    parent_space = zz_shape[1]
    kh_kw_i = zz_shape[3]
    parent_caps = zz_shape[4]
    kk = int(np.sum(spatial_routing_matrix[:,0]))
    child_caps = int(kh_kw_i / kk)
    
    zz = tf.reshape(zz, [batch_size, parent_space, parent_space, kk, 
                         child_caps, parent_caps])
    
    """
    # In un-log space
    with tf.variable_scope("to_sparse_unlog") as scope:
      zz_unlog = tf.exp(zz)
      #zz_sparse_unlog = utl.to_sparse(zz_unlog, spatial_routing_matrix, 
      # sparse_filler=1e-15)
      zz_sparse_unlog = utl.to_sparse(
          zz_unlog, 
          spatial_routing_matrix, 
          sparse_filler=0.0)
      # maybe this value should be even lower 1e-15
      zz_sparse_log = tf.log(zz_sparse_unlog + 1e-15) 
      zz_sparse = zz_sparse_log
    """

    
    # In log space
    with tf.variable_scope("to_sparse_log") as scope:
      # Fill the sparse matrix with the smallest value in zz (at least -100)
      sparse_filler = tf.minimum(tf.reduce_min(zz), -100)
#       sparse_filler = -100
      zz_sparse = utl.to_sparse(
          zz, 
          spatial_routing_matrix, 
          sparse_filler=sparse_filler)
  
    
    with tf.variable_scope("softmax_across_parents") as scope:
      rr_sparse = utl.softmax_across_parents(zz_sparse, spatial_routing_matrix)
    
    with tf.variable_scope("to_dense") as scope:
      rr_dense = utl.to_dense(rr_sparse, spatial_routing_matrix)
      
    rr = tf.reshape(
        rr_dense, 
        [batch_size, parent_space, parent_space, kh_kw_i, parent_caps, 1])
    #----- End -----#

    # AG 02/11/2018
    # In response to a question on OpenReview, Hinton et al. wrote the 
    # following:
    # "The gradient flows through EM algorithm. We do not use stop gradient. A 
    # routing of 3 is like a 3 layer network where the weights of layers are 
    # shared."
    # https://openreview.net/forum?id=HJWLfGWRb&noteId=S1eo2P1I3Q
    
    return rr
