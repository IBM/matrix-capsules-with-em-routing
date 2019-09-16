import tensorflow as tf
import numpy as np
import time

batch_size = 64
vote_datapoints_dimension = [batch_size, 400, 144] # to be reduced along 2nd axis
capsule_dimension = [4, 4]


def naive_variances(votes, rr):
  rr_sum = tf.reduce_sum(rr, axis=-3, keepdims=True)
  mean = tf.div(tf.reduce_sum(votes * rr, axis=-3, keepdims=True), rr_sum)
  var = tf.div(tf.reduce_sum(tf.square(votes - mean) * rr, axis=-3, keepdims=True), rr_sum)
  return var


def fancy_variances(votes, rr):
  rr_sum = tf.reduce_sum(rr, axis=-3, keepdims=True)
  x_squared = tf.square(votes)
  e_of_xsquared = tf.div(tf.reduce_sum(x_squared * rr, axis=-3, keepdims=True), rr_sum)
  e_of_x_squared = tf.square(tf.div(tf.reduce_sum(votes * rr, axis=-3, keepdims=True), rr_sum))
  var = e_of_xsquared - e_of_x_squared
  return var


def io_treatment(votes, rr):
  # for measuring how long it takes to do GPU I/O
  return votes, rr


def inspect_output():
  votes = np.random.rand(*(vote_datapoints_dimension + capsule_dimension))
  rr = np.random.rand(*(vote_datapoints_dimension + [1, 1]))
  votes_placeholder = tf.placeholder(tf.float32, shape=vote_datapoints_dimension+capsule_dimension)
  rr_placeholder = tf.placeholder(tf.float32, shape=vote_datapoints_dimension+[1,1])
  naive = naive_variances(votes_placeholder, rr_placeholder)
  fancy = fancy_variances(votes_placeholder, rr_placeholder)
  feeddict={votes_placeholder:votes, rr_placeholder:rr}
  with tf.Session() as sesh:
    print("naive")
    print(sesh.run(naive, feed_dict=feeddict)[0])
    print("fancy")
    print(sesh.run(fancy, feed_dict=feeddict)[0])


def main():
  votes = np.random.rand(*(vote_datapoints_dimension + capsule_dimension))
  rr = np.random.rand(*(vote_datapoints_dimension + [1, 1]))
  votes_placeholder = tf.placeholder(tf.float32, shape=vote_datapoints_dimension+capsule_dimension)
  rr_placeholder = tf.placeholder(tf.float32, shape=vote_datapoints_dimension+[1,1])
  naive = naive_variances(votes_placeholder, rr_placeholder)
  fancy = fancy_variances(votes_placeholder, rr_placeholder)
  control_votes, control_rr = io_treatment(votes_placeholder, rr_placeholder)
  feeddict={votes_placeholder:votes, rr_placeholder:rr}
  with tf.Session() as sesh:
    iters = 10000
    print("naive")
    start = time.time()
    for i in range(iters):
      sesh.run(naive, feed_dict=feeddict)
    print((time.time() - start)/iters)
    print("fancy")
    start = time.time()
    for i in range(iters):
      sesh.run(fancy, feed_dict=feeddict)
    print((time.time() - start)/iters)
    print("control_1")
    start = time.time()
    for i in range(iters):
      sesh.run([control_votes, control_rr], feed_dict=feeddict)
    print((time.time() - start)/iters)
    print("control_2")
    start = time.time()
    for i in range(iters):
      sesh.run(control_rr, feed_dict=feeddict)
    print((time.time() - start)/iters)
    print("control_3")
    start = time.time()
    for i in range(iters):
      sesh.run(control_votes, feed_dict=feeddict)
    print((time.time() - start)/iters)


if __name__ == "__main__":
  main()
