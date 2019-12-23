import tensorflow as tf
import tensorflow_datasets as tfds
from config import FLAGS


def _floatify_and_normalize(datapoint):
  img = tf.cast(datapoint["image"], tf.float32) / 255
  return img, datapoint["label"]


def create_inputs(is_train, force_set=None):
  # does not have test
  split = "train" if is_train else "validation"
  if force_set is not None:
    split = force_set
  data = tfds.image.imagenet_resized.ImagenetResized(split=split)
  data.builder_config.size = 64
  data = data.map(_floatify_and_normalize, num_parallel_calls=FLAGS.num_threads)
  if is_train:
    data = data.shuffle(2000 + 3 * FLAGS.batch_size).batch(FLAGS.batch_size, drop_remainder=True).repeat()
  else:
    data = data.batch(FLAGS.batch_size, drop_remainder=True).repeat()
  data = data.prefetch(1)
  iterator = data.make_one_shot_iterator()
  img, lab = iterator.get_next()
  output_dict = {'image': img, 'label': lab}
  return output_dict
