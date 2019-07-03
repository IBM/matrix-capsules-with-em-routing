import tensorflow as tf
import numpy as np

import logging
import daiquiri
from time import time
import os

from numpy.random import RandomState
prng = RandomState(1234567890)

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def convert_to_tfrecord(kind: str, chunkify=False):
    """Generate TFRecord for train and test datasets from smallNORB .mat files.
    
    Combine the images, labels and additional info from .mat files into TFRecord. The chunk approach 
    has not been fully tested tested.
    The following .mat files are required (download.sh):
        1. smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat
        2. smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat
        3. smallnorb-5x46789x9x18x6x2x96x96-training-info.mat
        4. smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat
        5. smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat
        6. smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat
    
    Author:
        Ashley Gritzman 19/10/2018
    Credit:
        Modified from: https://github.com/shashanktyagi/DC-GAN-on-NORB-dataset/blob/master/src/model.py
        Original Author: Shashank Tyagi (GitHub ID: shashanktyagi)
    Args: 
        kind : 'train' or 'test'
        chunkify : kernel size
        s : stride      
    Returns:
        A 2D numpy matrix containing mapping between children capsules along the rows, 
        and parent capsules along the columns.
    """
    
    # Plan A: write dataset into one big tfrecord
    # Plan B: write dataset into manageable chuncks
    CHUNK = int(24300 * 2 / 10)  # create 10 chunks
    TOTAL_NUM_IMAGES = int(24300 * 2)

    start = time()
    
    # Set up directories
    data_store = os.path.join('./', 'data')
    dir_mat = os.path.join(data_store,'smallNORB/mat')
    dir_tfrecords = os.path.join(data_store,'smallNORB/tfrecord/')
   
    if not tf.gfile.Exists(dir_tfrecords):
        tf.gfile.MakeDirs(dir_tfrecords)
    
    #----- READ -----#
    
    if kind == "train":
        fid_images = open(dir_mat + '/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat', 'rb')
        fid_labels = open(dir_mat +'/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat', 'rb')
        fid_info = open(dir_mat + '/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat', 'rb')
    elif kind == "test":
        fid_images = open(dir_mat + '/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat', 'rb')
        fid_labels = open(dir_mat + '/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat', 'rb')
        fid_info = open(dir_mat + '/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat', 'rb')
    else:
        logger.warning('Please choose either training or testing data to preprocess.')

    logger.info('Reading data finished:' + kind)

    # Preprocessing to remove headers
    # Images
    for i in range(6):
        a = fid_images.read(4)
    # Labels
    for i in range(5):
        a = fid_labels.read(4)
    # Info
    for i in range(5):
        a = fid_info.read(4)

    for j in range(TOTAL_NUM_IMAGES // CHUNK if chunkify else 1):
        
        num_images = CHUNK if chunkify else TOTAL_NUM_IMAGES  # 24300 * 2
        num_labels = int(num_images/2) # the images are in stereo pairs, so two images correspond to one label
        
        #----- LOAD -----#
        
        # Images
        images = np.zeros((num_images, 96 * 96))
        for idx in range(num_images):
            if idx % 100 == 0:
                logger.info('Load ' + kind + ' images %d' % ((j + 1) * idx))
            temp = fid_images.read(96 * 96)
            images[idx, :] = np.fromstring(temp, 'uint8')
        
        #----- PROCESS -----#
        
        # Labels
        labels = np.fromstring(fid_labels.read(num_images * np.dtype('int32').itemsize), 'int32')
        # The images are in stereo pairs, so two images correspond to one label
        labels = np.repeat(labels, 2)   
        
        # Info
        # 1. the instance in the category (0 to 9)
        # 2. the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
        # 3. the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
        # 4. the lighting condition (0 to 5)
        info = np.fromstring(fid_info.read(4 * num_labels * np.dtype('int32').itemsize), 'int32')
        info = info.reshape(num_labels, 4)
        # Elevation
        elevation = np.array([30,35,40,45,50,55,60,65,70])
        info[:,1] = elevation[info[:,1]]
        # Azimuth
        info[:,2] = info[:,2]*10
        # The images are in stereo pairs, so two images correspond to one info vector
        info = np.repeat(info, 2, axis=0) 
           
        logger.debug('Load data %d finish. Start filling chunk %d.' % (j, j))

        # make dataset permuatation reproduceable
        perm = prng.permutation(num_images)
        images = images[perm]
        labels = labels[perm]
        info = info[perm]
        
        #----- WRITE -----#
        
        tfrecord_path = dir_tfrecords + kind + "%d.tfrecords" % j
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for i in range(num_images):
            if i % 100 == 0:
                logger.info('Write ' + kind + ' images %d' % ((j + 1) * i))
                
            img = images[i, :].tobytes()
            lab = labels[i].astype(np.int32)
            category = info[i,0].astype(np.int32)
            elevation = info[i,1].astype(np.int32)
            azimuth = info[i,2].astype(np.int32)
            lighting = info[i,3].astype(np.int32)
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lab])),
                'category': tf.train.Feature(int64_list=tf.train.Int64List(value=[category])),
                'elevation': tf.train.Feature(int64_list=tf.train.Int64List(value=[elevation])),
                'azimuth': tf.train.Feature(int64_list=tf.train.Int64List(value=[azimuth])),
                'lighting': tf.train.Feature(int64_list=tf.train.Int64List(value=[lighting]))
            }))
            writer.write(example.SerializeToString())  # Translate Chinese comment: Serialized as a string
        writer.close()

    # Should take less than a minute
    logger.info('Done writing ' + kind + '. Total time: %f' % (time() - start))
    
    # Count 
    # If not using sharded approach, then should be 48 600 in both train and test tfrecords
    logger.info("Counting...")
    c = 0
    for record in tf.python_io.tf_record_iterator(tfrecord_path):
        c += 1
    logger.info("Number of records in {} tfrecord: {}".format(kind,c))



if __name__ == "__main__":
    convert_to_tfrecord(kind='train', chunkify=False)
    convert_to_tfrecord(kind='test', chunkify=False)
    
    
