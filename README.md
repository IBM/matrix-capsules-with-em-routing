# Implementation of "Matrix Capsules with EM Routing"

A TensorFlow implementation of Hinton's paper [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb) by [Perry Deng](https://github.com/PerryXDeng).

E-mail: [perry.deng@mail.rit.edu](mailto:perry.deng@mail.rit.edu)

This implementation experiments with modifications to Hinton's implementation, the main ones being:

1. affine instead of linear vote calculation
2. dropout and dropconnect capsule layers
3. unsupervised background class reconstruction
4. deep residual capsule network on larger images
5. detection of adversarial patches using reconstruction networks

# Usage

**Step 1.** Download this repository with ``git`` or click the [download ZIP](https://github.com/PerryXDeng/matrix-capsules-with-em-routing/archive/master.zip) button.

```
$ git clone https://github.com/IBM/matrix-capsules-with-em-routing.git
$ cd matrix-capsules-with-em-routing
```


**Step 2.** Download [smallNORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) dataset.

```
$ chmod +x data/download.sh
$ ./data/download.sh
```

The download is 251MB which will then be unzipped to about 856MB. The six ```.mat``` files are placed in the directory ```data/smallNORB/mat```. 


**Step 3.** Set up the environment with Anaconda. (See [here](https://docs.anaconda.com/anaconda/install/linux/) for instructions on how to install Anaconda.)

With Anaconda (recommended):
```
$ conda env create -f capsenv.yml
$ conda activate capsenv
```

Without Anaconda:
```
$ pip install --requirement requirements.txt
```

**Step 4.** Generate TFRecord for train and test datasets from ```.mat``` files.

```
$ python ./data/convert_to_tfrecord.py
```

The resulting TFRecords are about 3.4GB each. The TensorFlow api employs multithreading, so this process should be fast (within a minute). If you are planning to commit to GitHub, make sure to ignore these files as they are too large to upload. The ```.tfrecord``` files for train and test datasets are placed in the ```data/smallNORB/tfrecord``` directory.  

If you receive the errors:  
```Bus error (core dumped) python ./convert_to_tfrecord.py``` or   
```Killed python ./convert_to_tfrecord.py```  
these most likely indicate that you have insufficient memory (8GB should be enough), and you should try the sharded approach.


**Step 5.** Start the training and validation on smallNORB.

```
$ python train_val.py
```

If you need to monitor the training process, open tensorboard with this command.
```
$ tensorboard --logdir=./logs
```

To get the full list of command line flags, ```python train_val.py --helpfull```

**Step 6.** Calculate test accuracy. Make sure to specify the actual path to your directory, the directory below ```"./logs/smallNORB/20190731_wip"``` is just an example.

```
$ python test.py --load_dir="./logs/smallNORB/20190731_wip"
```



# Implementation Details

If you would like more information on the implementation details, please refer to the associated [paper](https://arxiv.org/pdf/1907.00652.pdf) and [blog](https://medium.com/@ashleygritzman/available-now-open-source-implementation-of-hintons-matrix-capsules-with-em-routing-e5601825ee2a).

# Acknowledgements

1. [Jonathan Hui's blog](https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/), "Understanding Matrix capsules with EM Routing (Based on Hinton's Capsule Networks)"
2. [Questions and answers](https://openreview.net/forum?id=HJWLfGWRb) on OpenReview, "Matrix capsules with EM routing"
3. [Suofei Zhang's implementation](https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow) on GitHub, "Matrix-Capsules-EM-Tensorflow" 
4. [Guang Yang's implementation](https://github.com/gyang274/capsulesEM) on GitHub, "CapsulesEM"
5. [A. Gritzman's implementation](https://arxiv.org/pdf/1907.00652.pdf)
