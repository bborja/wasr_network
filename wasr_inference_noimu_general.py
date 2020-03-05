"""
Run WASR NOIMU on any image...
Specify the image with a full name as an input argument to: --img-path example_1.jpg

This script segments provided image into three semantic regions: sky, water and obstacles.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import cv2
import scipy.io

from PIL import Image
from os import listdir
from os.path import isfile, join

import tensorflow as tf
import numpy as np

from wasr_models import wasr_NOIMU2, ImageReader, decode_labels, prepare_label

# COLOR MEANS OF IMAGES FROM MODDv1 DATASET
IMG_MEAN = np.array((148.8430, 171.0260, 162.4082), dtype=np.float32)

# Number of classes
NUM_CLASSES = 3

# Output dir, where segemntation mask is saved
SAVE_DIR = 'output/' # save directory

# Full path to the folder where images are stored
DATASET_PATH = 'test_images/'

# Path to trained weights
MODEL_WEIGHTS = 'example_weights/arm8imu3_noimu.ckpt-80000'

# Input image size. Our network expects images of resolution 512x384
IMG_SIZE = [384, 512]


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH,
                        help="Path to dataset files on which inference is performed.")
    parser.add_argument("--model-weights", type=str, default=MODEL_WEIGHTS,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--img-path", type=str, required=True,
                        help="Path to the image on which we want to run inference.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    # uncomment/set to an invalid value to run on the CPU instead of the GPU
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # Create network
    img_input = tf.placeholder(dtype=tf.uint8, shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Convert from opencv BGR to tensorflow's RGB format
    img_b, img_g, img_r = tf.split(axis=2, num_or_size_splits=3, value=img_input)

    # Join and subtract means
    img = tf.cast(tf.concat(axis=2, values=[img_r, img_g, img_b]), dtype=tf.float32)

    img -= IMG_MEAN

    # Expand first dimension
    #img = tf.expand_dims(img, dim=0) # tf 1.2
    img = tf.expand_dims(img, axis=0)

    with tf.variable_scope('', reuse=False):
        net = wasr_NOIMU2({'data': img}, is_training=False, num_classes=args.num_classes)

    # Which variables to load...
    restore_var = tf.global_variables()

    # Predictions (last layer of the decoder)
    raw_output = net.layers['fc1_voc12']

	# Upsample image to the original resolution
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(img)[1:3, ])
    #raw_output = tf.argmax(raw_output, dimension=3) # tf 1.2
    #pred = tf.expand_dims(raw_output, dim=3) # tf 1.2
    raw_output = tf.argmax(raw_output, axis=3)
    pred = tf.expand_dims(raw_output, axis=3)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)

    # create output folder/s if they dont exist yet
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Read image
    img_in = cv2.imread(os.path.join(args.dataset_path, args.img_path))

    # Run inference
    preds = sess.run(pred, feed_dict={img_input: img_in})
    # Decode segmentation mask
    msk = decode_labels(preds, num_classes=args.num_classes)

    # Save mask
    if os.path.exists(args.save_dir + 'output_mask.png'):
        cv2.imwrite(args.save_dir + 'output_mask_{}.png'.format(int(time.time())), msk[0])
    else:
        cv2.imwrite(args.save_dir + 'output_mask.png', msk[0])

    print('DONE!')
if __name__ == '__main__':
    main()
