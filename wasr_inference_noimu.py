"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
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

# PASCAL VOC COLOR MEANS OF IMAGES

# COLOR MEANS OF IMAGES FROM MODDv1 DATASET
IMG_MEAN = np.array((148.8430, 171.0260, 162.4082), dtype=np.float32)

SEQ_NUM = 28
NUM_CLASSES = 3 #2
SEQ_TXT = '/opt/workspace/host_storage_hdd/boat/inference_images_modd2_384_raw/seq%02d/seq%02d_inference.txt' #% (SEQ_NUM, SEQ_NUM)
SAVE_DIR = '/opt/workspace/host_storage_hdd/boat/inference_images_modd2_384_raw/seq%02d/masks_arm8imu3_noimu_reprod/' #% SEQ_NUM
DATASET_PATH = '/opt/workspace/host_storage_hdd/boat/inference_images_modd2_384_raw/'

MODEL_WEIGHTS = 'example_weights/arm8imu3_noimu.ckpt-80000'

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
    parser.add_argument("--seq", type=int, default = SEQ_NUM,
                        help="Sequence number to evaluate.")
    parser.add_argument("--seq-txt", type=str, default=SEQ_TXT,
                        help="Text sprintf to sequeunce  txt file")
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    """Create the model and start the evaluation process."""
    args = get_arguments()

    args.seq_txt = args.seq_txt % (args.seq, args.seq)
    args.save_dir = args.save_dir % (args.seq)

    # Create network
    img_input = tf.placeholder(dtype=tf.uint8, shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Convert from opencv BGR to tensorflow's RGB format
    img_b, img_g, img_r = tf.split(axis=2, num_or_size_splits=3, value=img_input)

    # Join and subtract means
    img = tf.cast(tf.concat(axis=2, values=[img_r, img_g, img_b]), dtype=tf.float32)

    img -= IMG_MEAN

    # Expand first dimension
    img = tf.expand_dims(img, dim=0)

    with tf.variable_scope('', reuse=False):
        net = wasr_NOIMU2({'data': img}, is_training=False, num_classes=args.num_classes)

    # Which variables to load...
    restore_var = tf.global_variables()

    # Predictions
    raw_output = net.layers['fc1_voc12']

    # Features at the end of the second block of convolutions...
    #middle_features = net.layers['res2c']  # net.layers['res3b3']

    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(img)[1:3, ])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3)

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

    # Get number of lines (images) in text file
    num_imgs = sum(1 for line in open(args.seq_txt))
    # Perform inferences on dataset
    f_id = open(args.seq_txt, 'r')

    alpha_param = 0.5
    counter = 1
    sum_times = 0

    # Perform inferences of MODD2 dataset
    for line in f_id:

        image_name, _ = line.strip('\r\n').split(' ')

        # read image
        img_in = cv2.imread(join(args.dataset_path, image_name))

        start_time = time.time()
        preds = sess.run(pred, feed_dict={img_input: img_in})
        elapsed_time = time.time() - start_time

        sum_times += elapsed_time
        print('Elapsed time: %.04f for image num %03d' % (elapsed_time, counter))

        # Extract prediction mask
        msk = decode_labels(preds, num_classes=args.num_classes)
        # Save generated mask
        cv2.imwrite(args.save_dir + 'mask_%03d.png' % counter, msk[0])

        counter = counter + 1

    print('Average time per image: %.05f' % (sum_times / num_imgs))

if __name__ == '__main__':
    main()
