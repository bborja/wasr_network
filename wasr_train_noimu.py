"""
Training script for semantic image segmentation using WASR model without IMU functionality.

This script train on MaSTr1325 marine dataset which contains 1325 images
This images should be further pre-augmented using: classical augmentation (rotation, scale, mirroring)
                                                   color augmentation
                                                   water component elastic deformation (to simulate differnt kinds of wavelets)
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import array_ops

from deeplab_resnet import wasr_NOIMU2, ImageReader, decode_labels, inv_preprocess, prepare_label

# COLOR MEANS OF IMAGES FROM MODDv1 DATASET
IMG_MEAN = np.array((148.8430, 171.0260, 162.4082), dtype=np.float32)

BATCH_SIZE = 2 #5
# Full path to the folder where images are located
DATA_DIRECTORY = '/opt/workspace/host_storage_hdd/boat/train_images_mastr_all/'
# Full path to txt file. The txt file should contain image, gt mask and imu mask in each line
# example: frames/image_name.jpg masks/image_name.png imus/image_name.png
# (lines in txt file should be pre-shuffled, since we do not perform shuffling while training)
DATA_LIST_PATH = '/opt/workspace/host_storage_hdd/boat/train_images_mastr_all/train_water_deformed.txt'

GRAD_UPDATE_EVERY = 10

IGNORE_LABEL = 4

INPUT_SIZE = '384,512'

LEARNING_RATE = 1e-6 #1e-5 #1e-4 #1e-3
MOMENTUM = 0.9

NUM_CLASSES = 3

# Number of training iterations
NUM_STEPS = 80001 #50001 #10600 #50001 #67001 #39001

POWER = 0.9

RANDOM_SEED = 1234

# Restore weights from...
RESTORE_FROM = './deeplab_resnet.ckpt'

SAVE_NUM_IMAGES = 1
# When to save checkpoint while training
SAVE_PRED_EVERY = 100
# Location where checkpoints are saved
SNAPSHOT_DIR = '/opt/workspace/host_storage_hdd/boat/weights_models/snapshots_wasr_noimu/'

WEIGHT_DECAY = 1e-6


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--grad-update-every", type=int, default=GRAD_UPDATE_EVERY,
                        help="Number of steps after which gradient update is applied.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to update the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def get_tensors_in_checkpoint_file(file_name,restore_last_bool,all_tensors=True,tensor_name=None):
    varlist=[]
    var_value =[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        if('fc' not in key):
          varlist.append(key)
          var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)

def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = list()
    # Loop all loaded tensors
    for i, tensor_name in enumerate(loaded_tensors[0]):
        # Extract tensor
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
        except:
            print('Not found: '+tensor_name)
        else: # modification
            full_var_list.append(tensor_aux)
    return full_var_list

def save(saver, sess, logdir, step):
   '''Save weights.

   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'arm8imu3_noimu.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

# Focal loss implementation...
def focal_loss_cost(labels, logits, gamma=2.0, alpha=4.0):
    # Epsilon for numerical stability
    epsilon = 1.e-9

    # Do softmax of logits and add epsilon for numerical stability
    softmax_logits = tf.add(tf.nn.softmax(logits), epsilon)

    # Get masks of each ground truth label
    mask_o = tf.cast(tf.equal(labels, 0), dtype=tf.float32)  # Mask for obstacle pixels
    mask_w = tf.cast(tf.equal(labels, 1), dtype=tf.float32)  # Mask for water pixels
    mask_s = tf.cast(tf.equal(labels, 2), dtype=tf.float32)  # Mask for sky pixels

    # Compute focal loss for each label
    # Focal loss equation: -1 * (1 - softmax_logits)**gamma * log(softmax_logits)
    fl_ce_o = -1. * mask_o * tf.log(softmax_logits[:,0]) * (1. - softmax_logits[:,0]) ** gamma  # Focal loss for obstacle pixels
    fl_ce_w = -1. * mask_w * tf.log(softmax_logits[:,1]) * (1. - softmax_logits[:,1]) ** gamma  # Focal loss for water pixels
    fl_ce_s = -1. * mask_s * tf.log(softmax_logits[:,2]) * (1. - softmax_logits[:,2]) ** gamma  # Focal loss for sky pixels

    # Reduce sum of Focal Loss (add together all focal losses)
    fl_ce = fl_ce_o + fl_ce_w + fl_ce_s

    # Reduce mean of Focal Loss (so we get one scalar value as an output)
    return tf.reduce_mean(fl_ce)

# This cost function serves for separating pixels belonging to obstacles from pixels belonging to sea/sky.
def cost_function_separate_water_obstacle(features_output, gt_mask):
    epsilon_watercost = 0.01

    # Get the shape of extracted features
    features_shape = features_output.get_shape()
    # Resize gt mask to match the extracted features shape (x,y)
    gt_mask = tf.image.resize_images(gt_mask, size=features_shape[1:3],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Create water mask. Extract only pixels belonging to the water component
    # The extracted mask should be type float so we can multiply it later in order to mask the elements
    # (1 = water, 2 = sky, 0 = obstacles)
    mask_water = tf.equal(gt_mask[:, :, :, 0], 1)  # For water component only
    mask_water = tf.expand_dims(mask_water, 3) # Add one extra dimension where features are
    mask_water = tf.cast(mask_water, dtype=tf.float32)

    # Create obstacles mask. Extract only pixels belonging to the obstacle component
    # The extracted mask should be type float so we can multiply it later in order to mask the elements
    # (1 = water, 2 = sky, 0 = obstacles)
    mask_obstacles = tf.equal(gt_mask[:, :, :, 0], 0)  # For obstacles component only
    mask_obstacles = tf.expand_dims(mask_obstacles, 3) # Add one extra dimension where features are
    mask_obstacles = tf.cast(mask_obstacles, dtype=tf.float32)

    # Get number of water pixels in each image (and for each feature channel - it should be the same)
    elements_water = tf.reduce_sum(mask_water, axis=[1, 2])
    # Get number of obstacle pixels in each image (and for each feature channel - it should be the same)
    elements_obstacles = tf.reduce_sum(mask_obstacles, axis=[1, 2])

    # Get rid of special cases
    # If there are zero obstacle pixels in an image, then set the number of total obstacle pixels in this image to one for numerical stability
    # otherwise dont change the number of total water pixels in an image
    elements_obstacles = tf.where(tf.equal(elements_obstacles, 0), tf.ones_like(elements_obstacles), elements_obstacles)
    # If there are zero water pixels in an image, then set the number of total water pixels in this image to one for numerical stability,
    # otherwise dont change the number of total water pixels in an image
    elements_water = tf.where(tf.equal(elements_water, 0), tf.ones_like(elements_water), elements_water)

    # Extract from the extracted features output only pixels belonging to the water component (by multiplying it with a water mask)
    # Values of pixels that do not belong to the water component will be set to 0
    water_pixels = features_output * mask_water #tf.multiply(features_output, mask_water)
    # Extract from the extracted features output only pixels belonging to the obstacles (by multiplying it with an obstacle mask)
    # Values of pixels that do not belong to the obstacle component will be set to 0
    obstacle_pixels = features_output * mask_obstacles #tf.multiply(features_output, mask_obstacles)

    # Calculate the mean value of water pixels (return [n x num_features matrix] of mean values, where n is batch number)
    mean_water = tf.reduce_mean(tf.divide(tf.reduce_sum(water_pixels, axis=[1, 2]),
                                          elements_water), axis=0, keep_dims=True)  # Calculate the averge water value across all images in a batch

    # Create mean water matrix where only pixels belonging to the water have mean values, other pixels are set to 0
    # The bellow two lines create a matrix of a size [batch_number, 1, 1, features_number],
    # where for feature_number i the values in each batch_number are the same; But values of different feature_number i are different
    mean_water_matrix = tf.expand_dims(mean_water, 1)
    mean_water_matrix_all = tf.expand_dims(mean_water_matrix, 1)
    # Create a matrix of size [batch_number, size_y, size_x, features_number] where pixels belonging to the water component
    # have values of an average water value across all images in a batch, while other pixels have a value 0
    mean_water_matrix_wat = mean_water_matrix_all * mask_water #tf.multiply(mean_water_matrix_all, mask_water)
    # Create a matrix of size [batch_number, size_y, size_x, features_number] where pixels belonging to the obstacle component
    # have values of an average water values across all images in a batch, while other pixels have a value 0 ???
    mean_water_matrix_obs = mean_water_matrix_all * mask_obstacles #tf.multiply(mean_water_matrix_all, mask_obstacles)

    # Calculate the variance of water pixels
    # Sum of squared differences between water elements and their mean values, divided by the number of all water elements
    var_water = tf.divide(tf.reduce_sum(tf.squared_difference(water_pixels, mean_water_matrix_wat), axis=[1, 2]),
                          elements_water)

    # Reduce the mean of water variance (This computes mean variance for each image element across all images in a batch)
    var_water = tf.reduce_mean(var_water, axis=0, keep_dims=True)

    # Calculate squared difference between obstacle pixels and mean water values for each pixel and reduce sum in x,y
    # We get a matrix of size [batch_number, features_number]
    difference_obs_wat = tf.reduce_sum(tf.squared_difference(obstacle_pixels, mean_water_matrix_obs), axis=[1, 2])

    # Compute the separation
    loss_c = tf.divide(var_water + epsilon_watercost,
                       tf.divide(difference_obs_wat, elements_obstacles) + epsilon_watercost)

    # Reduce mean to get a scalar for output
    var_cost = tf.reduce_mean(loss_c)

    return var_cost

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    """Create the model and start the training."""
    args = get_arguments()

    # Get width (w) and height (h) of an input image
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # Set random seed for reproducibility of the results
    tf.set_random_seed(args.random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch, _ = reader.dequeue(args.batch_size)  # 1st = images, 2nd = gt labels, 3rd = imu (we do not need IMU here)

    # Create network.
    with tf.variable_scope('', reuse=False):
        net = wasr_NOIMU2({'data': image_batch}, is_training=args.is_training, num_classes=args.num_classes)

    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    # The layer at which the final output is located
    raw_output = net.layers['fc1_voc12']

    # The layer from which we extract features for computing water-obstacle separation loss
    inthemiddle_output = net.layers['res4b20']

    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    #all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma']  # all trainable variables
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name] #new all trainable
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]    # only variables from the ASPP module
    arm_trainable = [v for v in all_trainable if 'arm_conv' in v.name]  # only variables from the ARM module
    ffm_trainable = [v for v in all_trainable if 'ffm_conv' in v.name]  # only variables from the FFM module
    batch_trainable = [v for v in tf.trainable_variables() if 'beta' in v.name or 'gamma' in v.name] # for batchnorms

    conv_trainable = [v for v in all_trainable if 'fc' not in v.name and 'arm_conv' not in v.name and 'ffm_conv' not in v.name]  # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]   # lr * 20.0

    # Check if everything sums up correctly. Do the neccessary assertions
    print("----")
    print("Number of all trainable:  {:d}\nNumber of fc trainable:   {:d}\nNumber of conv trainable: {:d}\nNumber of ARM trainable:  {:d}\nNumber of FFM trainable:  {:d}\n".format(len(all_trainable), len(fc_trainable), len(conv_trainable), len(arm_trainable), len(ffm_trainable)))
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable) + len(arm_trainable) + len(ffm_trainable))

    print("----")
    print("Number of fc trainable:   {:d}\nNumber of fc_w trainable: {:d}\nNumber of fc_b trainable: {:d}\n".format(len(fc_trainable), len(fc_w_trainable), len(fc_b_trainable)))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])

    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False)

    raw_gt = tf.reshape(label_proc, [-1,])

    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)

    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)

    prediction = tf.gather(raw_prediction, indices)

    # Features loss from somewhere in the middle. This forces the network to separate water pixels from obstacles
    loss_0 = cost_function_separate_water_obstacle(inthemiddle_output, label_batch)
    #loss_0 = tf.Print(loss_0, [loss_0], 'Water separation loss ')

    # Pixel-wise softmax cross entropy loss (This is the tensorflow implementation)
    ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
    ce_loss = tf.Print(ce_loss, [ce_loss], 'Default TF crossentropy loss ')

    # Weight decay losses (l2 regularization)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    added_l2_losses = 10.e-2 * tf.add_n(l2_losses) # add together all l2 losses
    added_l2_losses = tf.Print(added_l2_losses, [added_l2_losses], message="l2 losses ")

    # Focal loss
    focal_loss = focal_loss_cost(labels=gt, logits=prediction)
    focal_loss = tf.Print(focal_loss, [focal_loss], message="Focal loss ")

    # Add together all of the losses (focal loss, weight decay and water-separation loss)
    reduced_loss = added_l2_losses + focal_loss + loss_0  #(10.e-2 * loss_0) # focal loss
    #reduced_loss = added_l2_losses + ce_loss + loss_0  #(10.e-2 * loss_0) # normal cross entropy

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())

    # Learning rate modified based on the the current step
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power)) # version 1
    #learning_rate = tf.train.exponential_decay(base_lr, step_ph, 750, 0.7, staircase=True)    # version 2

    # RMSProp optimizer
    opt_conv = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=args.momentum, centered=True, name='RMSProp_conv')
    opt_sp_w = tf.train.RMSPropOptimizer(learning_rate=learning_rate * 10, decay=0.9, momentum=args.momentum, centered=True, name='RMSProp_special_w')
    opt_sp_b = tf.train.RMSPropOptimizer(learning_rate=learning_rate * 20, decay=0.9, momentum=args.momentum, centered=True, name='RMSProp_special_b')

    # Momentum optimizer (original)
    #opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    #opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    #opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    # Minimization of optimizers for specific trainable variables...
    op_c_all = opt_conv.minimize(reduced_loss, var_list=[conv_trainable, batch_trainable])
    op_spc_w = opt_sp_w.minimize(reduced_loss, var_list=[fc_w_trainable, arm_trainable, ffm_trainable])
    op_spc_b = opt_sp_b.minimize(reduced_loss, var_list=[fc_b_trainable])

    train_op = tf.group(op_c_all, op_spc_w, op_spc_b)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

    # Load variables if the checkpoint is provided.
    #if args.restore_from is not None:
    #    loader = tf.train.Saver(var_list=restore_var)
    #    load(loader, sess, args.restore_from)

    # RESTORE PARTIAL WEIGHTS (which are available)
    restored_vars = get_tensors_in_checkpoint_file(file_name=args.restore_from, restore_last_bool=args.not_restore_last)
    tensors_to_load = build_tensors_in_checkpoint_file(restored_vars)
    loader = tf.train.Saver(var_list=tensors_to_load)
    loader.restore(sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }

        loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)

        if step % args.save_pred_every == 0:
            save(saver, sess, args.snapshot_dir, step)

        duration = time.time() - start_time

        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

    # join threads
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
