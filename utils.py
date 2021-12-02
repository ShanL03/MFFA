"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import cv2
import os
from image_inpaint import *

import tensorflow as tf
import tensorflow.contrib.slim as slim
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def load_color_table(json_file):
    # load color table
    f= open(json_file, "r", encoding='utf-8')
    colors = json.loads(f.read())
    class_num=len(colors)
    R,G,B=[[],[],[]]
    for c in colors:
        R.append(c['color'][0])
        G.append(c['color'][1])
        B.append(c['color'][2])
    return [R,G,B]        
    
def idxmap2colormap(im_idx,color_table):
    R,G,B = color_table
    class_num = len(R)
    imR = np.zeros_like(im_idx,np.uint8)
    imG = np.zeros_like(im_idx,np.uint8)
    imB = np.zeros_like(im_idx,np.uint8)
    for i in range(class_num):
        imR[im_idx==i]=R[i]
        imG[im_idx==i]=G[i]
        imB[im_idx==i]=B[i]
    imcolor = np.dstack((imR,imG,imB))
    return imcolor

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def save_images(images, size, image_path):
  return imsave(images, size, image_path)

def imread(path,resize_wh=None, nearest_interpolate=False, grayscale = False):
#   print("#######",path)
  image = cv2.imread(path)
  if grayscale and image.shape[2]>0:
      image = image[:,:,0]
  if resize_wh is not None:
      if nearest_interpolate:
          image = cv2.resize(image,resize_wh,interpolation=cv2.INTER_NEAREST)
      else:
          image = cv2.resize(image,resize_wh)
  return image

# read from folder
def sequence_read(path_train, dir_frame, temporal_len, interval=2, resize_wh=None, nearest_interpolate=False, grayscale = False):
    file = os.path.basename(path_train)
    vname,idx = file[:-4].split('_')
    if(os.path.exists(path_train)):
        frames=[imread(path_train, resize_wh, nearest_interpolate, grayscale)]  
    else:
        path_train = os.path.join(dir_frame,vname+'_'+idx+file[-4:])
        frames=[imread(path_train, resize_wh, nearest_interpolate, grayscale)] 
    # print("$$$$$$$0",len(frames),path_train,temporal_len)
    for t in range(1,temporal_len):
        idxt = str(int(idx)-interval*t)
        patht = os.path.join(dir_frame,vname+'_'+idxt+file[-4:])
        if(os.path.exists(patht)):
            img = imread(patht, resize_wh, nearest_interpolate, grayscale)
            frames.append(img)
        else:
            # print("iamhere",t)
            break
    # print("$$$$$$$1",patht)
    # print("$$$$$$$0",len(frames),vname+'_'+idxt+file[-4:],file)
    if len(frames) == temporal_len:
        # print("&&&&&&good")
        return frames
    else:
        # print("&&&&&&bad")
        interval = -interval
        frames=[imread(path_train, resize_wh, nearest_interpolate, grayscale)]  
        for t in range(1,temporal_len):
            idxt = str(int(idx)-interval*t)
            patht = os.path.join(dir_frame,vname+'_'+idxt+file[-4:])
            if(os.path.exists(patht)):
                img = imread(patht, resize_wh, nearest_interpolate, grayscale)
                frames.append(img)
        if len(frames) == temporal_len:
            return frames
        else:
            return None

def full_sequence_read(imgfile, labelfile, temporal_len, resize_wh=None, nearest_interpolate=False, grayscale = False):
    if "EP" in os.path.basename(imgfile):
        inpaint_dir = "./get_miccai_dataset/inpaint_images"
    else:
        inpaint_dir = "../sinus_data/cadaver/inpaint_images"
    frames, labels, gt_valid_id = inpaint_image(imgfile, labelfile, inpaint_dir, temporal_len, resize_wh=resize_wh)
    return frames, labels

# def full_sequence_read(imgfile, labelfile, temporal_len, resize_wh=None):
#     syn_path = "./syn_images"
#     _imgfile = os.path.join(syn_path,os.path.basename(imgfile))
#     _labelfile = os.path.join(syn_path,os.path.basename(labelfile))
#     # print(cv2.imread(_imgfile).shape,cv2.imread(_labelfile,0).shape)
#     frames = np.reshape(cv2.imread(_imgfile),(temporal_len,resize_wh[1],resize_wh[0],3))
#     labels = np.reshape(cv2.imread(_labelfile,0),(temporal_len,resize_wh[1],resize_wh[0]))
#     # cv2.imwrite(os.path.join("./samples",os.path.basename(imgfile)),np.concatenate(frames,axis=0))
#     # cv2.imwrite(os.path.join("./samples",os.path.basename(labelfile)),np.concatenate(labels,axis=0)*255)
#     return frames, labels

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def evaluate_seg_result(result_path, label_path, save_name='test_rst.txt', cum_time=None):
    dices = []
    ious = []
    ct_dices = []
    ct_ious = []
    names=[]
    # files=os.listdir(label_path)
    files=os.listdir(result_path)
    for file in files:
        if not file.endswith(".png"):
            continue
        
        #    
        gt = cv2.imread(os.path.join(label_path,file))
         
        gt = gt[:,:,0]

        if 'EP' in file:
            gt[gt>0]=1

        ## coutour loss
        contour_mask = np.zeros_like(gt)
        try:        
            contours,_ = cv2.findContours(gt*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        except:
            _,contours,_ = cv2.findContours(gt*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(contour_mask,contours,-1,(1,1,1),20)
        #
        ct_gt = gt*contour_mask
        
        #
        output = cv2.imread(os.path.join(result_path,file))
        output = cv2.resize(output,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)
    
        output=output[:,:,1]/255
        #
        ct_output = output*contour_mask
    

        #
        if (np.count_nonzero(output)+np.count_nonzero(gt)) is 0:
            dice = 1
            iou = 1
        else:
            dice = (2*np.count_nonzero(gt*output))/(np.count_nonzero(output)+np.count_nonzero(gt)+0.000001) 
                
            iou = np.count_nonzero(gt*output)/(np.count_nonzero(output+gt)+0.000001)
        #
        if (np.count_nonzero(ct_output)+np.count_nonzero(ct_gt)) is 0:
            ct_dice = 1
            ct_iou = 1
        else:
            ct_dice = (2*np.count_nonzero(ct_gt*ct_output))/(np.count_nonzero(ct_output)+np.count_nonzero(ct_gt)+0.000001) 
                
            ct_iou = np.count_nonzero(ct_gt*ct_output)/(np.count_nonzero(ct_output+ct_gt)+0.000001)
        
        
        
        dices.append(dice)
        ious.append(iou)
        ct_dices.append(ct_dice)
        ct_ious.append(ct_iou)
        names.append(file[:-4])
    

    mean_dice = np.mean(dices)
    mean_iou = np.mean(ious)
    ct_mean_dice = np.mean(ct_dices)
    ct_mean_iou = np.mean(ct_ious)

    mean_time = np.mean(cum_time)
    num_time = len(cum_time)
    
    print("mean_dice={},mean_iou={},ct_mean_dice={},ct_mean_iou={}".format(mean_dice,mean_iou,ct_mean_dice,ct_mean_iou))
    print("mean time: {}ms".format(mean_time))
    file = open(save_name, 'w')
    file.write("mean_dice={},mean_iou={},ct_mean_dice={},ct_mean_iou={},mean_time={},num_time={}\n".format(mean_dice,mean_iou,ct_mean_dice,ct_mean_iou,mean_time,num_time))
    file.close()


def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    #import pdb;pdb.set_trace()
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])

    wmask = w00+w01+w10+w11

    return output,wmask