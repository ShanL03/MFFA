from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from six.moves import xrange
import random
import cv2
import copy
from resnetv2 import *
from mobilenetv1 import *
from gcn_layers import *
slim = tf.contrib.slim


try:
    from .ops import *
    from .utils import *
except:    
    from ops import *
    from utils import *

# deleted commonly used code
# self.prev_output:prev_output
# self.kernel_sp: kernel_sp
# self.kernel_tp: kernel_tp

class DeepLab(object):
  def __init__(self,sess,
          input_width,
          input_height,
          crop_width,
          crop_height,
          batch_size,
          seed,
          temporal_len,
          img_pattern,
          label_pattern,
          checkpoint_dir,
          save_checkpoint_dir,
          pretrain_dir,
          datasets,
          train_dataset,
          frame_dataset,
          video_dir,
          continue_train, ###
          pass_hidden,
          seq_label,
          teacher_mode,
          disable_gcn,
          model_type,
          rnn_mode,
          learning_rate,
          # fold_id, ###
          num_class,
          color_table,
          test_video=False,
          is_train=False):

    self.sess = sess
    self.is_train=is_train
    self.test_video=test_video

    self.batch_size = batch_size
    self.num_class = num_class
    self.input_height = int(input_height)
    self.input_width = int(input_width)
    self.crop_height = int(crop_height)
    self.crop_width = int(crop_width)
    self.chn = 3
    self.temporal_len = int(temporal_len)
    self.frame_interval = 3

    self.learning_rate=0.0005
    self.beta1=0.9
    self.seed=seed

    self.img_pattern = img_pattern
    self.label_pattern = label_pattern
    self.checkpoint_dir = checkpoint_dir
    if save_checkpoint_dir == "":
      self.save_checkpoint_dir = self.checkpoint_dir
    else:
      self.save_checkpoint_dir = save_checkpoint_dir
    self.pretrain_dir = pretrain_dir
    self.color_table = color_table

    # self.teacher_mode = teacher_mode
    # if self.teacher_mode:
    #   self.teacher_model = "deeplab_resnet"
    #   self.teacher_checkpoint_dir = "teacher_model"
    self.disable_gcn = disable_gcn

    self.model_type = model_type
    if "deeplab" in self.model_type:
      if "resnet" in self.model_type:
        self.model_name = "resnet_v2_50"
      elif "mobilenet" in self.model_type:
        self.model_name = 'MobilenetV1'
    elif self.model_type=="unet":
      self.pretrain_dir = ""
      self.model_name = ""

    ### extract valid data ###
    self.train_dataset = train_dataset
    self.frame_dataset = frame_dataset
    self.datasets = datasets
    
    self.data = []
    self.label = []
    dataset_keys = ['cf1','cf2','cf3','lc1','lc2','lc3','ep1','ep2','ep3']
    data_prefix = {'cf1': ['S01','S02','S03','S04'],'cf2':['S05','S06','S07'],'cf3':['S08','S09','S10'],\
                  'lc1':['L01'],'lc2':['L02'],'lc3':['L03'],\
                  'ep1':['EP1','EP2','EP3','EP4','EP5','EP8'],'ep2':['EP9'],'ep3':['EP10']}
    self.valid_data = []
    for key in dataset_keys:
      if key in datasets:
        for prefix in data_prefix[key]:
          self.valid_data.append(prefix)
          self.data += glob(os.path.join(self.train_dataset,"images", prefix+self.img_pattern))
          self.label += glob(os.path.join(self.train_dataset,"labels", prefix+self.label_pattern))
    self.data.sort()
    self.label.sort()

    # clean the dataset for endovis
    if 'ep' in datasets and seq_label==1:
      new_data = []
      new_label = []
      bad_endovis_list = "EP1_114000.jpg"+\
      "EP2_336262.jpg"+"EP2_51000.jpg"+\
      "EP3_1500.jpg"+"EP3_244001.jpg"+\
      "EP5_105180.jpg"+"EP5_263219.jpg"+"EP5_163680.jpg"+"EP5_27180.jpg"+\
      "EP10_30960.jpg"+"EP10_35179.jpg"+"EP10_36110.jpg"+"EP10_68469.jpg"+"EP10_80862.jpg"+"EP10_63557.jpg"+"EP10_421500.jpg"+"EP10_63307.jpg"+"EP10_63309.jpg"+"EP10_82497.jpg"+"EP10_25615.jpg"+\
      "EP10_67469.jpg"+"EP1_346156.jpg"+"EP10_148785.jpg"+"EP1_90000.jpg"+"EP4_108123.jpg"+"EP1_124500.jpg"+"EP10_144795.jpg"+"EP10_145095.jpg"+"EP10_67644.jpg"+"EP10_384000.jpg"+"EP1_190500.jpg"+\
      "EP5_261719.jpg"+"EP8_13500.jpg"+"EP8_21000.jpg"+"EP5_295173.jpg"+"EP10_107133.jpg"+"EP5_269219.jpg"+"EP10_25690.jpg"+"EP10_253313.jpg"+"EP5_249719.jpg"+"EP10_25490.jpg"+"EP5_42180.jpg"+"EP3_155165.jpg"+\
      "EP5_141180.jpg"+"EP5_21180.jpg"+"EP2_327262.jpg"+"EP2_149684.jpg"+"EP8_210000.jpg"+"EP5_118680.jpg"+"EP10_376500.jpg"+"EP10_144345.jpg"+"EP1_63000.jpg"+"EP10_81822.jpg"+"EP10_144295.jpg"+\
      "EP10_148760.jpg"+"EP10_81847.jpg"+"EP10_144720.jpg"+"EP10_148735.jpg"+"EP10_80812.jpg"+"EP5_100680.jpg"+"EP5_251219.jpg"+"EP10_144420.jpg"+"EP10_119112.jpg"+"EP10_82422.jpg"+"EP1_344656.jpg"+\
      "EP10_149285.jpg"+"EP5_111180.jpg"+"EP5_317004.jpg"+"EP3_86165.jpg"+"EP2_334762.jpg"+"EP10_228000.jpg"+"EP10_149610.jpg"+"EP2_16500.jpg"+"EP3_62165.jpg"
      for endovis_img,endovis_label in zip(self.data,self.label):
        if os.path.basename(endovis_img) not in bad_endovis_list:
          new_data.append(endovis_img)
          new_label.append(endovis_label)
      self.data = new_data
      self.label = new_label
    
    self.val_data = copy.deepcopy(self.data)
    self.val_label = copy.deepcopy(self.label)

    self.video_dir = video_dir
    if not self.is_train and not 'ep' in datasets:
      self.vcap_list = [[]]*10
      for videofile in self.valid_data:
        self.vcap_list[int(videofile[1:3])-1] = cv2.VideoCapture(os.path.join(video_dir,videofile+'.mp4'))
    ### extract valid data (end) ###

    ### rnn setting ###
    # 0: original code
    # 1: gcn+original code
    # 2: lstm
    # 3: gru
    self.rnn_mode = rnn_mode
    if self.rnn_mode==0:
      self.temporal_len = 1

    if "res4" in self.save_checkpoint_dir or "mob13" in self.save_checkpoint_dir:
      self.half_model = False
    else:
      self.half_model = True

    if "noSingle" in self.save_checkpoint_dir:
      self.use_single_loss = False
    else:
      self.use_single_loss = True

    if "noCoarse" in self.save_checkpoint_dir:
      self.use_coarse_loss = False
    else:
      self.use_coarse_loss = True

    # if "deeplab" in self.model_type:
    if 'EP' in self.valid_data[0]:
      self.h_chn = 256
      self.h_size = [math.ceil(self.crop_height/16),math.ceil(self.crop_width/16),self.h_chn]
    else:
      self.h_chn = 128
      self.h_size = [int(self.crop_height/16),int(self.crop_width/16),self.h_chn]
    self.n = self.h_size[0]*self.h_size[1]
    # self.skip_size = [int(self.input_height/4-1),int(self.input_width/4-1),32]
    if "resnet" in self.model_type:
      self.skip_size = [int(self.crop_height/4-1),int(self.crop_width/4-1),32]
    elif "mobilenet" in self.model_type:
      self.skip_size = [int(self.crop_height/4),int(self.crop_width/4),32]
    # elif self.model_type=="unet":
    #   self.h_chn = 512
    #   self.h_size = [24,24,512]
    #   self.n = self.h_size[0]*self.h_size[1]

    # calculate the kernels
    self.get_kernels()

    self.save_sample = True

    self.seq_label = seq_label
    # self.pass_hidden = pass_hidden
    # if self.pass_hidden:
    #   self.seq_label = False

    self.continue_train = continue_train
    if self.continue_train and self.is_train:
      self.learning_rate=learning_rate
      if not self.pretrain_dir=='cont_mffa':
        self.pretrain_dir = ""

    self.gcn_mode = 2
    # 0: two sp gcn blocks -> pred sub softmax
    # 1: temp gcn -> sp gcn -> pred sub softmax

    # if self.test_video or not self.seq_train:
    if self.test_video:
      self.temporal_len = 1

    self.image_dims = [self.crop_height, self.crop_width, self.chn]
    self.label_dims = [self.crop_height, self.crop_width]
    ######

    if self.rnn_mode == 0:
      self.build_model()
    elif self.rnn_mode == 1:
      if not self.is_train:
        self.build_seq_model_for_test()
      else:
        self.build_seq_model()
    # if self.teacher_mode:
    #   self.build_teacher_mode_savers()
    # else:
    #   self.build_savers()
    self.build_savers()
    self.build_sequence_augmentation()

  def get_kernels(self):

    h,w = self.h_size[0],self.h_size[0]
    n = h*w

    # meshgrid
    x = np.arange(w)
    y = np.arange(h)
    xv,yv = np.meshgrid(x,y)

    kernel = []
    for i in range(h):
        for j in range(w):

            _temp = (yv-i)**2+(xv-j)**2
            _temp = np.exp(-_temp/(5**2))
            _temp[_temp<0.5]=0
            _temp[_temp>=0.5]=1

            kernel.append(np.reshape(_temp,(1,-1)).T)

    kernel = np.concatenate(kernel,axis=1)
    #
    kernel_a = np.concatenate([kernel,np.eye(n)],axis=1)
    kernel_b = np.concatenate([np.eye(n),kernel],axis=1)
    kernel1 = np.concatenate([kernel_a,kernel_b],axis=0)
    #
    kernel1 = kernel1 + np.eye(2*n)
    self.kernel_sp = tf.constant(kernel1)

    kernel2 = np.concatenate([kernel,np.zeros((n,n))],axis=1)
    kernel2 = np.concatenate([kernel2,kernel2],axis=0)

    kernel2 = kernel2 + np.eye(2*n)
    self.kernel_tp = tf.constant(kernel2)

  ################## augmentation ##################
  def build_sequence_augmentation(self):
    sequence_dims = [None,self.input_height, self.input_width, self.chn]
    if self.seq_label:
      label_dims = [self.temporal_len, self.input_height, self.input_width, 1]
    else:
      label_dims = [self.input_height, self.input_width, 1]
        
    # augmentation modual
    self.im_raw = tf.placeholder(tf.float32,  sequence_dims, name='sequence_raw') # input is a BTHWC tensor
    self.label_raw = tf.placeholder(tf.int32, label_dims, name='label_raw')
    seed = self.seed
    def color_augment(image):
      r = image
      r/=255.
      #
      if 'EP' in self.valid_data[0]:
        r = tf.image.random_hue(r,max_delta=0.1, seed=seed)
        r = tf.image.random_brightness(r,max_delta=0.1, seed=seed)
        r = tf.image.random_saturation(r,0.7,1.3, seed=seed)
        r = tf.image.random_contrast(r,0.7,1.3, seed=seed)
      else:
        r = tf.image.random_hue(r,max_delta=0.1, seed=seed)
        r = tf.image.random_brightness(r,max_delta=0.3, seed=seed)
        r = tf.image.random_saturation(r,0.7,1.3, seed=seed)
        r = tf.image.random_contrast(r,0.7,1.3, seed=seed)
      #
      r = tf.minimum(r, 1.0)
      r = tf.maximum(r, 0.0)
      r*=255.
      return r

    def augment(image,nearest=False):
      r = image
      # STEP0: flip image
      r = tf.image.random_flip_left_right(r, seed=seed)
      r = tf.image.random_flip_up_down(r,seed=seed)
      
      # STEP1: rotate image
      if not nearest:
        r = tf.contrib.image.rotate(r,tf.random_uniform((), minval=-np.pi/180*180,maxval=np.pi/180*180,seed=seed),interpolation='BILINEAR')
      else:
        r = tf.contrib.image.rotate(r,tf.random_uniform((), minval=-np.pi/180*180,maxval=np.pi/180*180,seed=seed),interpolation='NEAREST')
     
      # STEP2: resize
      if 'EP' in self.valid_data[0]:
        r_width = tf.random_uniform([1],minval=int(0.9*self.input_width),maxval=int(1.1*self.input_width),dtype=tf.int32,seed=seed)
        r_height = tf.random_uniform([1],minval=int(0.9*self.input_height),maxval=int(1.1*self.input_height),dtype=tf.int32,seed=seed)
      else:
        r_width = tf.random_uniform([1],minval=int(0.7*self.input_width),maxval=int(1.3*self.input_width),dtype=tf.int32,seed=seed)
        r_height = tf.random_uniform([1],minval=int(0.7*self.input_height),maxval=int(1.3*self.input_height),dtype=tf.int32,seed=seed)
      
      if not nearest:
          r = tf.image.resize_images(r,tf.concat([r_height,r_width],axis=0),method=tf.image.ResizeMethod.BILINEAR)
      else:
          r = tf.image.resize_images(r,tf.concat([r_height,r_width],axis=0),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      # pad if needed
      p_width = tf.reduce_max( tf.concat([r_width, tf.constant(self.crop_width,shape=[1])],axis=0))
      p_height = tf.reduce_max(tf.concat([r_height,tf.constant(self.crop_height,shape=[1])],axis=0))
      r = tf.image.resize_image_with_crop_or_pad(r, target_height=p_height, target_width=p_width)
      
      # STEP3: crop randomly
      if 'E' in self.valid_data[0]:
        dh = tf.cast(tf.random_uniform([],minval=0,maxval=1,dtype=tf.float32,seed=seed)*tf.cast(p_height-self.crop_height,dtype=tf.float32),dtype=tf.int32)
        dw = tf.cast(tf.random_uniform([],minval=0,maxval=1,dtype=tf.float32,seed=seed)*tf.cast(p_width-self.crop_width,dtype=tf.float32),dtype=tf.int32)
      else:
        dh = tf.cast(tf.random_uniform([],minval=0,maxval=1,dtype=tf.float32,seed=seed)*tf.cast(p_height-self.crop_height,dtype=tf.float32),dtype=tf.int32)
        dw = tf.cast(tf.random_uniform([],minval=0,maxval=1,dtype=tf.float32,seed=seed)*tf.cast(p_width-self.crop_width,dtype=tf.float32),dtype=tf.int32)
      r = tf.image.crop_to_bounding_box(r, offset_height = tf.reduce_sum(dh), offset_width = tf.reduce_sum(dw),
                                            target_height=self.crop_height, target_width=self.crop_width)
      return r
    
    ## augment image
    # color augment
    im_raw = color_augment(self.im_raw)
    # flip, rotation, size augment
    im_raw = tf.reshape(tf.transpose(im_raw,perm=(1,2,3,0)),\
    [self.input_height, self.input_width, -1])
    im_raw = augment(im_raw)
    # put the augmented image into the desired shape
    self.im_aug = tf.transpose(tf.reshape(im_raw,\
    [self.crop_height, self.crop_width, self.chn, -1]),\
    perm=(3,0,1,2))

    # augment label
    if not self.seq_label:
      self.label_aug = augment(self.label_raw,nearest=True)  
    else:
      label_raw = tf.reshape(tf.transpose(self.label_raw,perm=(1,2,3,0)),\
      [self.input_height, self.input_width, self.temporal_len])
      label_raw = augment(label_raw)
      self.label_aug = tf.transpose(tf.reshape(label_raw,\
      [self.crop_height, self.crop_width, 1, self.temporal_len]),\
      perm=(3,0,1,2))
  ################## augmentation (end) ##################

  ################## build savers #################
  def build_savers(self):
    # saver
    g_vars = tf.global_variables()
    bn_moving_vars = [g for g in g_vars if 'moving_' in g.name]
    self.tvars=tf.trainable_variables()

    if not self.half_model:
      print("###### build full model...")
      self.optimize_vars = self.tvars
      self.saver = tf.train.Saver(self.tvars+bn_moving_vars,max_to_keep=1)

      if self.pretrain_dir is '':
        self.load_vars = self.tvars
        self.load_vars+=bn_moving_vars
      else:
        self.load_vars = [var for var in self.tvars if self.model_name in var.name and "biases" not in var.name]
      self.loader = tf.train.Saver(self.load_vars)  
    else:
      print("###### build half model...")
      if 'resnet' in self.model_type:
        # 'resnet_v2_50/block3' not in t.name and
        self.optimize_vars = [t for t in self.tvars if 'resnet_v2_50/block4' not in t.name]
        sub_bn_moving_vars = [g for g in g_vars if 'moving_' in g.name and 'resnet_v2_50/block4' not in g.name]
      elif "mobilenet" in self.model_type:
        self.optimize_vars = [t for t in self.tvars if 'Conv2d_9_pointwise' not in t.name and \
                                                       'Conv2d_9_depthwise' not in t.name and \
                                                       'Conv2d_10_pointwise' not in t.name and \
                                                       'Conv2d_10_depthwise' not in t.name and \
                                                       'Conv2d_11_pointwise' not in t.name and \
                                                       'Conv2d_11_depthwise' not in t.name and \
                                                       'Conv2d_12_pointwise' not in t.name and \
                                                       'Conv2d_12_depthwise' not in t.name and \
                                                       'Conv2d_13_pointwise'not in t.name and \
                                                       'Conv2d_13_depthwise' not in t.name]
        sub_bn_moving_vars = [g for g in g_vars if 'moving_' in g.name and \
                                                   'Conv2d_9_depthwise' not in g.name and \
                                                   'Conv2d_10_pointwise' not in g.name and \
                                                   'Conv2d_10_depthwise' not in g.name and \
                                                   'Conv2d_11_pointwise' not in g.name and \
                                                   'Conv2d_11_depthwise' not in g.name and \
                                                   'Conv2d_12_pointwise' not in g.name and \
                                                   'Conv2d_12_depthwise' not in g.name and \
                                                   'Conv2d_13_pointwise'not in g.name and \
                                                   'Conv2d_13_depthwise' not in g.name]
      # print(len(self.tvars),len(self.optimize_vars))
      self.saver = tf.train.Saver(self.optimize_vars+sub_bn_moving_vars,max_to_keep=1)

      # print([var.name for var in self.optimize_vars if "MobilenetV1" in var.name or "resnet_v2_50" in var.name])
      if self.pretrain_dir=='cont_mffa':
        self.load_vars = [var for var in self.optimize_vars if "MobilenetV1" in var.name or "resnet_v2_50" in var.name]
        self.load_vars+=[var for var in sub_bn_moving_vars if "MobilenetV1" in var.name or "resnet_v2_50" in var.name]
      elif self.pretrain_dir is '':
        self.load_vars = self.optimize_vars
        self.load_vars+=sub_bn_moving_vars
      else:
        self.load_vars = [var for var in self.tvars if self.model_name in var.name and "biases" not in var.name]
      self.loader = tf.train.Saver(self.load_vars)  

    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                      for v in self.optimize_vars]) # tf.trainable_variables()
    print("parameter_count =", self.sess.run(parameter_count))

    # if self.pretrain_dir is '':
    #   self.load_vars = self.tvars
    #   self.load_vars+=bn_moving_vars
    # else:
    #   self.load_vars = [var for var in self.tvars if self.model_name in var.name and "biases" not in var.name]
    # self.loader = tf.train.Saver(self.load_vars)  
  ################## build savers (end) #################
  
  # def build_teacher_mode_savers(self):
  #   # saver
  #   g_vars = tf.global_variables()
  #   bn_moving_vars = [g for g in g_vars if 'moving_' in g.name and 'teacher' not in g.name]
  #   teacher_bn_moving_vars = [g for g in g_vars if 'moving_' in g.name and 'teacher' in g.name]
    
  #   self.tvars=tf.trainable_variables()
  #   tvars = [t for t in self.tvars if 'teacher' not in t.name]
  #   teacher_tvars = [t for t in self.tvars if 'teacher' in t.name]

  #   self.saver = tf.train.Saver(tvars+bn_moving_vars,max_to_keep=1)
  #   self.optimize_vars = tvars
    
  #   if self.pretrain_dir is '':
  #     self.load_vars = tvars
  #     self.load_vars+=bn_moving_vars
  #   else:
  #     self.load_vars = [var for var in tvars if self.model_name in var.name and "biases" not in var.name]
  #   self.loader = tf.train.Saver(self.load_vars)  

  #   load_teacher_vars = teacher_tvars + teacher_bn_moving_vars
  #   self.teacher_loader = tf.train.Saver(load_teacher_vars)

  ##################################### build_model ######################################
  ########################################################################################
  def build_model(self):
    # def conv_conv_pool(x, out_dim, ks, keep_prob=False, pool=True, scope="convx2_pool_"):
    #   with tf.variable_scope(scope) as sp:
    #     x = conv2d(x,out_dim,ksize=ks,stride=1,name=scope+"conv_1a") 
    #     x = tf.nn.relu(batchnorm(x,self.is_train,scope+'bn_1a'))
    #     x = conv2d(x,out_dim,ksize=ks,stride=1,name=scope+"conv_1b") 
    #     x = tf.nn.relu(batchnorm(x,self.is_train,scope+'bn_1b'))

    #     if keep_prob:
    #       x = tf.nn.dropout(x,self.keep_prob,seed = self.seed)
        
    #     x_pool = None
    #     if pool:
    #       x_pool = slim.max_pool2d(x,[2,2])
    #   return x, x_pool
    # def conv_up(x, skip, ks, scope="convx2_up_"):
    #   with tf.variable_scope(scope) as sp:
    #     out_dim = skip.shape[-1].value
    #     x = conv2d(x,out_dim,ksize=ks[0],stride=1,name=scope+"conv_1a") 
    #     x = tf.nn.relu(batchnorm(x,self.is_train,scope+'bn_1a'))

    #     x = tf.image.resize_bilinear(x, tf.shape(skip)[1:3])
    #     x = tf.concat([x,skip],axis=-1)

    #     x,_ = conv_conv_pool(x,out_dim,ks[1],pool=False,scope="convx2_")
    #   return x

    ####################
    self.keep_prob = tf.placeholder(tf.float32)
    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='image')
    self.targets = tf.placeholder(tf.int32, [self.batch_size] + self.label_dims, name='label')
    ##################
    # if self.model_type=="unet":
    #   x = self.inputs - 127.5

    #   x1_skip,x1 = conv_conv_pool(x,32,3,scope="convx2_pool_1_") #skip: 192x192x64, 64 -> 32
    #   x2_skip,x2 = conv_conv_pool(x1,32,3,scope="convx2_pool_2_") #skip: 96x96x128, 128 -> 32
    #   x3_skip,x3 = conv_conv_pool(x2,64,3,keep_prob=True,scope="convx2_pool_3_") #skip: 48x48x256, 256 -> 64 => use this if we use only one skip
    #   # x4_skip,x4 = conv_conv_pool(x3,512,3,keep_prob=True,scope="convx2_pool_4_") #skip: 24x24x512, output: 12x12x512
    #   # x5,_ = conv_conv_pool(x4,1024,3,pool=False,scope="convx2_pool_5_") #output: 12x12x1024
    #   x4,_ = conv_conv_pool(x3,128,3,pool=False,scope="convx2_pool_4_") #output: 24x24x512, 512 -> 128

    #   # up6 = conv_up(x5, x4_skip, [2,3], scope="convx2_up_6") # 24x24x512
    #   # up7 = conv_up(up6, x3_skip, [2,3], scope="convx2_up_7") # 48x48x256
    #   up7 = conv_up(x4, x3_skip, [2,3], scope="convx2_up_7") # 48x48x256
    #   up8 = conv_up(up7, x2_skip, [2,3], scope="convx2_up_8") # 96x96x128
    #   up9 = conv_up(up8, x1_skip, [2,3], scope="convx2_up_9") # 192x192x64

    #   up9 = conv2d(up9,self.num_class,ksize=3,stride=1,name="out_conv") 
    #   up9 = tf.nn.relu(batchnorm(up9,self.is_train,'bn_out'))
    #   #################
    #   self.output_softmax = tf.nn.softmax(up9)
    #   self.output = tf.cast(tf.argmax(self.output_softmax,axis=3),tf.uint8)
    
    # elif "deeplab" in self.model_type:
    # layers
    layers = []

    h = self.inputs-127.5
    
    if "resnet" in self.model_type:
      end_points = {}
      with slim.arg_scope([slim.batch_norm],is_training=self.is_train):
          _, end_points = resnet_v2_50(h,
              num_classes=0,
              is_training=self.is_train,
              global_pool=False,
              output_stride=16,
              spatial_squeeze=False,
              reuse=None,
              scope=self.model_name)
      
      if not self.half_model:
        h = end_points['resnet_v2_50/block4']  
      else:
        h = end_points['resnet_v2_50/block3'] 
      skip = end_points['resnet_v2_50/block1/unit_3/bottleneck_v2/conv1']   
    elif "mobilenet" in self.model_type:
      end_points = {}
      with slim.arg_scope([slim.batch_norm],is_training=self.is_train):
          backbone_scope='MobilenetV1'
          _,end_points = mobilenet_v1(h,
                  num_classes=0,
                  dropout_keep_prob=1.0,
                  is_training=self.is_train,
                  min_depth=8,
                  depth_multiplier=1.0,
                  conv_defs=None,
                  prediction_fn=contrib_layers.softmax,
                  spatial_squeeze=False,
                  output_stride=16,
                  reuse=None,
                  scope=backbone_scope,
                  global_pool=False)
      
      if not self.half_model:
        h = end_points['Conv2d_13_pointwise']
      else:
        h = end_points['Conv2d_8_pointwise']
      skip = end_points['Conv2d_2_pointwise']  

    if not 'EP' in self.valid_data[0]:    
      h = tf.nn.dropout(h,self.keep_prob,seed = self.seed)
    # h = tf.nn.dropout(h,self.keep_prob,seed = self.seed)
    
    # atrous spatial pyramid pooling
    h = atrous_spatial_pyramid_pooling(h,output_stride=16,depth=self.h_chn,is_train=self.is_train)
    
    # skip connect low level features
    skip = conv2d(skip,32,ksize=1,stride=1,name="conv_skip")
    skip = tf.nn.relu(batchnorm(skip,self.is_train,'bn_skip'))
    # skip = tf.nn.relu(skip)

    # upsample*4
    h = tf.image.resize_bilinear(h, tf.shape(skip)[1:3])
    
    # concate and segment
    h = tf.concat([h,skip],axis=3)    
    
    h = separable_conv2d(h,128,ksize=3,name="conv_out1")
    h = tf.nn.relu(batchnorm(h,self.is_train,'bn_out1'))
    # h = tf.nn.relu(h)

    h = separable_conv2d(h,128,ksize=3,name="conv_out2")
    h = tf.nn.relu(h)
    
    h = separable_conv2d(h,self.num_class,ksize=3,name="conv_out3")
    # upsample
    h = tf.image.resize_bilinear(h, [self.crop_height, self.crop_width]) # [self.input_height, self.input_width]
    #
    output_logits = h
    self.output_softmax = tf.nn.softmax(output_logits)   
    
    self.output = tf.cast(tf.argmax(self.output_softmax,axis=3),tf.uint8,name='outputs')
    
    #################
    if self.is_train:

      #loss
      K=self.num_class
      label_map = tf.one_hot(tf.cast(self.targets,tf.int32),K)
      flat_label = tf.reshape(label_map,[-1,K])
      flat_out = tf.reshape(self.output_softmax,[-1,K])
      self.seg_loss = tf.reduce_mean(tf.multiply(-flat_label,tf.log(flat_out+0.000001)))
      self.total_loss = self.seg_loss
        
      self.loss_sum = scalar_summary("loss", self.seg_loss)
      self.val_loss_sum = scalar_summary("val_loss", self.seg_loss)
  ##################################### build_model (end) ######################################

  ############################### build_seq_model: for training ##########################
  ########################################################################################
  # def global_matching(self, x, y):
  #   def distance(p, q):
  #     norm = tf.norm(p-q,axis=-1,keepdims=True)
  #     res = 1. - (2. / (1. + tf.math.exp(norm)))
  #     return res

  #   batch_size = x.get_shape()[0].value
  #   xh = x.get_shape()[1].value
  #   xw = x.get_shape()[2].value

  #   output = []
  #   for i in range(batch_size):
  #     output.append(distance(x[i:i+1,:,:,:], y[i:i+1,:,:,:]))
  #   return tf.concat(output,axis=0)

  def build_seq_model(self):

    ## build_seq_model
    # ##== the conv*2+pool block for unet ==##
    # def conv_conv_pool(x, out_dim, ks, keep_prob=False, pool=True, scope="convx2_pool_"):
    #   with tf.variable_scope(scope) as sp:
    #     x = conv2d(x,out_dim,ksize=ks,stride=1,name=scope+"conv_1a") 
    #     x = tf.nn.relu(batchnorm(x,self.is_train,scope+'bn_1a'))
    #     x = conv2d(x,out_dim,ksize=ks,stride=1,name=scope+"conv_1b") 
    #     x = tf.nn.relu(batchnorm(x,self.is_train,scope+'bn_1b'))

    #     if keep_prob:
    #       x = tf.nn.dropout(x,self.keep_prob,seed = self.seed)
        
    #     x_pool = None
    #     if pool:
    #       x_pool = slim.max_pool2d(x,[2,2])
    #   return x, x_pool
    # def conv_up(x, skip, ks, scope="convx2_up_"):
    #   with tf.variable_scope(scope) as sp:
    #     out_dim = skip.shape[-1].value
    #     x = conv2d(x,out_dim,ksize=ks[0],stride=1,name=scope+"conv_1a") 
    #     x = tf.nn.relu(batchnorm(x,self.is_train,scope+'bn_1a'))

    #     x = tf.image.resize_bilinear(x, tf.shape(skip)[1:3])
    #     x = tf.concat([x,skip],axis=-1)

    #     x,_ = conv_conv_pool(x,out_dim,ks[1],pool=False,scope="convx2_")
    #   return x
    # ##====##

    ## build_seq_model
    def single_model(inputs, prev_h=None):
      # if self.model_type=="unet":
      #   x = inputs - 127.5

      #   x1_skip,x1 = conv_conv_pool(x,32,3,scope="convx2_pool_1_") #skip: 192x192x64
      #   x2_skip,x2 = conv_conv_pool(x1,32,3,scope="convx2_pool_2_") #skip: 96x96x128
      #   x3_skip,x3 = conv_conv_pool(x2,64,3,keep_prob=True,scope="convx2_pool_3_") #skip: 48x48x256
      #   # x4_skip,x4 = conv_conv_pool(x3,512,3,keep_prob=True,scope="convx2_pool_4_") #skip: 24x24x512, output: 12x12x512
      #   # x5,_ = conv_conv_pool(x4,1024,3,pool=False,scope="convx2_pool_5_") #output: 12x12x1024
      #   x4,_ = conv_conv_pool(x3,128,3,pool=False,scope="convx2_pool_4_") #output: 24x24x512

      #   # gcn
      #   # h: batch_size x [h_size]
      #   _h = tf.expand_dims(tf.reshape(x4,[self.batch_size,-1,self.h_chn]),axis=1)
      #   # prev_h = None
      #   if prev_h is not None:
      #     if self.gcn_mode==0:
      #       _h = tf.concat([_h,prev_h],axis=2)
      #       kernel_sp = self.kernel_sp
      #       kernel_tp = None
      #     elif self.gcn_mode==1:
      #       kernel_sp = self.kernel_sp[:self.n,:self.n]
      #       _h = tf.concat([_h,prev_h],axis=2)
      #       kernel_tp = self.kernel_tp
      #     elif self.gcn_mode==2:
      #       kernel_sp = self.kernel_sp[:self.n,:self.n]
      #       _h = tf.concat([prev_h,_h],axis=1)
      #       kernel_tp = None
      #   else:
      #     kernel_sp = self.kernel_sp[:self.n,:self.n]
      #     kernel_tp = None
        
      #   # _h, sub_output_softmax = st_conv_block(_h, kernel_sp, kernel_tp, [self.h_chn,self.h_chn,self.h_chn,3], \
      #   #                           self.keep_prob, h_size=self.h_size, is_train=self.is_train, scope="gcn", gcn_mode=self.gcn_mode)
      #   _h,_ = st_conv_block(_h, kernel_sp, kernel_tp, [self.h_chn,self.h_chn,self.h_chn,3], \
      #                             self.keep_prob, h_size=self.h_size[0:2]+[self.h_chn], is_train=self.is_train, scope="gcn", gcn_mode=self.gcn_mode)
      #   _h = _h[:,:,:self.n,:]
      #   # sub_output_softmax = sub_output_softmax[:,0,:self.n,:]
      #   # sub_output_softmax = tf.reshape(sub_output_softmax,[self.batch_size,self.h_size[0],self.h_size[1],3])
      #   x4 = tf.reshape(_h[:,0,:,:],[self.batch_size,self.h_size[0],self.h_size[1],self.h_chn])

      #   # up6 = conv_up(x5, x4_skip, [2,3], scope="convx2_up_6") # 24x24x512
      #   up7 = conv_up(x4, x3_skip, [2,3], scope="convx2_up_7") # 48x48x256
      #   up8 = conv_up(up7, x2_skip, [2,3], scope="convx2_up_8") # 96x96x128
      #   up9 = conv_up(up8, x1_skip, [2,3], scope="convx2_up_9") # 192x192x64

      #   up9 = conv2d(up9,self.num_class,ksize=3,stride=1,name="out_conv") 
      #   up9 = tf.nn.relu(batchnorm(up9,self.is_train,'bn_out'))
      #   #################
      #   output_softmax = tf.nn.softmax(up9)
      #   output = tf.cast(tf.argmax(output_softmax,axis=3),tf.uint8)

      #   return _h, output_softmax, output
      
      # elif "deeplab" in self.model_type:
      h = inputs-127.5
    
      if "resnet" in self.model_type:
        end_points = {}
        with slim.arg_scope([slim.batch_norm],is_training=self.is_train):
            _, end_points = resnet_v2_50(h,
                num_classes=0,
                is_training=self.is_train,
                global_pool=False,
                output_stride=16,
                spatial_squeeze=False,
                reuse=None,
                scope=self.model_name)
        if not self.half_model:
          h = end_points['resnet_v2_50/block4']  
        else:
          h = end_points['resnet_v2_50/block3'] 
        skip = end_points['resnet_v2_50/block1/unit_3/bottleneck_v2/conv1']   
      elif "mobilenet" in self.model_type:
        end_points = {}
        with slim.arg_scope([slim.batch_norm],is_training=self.is_train):
            backbone_scope='MobilenetV1'
            _,end_points = mobilenet_v1(h,
                    num_classes=0,
                    dropout_keep_prob=1.0,
                    is_training=self.is_train,
                    min_depth=8,
                    depth_multiplier=1.0,
                    conv_defs=None,
                    prediction_fn=contrib_layers.softmax,
                    spatial_squeeze=False,
                    output_stride=16,
                    reuse=None,
                    scope=backbone_scope,
                    global_pool=False)
        if not self.half_model:
          h = end_points['Conv2d_13_pointwise'] 
        else:
          h = end_points['Conv2d_8_pointwise']   
        skip = end_points['Conv2d_2_pointwise']  

      if not 'EP' in self.valid_data[0]:    
        h = tf.nn.dropout(h,self.keep_prob,seed = self.seed)
      
      # atrous spatial pyramid pooling
      h = atrous_spatial_pyramid_pooling(h, output_stride=16, depth=self.h_chn,is_train=self.is_train)
      
      # skip connect low level features
      skip = conv2d(skip,32,ksize=1,stride=1,name="conv_skip") 
      skip = tf.nn.relu(batchnorm(skip,self.is_train,'bn_skip'))

      # gcn
      if self.temporal_len == 1:
        _,h,_ = single_gcn(h,prev_h=prev_h,output_sub=False)

      # upsample*4
      h = tf.image.resize_bilinear(h, tf.shape(skip)[1:3])
      
      # concate and segment
      h = tf.concat([h,skip],axis=3)    
      
      h = separable_conv2d(h,128,ksize=3,name="conv_out1")
      h = tf.nn.relu(batchnorm(h,self.is_train,'bn_out1'))

      h = separable_conv2d(h,128,ksize=3,name="conv_out2")
      h = tf.nn.relu(h)
      
      h = separable_conv2d(h,self.num_class,ksize=3,name="conv_out3")
      # upsample
      h = tf.image.resize_bilinear(h, [self.crop_height, self.crop_width]) # [self.input_height, self.input_width]
      
      output_softmax = tf.nn.softmax(h)
      output = tf.cast(tf.argmax(output_softmax,axis=3),tf.uint8)
      return output_softmax, output

    ## build_seq_model
    def deeplab_decoder(skip,h):
      # if "unet" == self.model_type:
      #   x1_skip,x2_skip,x3_skip=skip
      #   up7 = conv_up(h, x3_skip, [2,3], scope="convx2_up_7") # 96x96x128
      #   up8 = conv_up(up7, x2_skip, [2,3], scope="convx2_up_8") # 96x96x128
      #   up9 = conv_up(up8, x1_skip, [2,3], scope="convx2_up_9") # 192x192x64

      #   up9 = conv2d(up9,self.num_class,ksize=3,stride=1,name="out_conv") 
      #   up9 = tf.nn.relu(batchnorm(up9,self.is_train,'bn_out'))
      #   output_softmax = tf.nn.softmax(up9)
      #   output = tf.cast(tf.argmax(output_softmax,axis=3),tf.uint8)

      #   return output_softmax, output
      
      # elif "deeplab" in self.model_type:

      # upsample*4
      h = tf.image.resize_bilinear(h, tf.shape(skip)[1:3])
      
      # concate and segment
      h = tf.concat([h,skip],axis=3)    
      
      h = separable_conv2d(h,128,ksize=3,name="conv_out1")
      h = tf.nn.relu(batchnorm(h,self.is_train,'bn_out1'))

      h = separable_conv2d(h,128,ksize=3,name="conv_out2")
      h = tf.nn.relu(h)
      
      h = separable_conv2d(h,self.num_class,ksize=3,name="conv_out3")
      # upsample
      h = tf.image.resize_bilinear(h, [self.crop_height, self.crop_width]) # [self.input_height, self.input_width]
      
      output_softmax = tf.nn.softmax(h)
      output = tf.cast(tf.argmax(output_softmax,axis=3),tf.uint8)
      return output_softmax, output

    ## build_seq_model
    def single_enc(inputs,model_type,teacher=False):
      # if teacher:
      #   is_train=False
      # else:
      is_train=self.is_train

      # if model_type=="unet":
      #   x = inputs - 127.5

      #   x1_skip,x1 = conv_conv_pool(x,64,3,scope="convx2_pool_1_") #skip: 192x192x64
      #   x2_skip,x2 = conv_conv_pool(x1,128,3,scope="convx2_pool_2_") #skip: 96x96x128
      #   x3_skip,x3 = conv_conv_pool(x2,256,3,keep_prob=True,scope="convx2_pool_3_") #skip: 48x48x256
      #   # x4_skip,x4 = conv_conv_pool(x3,512,3,keep_prob=True,scope="convx2_pool_4_") #skip: 24x24x512, output: 12x12x512
      #   # _,x5 = conv_conv_pool(x4,1024,3,pool=False,scope="convx2_pool_5_") #output: 12x12x1024
      #   x4,_ = conv_conv_pool(x3,512,3,pool=False,scope="convx2_pool_4_")

      #   skip = [x1_skip,x2_skip,x3_skip]

      #   return skip, x4
      
      # elif "deeplab" in model_type:
      h = inputs-127.5
    
      if "resnet" in model_type:
        end_points = {}
        with slim.arg_scope([slim.batch_norm],is_training=is_train):
            _, end_points = resnet_v2_50(h,
                num_classes=0,
                is_training=is_train,
                global_pool=False,
                output_stride=16,
                spatial_squeeze=False,
                reuse=None,
                scope="resnet_v2_50")
                # scope=self.model_name)
        
        # if teacher:
        #   h = end_points['teacher/resnet_v2_50/block4']   
        #   skip = end_points['teacher/resnet_v2_50/block1/unit_3/bottleneck_v2/conv1'] 
        # else:
        if not self.half_model:
          h = end_points['resnet_v2_50/block4']
        else:
          h = end_points['resnet_v2_50/block3']      
        skip = end_points['resnet_v2_50/block1/unit_3/bottleneck_v2/conv1']  
        # end_points
        # 
        # 'resnet_v2_50/block1':64, 24, 24, 256
        # 'resnet_v2_50/block2':64, 12, 12, 512
        # 'resnet_v2_50/block3':64, 12, 12, 1024
        # 'resnet_v2_50/block4':64, 12, 12, 2048
        # print(end_points.keys())
      elif "mobilenet" in model_type:
        end_points = {}
        with slim.arg_scope([slim.batch_norm],is_training=is_train):
            backbone_scope='MobilenetV1'
            _,end_points = mobilenet_v1(h,
                    num_classes=0,
                    dropout_keep_prob=1.0,
                    is_training=is_train,
                    min_depth=8,
                    depth_multiplier=1.0,
                    conv_defs=None,
                    prediction_fn=contrib_layers.softmax,
                    spatial_squeeze=False,
                    output_stride=16,
                    reuse=None,
                    scope=backbone_scope,
                    global_pool=False)
        
        # if teacher:
        #   # h = end_points['teacher/Conv2d_13_pointwise']   
        #   h = end_points['teacher/Conv2d_8_pointwise']   
        #   skip = end_points['teacher/Conv2d_2_pointwise']
        # else:
        if not self.half_model:
          h = end_points['Conv2d_13_pointwise']  
        else: 
          h = end_points['Conv2d_8_pointwise']   
        skip = end_points['Conv2d_2_pointwise']

      if not 'EP' in self.valid_data[0]:   
        h = tf.nn.dropout(h,self.keep_prob,seed = self.seed)
      
      # atrous spatial pyramid pooling
      h = atrous_spatial_pyramid_pooling(h, output_stride=16, depth=self.h_chn,is_train=is_train)
      
      # skip connect low level features
      skip = conv2d(skip,32,ksize=1,stride=1,name="conv_skip") 
      skip = tf.nn.relu(batchnorm(skip,is_train,'bn_skip'))
      
      return skip, h

    ## build_seq_model
    def merge_prev(prev_output,prev_h):
      prev_output = tf.expand_dims(prev_output,axis=3)
      prev_output = tf.image.resize_images(prev_output,[self.h_size[0],self.h_size[1]],method=tf.image.ResizeMethod.BILINEAR)
      
      prev_output = tf.expand_dims(tf.reshape(prev_output,[self.batch_size,-1,1]),axis=1)
      return prev_output*prev_h

    ## build_seq_model
    def single_gcn(h,prev_h=None,output_sub=True,scope="gcn"):
      # h: batch_size x [h_size]
      _h = tf.reshape(h,[self.batch_size,1,-1,self.h_chn])

      if prev_h is not None:
        # global_map = self.global_matching(_h, prev_h)
        if self.gcn_mode==0:
          _h = tf.concat([_h,prev_h],axis=2)
          kernel_sp = self.kernel_sp
          kernel_tp = None
        elif self.gcn_mode==1:
          kernel_sp = self.kernel_sp[:self.n,:self.n]
          _h = tf.concat([_h,prev_h],axis=2)
          kernel_tp = self.kernel_tp
        elif self.gcn_mode==2:
          _h = tf.concat([prev_h,_h],axis=1)
      
      h, _h, sub_softmax = st_conv_block(_h, [self.h_chn,self.h_chn,self.h_chn,3], \
                                h_size=self.h_size, is_train=self.is_train, output_sub=output_sub, \
                                scope=scope, gcn_mode=self.gcn_mode)
      # other potential inputs: kernel_sp, kernel_tp, self.keep_prob, glb_map=global_map
      return _h, h, sub_softmax

    # def get_attn(h1,h2,prev_h,prev_output,prev_softmax=None,prev_image=None,scope="get_attn"):
    #   # h1: current h, h2: prev h
    #   prev_output = tf.expand_dims(prev_output,axis=3)
    #   prev_output = tf.image.resize_images(prev_output,[self.h_size[0],self.h_size[1]],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #   prev_output = tf.cast(prev_output,tf.float32)
    #   # prev_output = tf.reshape(prev_output,[self.batch_size,1,-1,1])
    #   # if prev_h is not None:
    #   #   prev_h = tf.reshape(prev_h,[self.batch_size,self.h_size[0],self.h_size[1],-1])
    #   #   concat_h = tf.concat([prev_h,prev_output],axis=3)
    #   # else:
    #   #   concat_h = prev_output
    #   prev_h = tf.reshape(prev_h,[self.batch_size,self.h_size[0],self.h_size[1],-1])
    #   concat_h = tf.concat([prev_h,prev_output],axis=3)
    #   if prev_softmax is not None:
    #     prev_softmax = tf.image.resize_images(prev_softmax,[self.h_size[0],self.h_size[1]])
    #     concat_h = tf.concat([concat_h,prev_softmax],axis=3)
    #   if prev_image is not None:
    #     prev_image = tf.image.resize_images(prev_image,[self.h_size[0],self.h_size[1]])
    #     concat_h = tf.concat([concat_h,prev_image],axis=3)

    #   chn = 64
    #   with tf.variable_scope(scope) as sp:
    #     h1 = conv2d(h1,chn,ksize=1,stride=1,name="conv_h1") 
    #     h1 = tf.nn.relu(batchnorm(h1,self.is_train,'bn_h1'))

    #     h2 = conv2d(h2,chn,ksize=1,stride=1,name="conv_h2") 
    #     h2 = tf.nn.relu(batchnorm(h2,self.is_train,'bn_h2'))

    #     h1 = tf.reshape(h1,[self.batch_size,-1,chn])
    #     h2 = tf.reshape(h2,[self.batch_size,-1,chn])

    #     attn = tf.matmul(h2,h1,transpose_b=True) * self.kernel_sp[:self.n,:self.n]
    #     idxs = tf.math.argmax(attn,axis=1)

    #     x = np.arange(self.h_size[1])
    #     y = np.arange(self.h_size[0])
    #     X, Y = tf.meshgrid(x, y)
    #     X = tf.reshape(X,[-1])
    #     Y = tf.reshape(Y,[-1])

    #     coords = []
    #     for i in range(self.batch_size):
    #       _idx = tf.expand_dims(idxs[i],axis=1)
    #       _x = tf.reshape(tf.gather_nd(X,[_idx]),[1,self.h_size[0],self.h_size[1],1])
    #       _y = tf.reshape(tf.gather_nd(Y,[_idx]),[1,self.h_size[0],self.h_size[1],1])
    #       coords.append(tf.concat([_x,_y],axis=3))
    #     coords = tf.concat(coords,axis=0)
    #     # self.coords = coords

    #     concat_h,_ = bilinear_sampler(concat_h, coords)

    #     prev_h = concat_h[:,:,:,:self.h_chn]
    #     prev_h = tf.reshape(prev_h,[self.batch_size,1,-1,self.h_chn])
        
    #     prev_output = concat_h[:,:,:,self.h_chn:self.h_chn+1]

    #     if prev_softmax is not None and prev_image is not None:
    #       prev_softmax = concat_h[:,:,:,self.h_chn+1:self.h_chn+3]
    #       prev_image = concat_h[:,:,:,self.h_chn+3:]
    #       return prev_h, tf.cast(prev_output,tf.float32), prev_softmax, prev_image
    #     elif prev_softmax is not None:
    #       prev_softmax = concat_h[:,:,:,self.h_chn+1:self.h_chn+3]
    #       return prev_h, tf.cast(prev_output,tf.float32), prev_softmax
    #     else:
    #       return prev_h, tf.cast(prev_output,tf.float32)
    #     # else:
    #     #   return None, tf.cast(concat_h,tf.float32)
    #################################

    # placeholders
    ##############
    # inputs is a 5D tensor, BTHWC
    self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.temporal_len] + self.image_dims) 
    
    if self.seq_label:
      self.targets = tf.placeholder(tf.int32, [self.batch_size, self.temporal_len] + self.label_dims)
    else:
      self.targets = tf.placeholder(tf.int32, [self.batch_size] + self.label_dims)
    self.keep_prob = tf.placeholder(tf.float32)

    # hidden state
    # if self.pass_hidden:
    #   self.prev_h = tf.placeholder(tf.float32, [self.batch_size,1,self.h_size[0]*self.h_size[1],self.h_size[2]])
    #   prev_h = self.prev_h

    #   self.prev_output = tf.placeholder(tf.float32, [self.batch_size,self.crop_height,self.crop_width])
    #   prev_output = self.prev_output
    # else:
    prev_h = None
    prev_output = None

    if self.seq_label and self.is_train:
      real_idx = self.temporal_len//2
    else:
      real_idx = 0
    ##############
    
    if self.save_sample:
      output_list = [] # for display
    if self.seq_label:
      output_softmax_list = []
      fw_output_softmax_list = []
      sub_output_softmax_list = []
      fw_sub_output_softmax_list = []

    with tf.variable_scope('',reuse=tf.AUTO_REUSE) as scope:
      # get the feature map
      if self.temporal_len > 1:
        inputs = tf.reshape(self.inputs,[self.batch_size*self.temporal_len] + self.image_dims)
        skips, hs = single_enc(inputs,self.model_type)
        if "deeplab" in self.model_type:
          skips = tf.reshape(skips,[self.batch_size,self.temporal_len]+self.skip_size)
          hs = tf.reshape(hs,[self.batch_size,self.temporal_len]+self.h_size)
        elif "unet" in self.model_type:
          skips[0] = tf.reshape(skips[0],[self.batch_size,self.temporal_len,192,192,64])
          skips[1] = tf.reshape(skips[1],[self.batch_size,self.temporal_len,96,96,128])
          skips[2] = tf.reshape(skips[2],[self.batch_size,self.temporal_len,48,48,256])
          hs = tf.reshape(hs,[self.batch_size,self.temporal_len]+self.h_size)

      # if self.teacher_mode:
      #     with tf.variable_scope('teacher') as scope:
      #       _, teacher_h = single_enc(self.inputs[:,0],"deeplab_resnet",teacher=True)

      single_output_softmax, single_output = single_model(self.inputs[:,real_idx,:,:,:])

      if self.temporal_len == 1:
        self.output_softmax = single_output_softmax
        self.output = single_output
      else:

        # if self.pass_hidden:
        #   prev_h_out = None
        #   for i in range(self.temporal_len-1,-1,-1):
        #     prev_h_out,_,_ = single_gcn(hs[:,i],prev_h=prev_h_out)
        #   self.prev_h_out = prev_h_out

        ### FW pass ###
        if self.seq_label:
          prev_h = None
          # prev_output = tf.expand_dims(self.targets[:,0],axis=3)
          # prev_output = tf.image.resize_images(prev_output,[self.h_size[0],self.h_size[1]],method=tf.image.ResizeMethod.BILINEAR)
          # _,prev_output = get_attn(hs[:,0],hs[:,1],None,self.targets[:,1],scope="get_attn")

          for i in range(self.temporal_len):
            if not self.disable_gcn:
              if prev_h is not None:
                prev_h = merge_prev(prev_output,prev_h)
              prev_h,h,sub_softmax = single_gcn(hs[:,i],prev_h=prev_h,output_sub=self.use_coarse_loss) # scope="gcn_"+str(i)
              if self.seq_label and self.use_coarse_loss:
                fw_sub_output_softmax_list.append(sub_softmax)
            else:
              h = hs[:,i]

            # # output -> prev_output
            # if "unet" == self.model_type:
            #   output_softmax, prev_output = deeplab_decoder([skips[0][:,i],skips[1][:,i],skips[2][:,i]],h)
            # elif "deeplab" in self.model_type:
            output_softmax, prev_output = deeplab_decoder(skips[:,i],h)
            #   prev_h, prev_output = get_attn(hs[:,i+1],hs[:,i],prev_h,prev_output)
            fw_output_softmax_list.append(tf.expand_dims(output_softmax,axis=1))
        
        ### BW pass ###
        # if self.pass_hidden:
        #   prev_h = self.prev_h
        # else:
        prev_h = None
        for i in range(self.temporal_len-1,-1,-1):
          if not self.disable_gcn:
            if prev_h is not None:
              prev_h = merge_prev(prev_output,prev_h)
            prev_h,h,sub_softmax = single_gcn(hs[:,i],prev_h=prev_h,output_sub=self.use_coarse_loss) #scope="gcn_"+str(self.temporal_len-1-i)
            if self.seq_label and self.use_coarse_loss:
              sub_output_softmax_list.append(sub_softmax)
          else:
            h = hs[:,i]

          # if "unet" == self.model_type:
          #   output_softmax, output = deeplab_decoder([skips[0][:,i],skips[1][:,i],skips[2][:,i]],h)
          # elif "deeplab" in self.model_type:
          output_softmax, output = deeplab_decoder(skips[:,i],h)
        
          prev_output = output
          #   prev_h, prev_output = get_attn(hs[:,i-1],hs[:,i],prev_h,prev_output)

          if self.save_sample:
            output_list.append(output)
          if self.seq_label:
            output_softmax_list.append(tf.expand_dims(output_softmax,axis=1))
        ######
    
        self.output_softmax = output_softmax
        if not self.is_train:
          self.output_h_prev = prev_h

        if self.save_sample:
          self.output_list = tf.concat(output_list[::-1],axis=2)
        if self.seq_label:
          output_softmax_list = tf.concat(output_softmax_list[1:][::-1],axis=1)
          fw_output_softmax_list = tf.concat(fw_output_softmax_list[1:],axis=1)

    ### calculate losses ###
    self.losses = []

    K = self.num_class

    ## calculate the seg loss (single image)
    if self.seq_label:
      label_map = tf.one_hot(tf.cast(self.targets[:,real_idx,:,:],tf.int32),K)
    else:
      label_map = tf.one_hot(tf.cast(self.targets,tf.int32),K)
    flat_label = tf.reshape(label_map,[-1,K])
    flat_out = tf.reshape(single_output_softmax,[-1,K])
    self.seg_loss = tf.reduce_mean(tf.multiply(-flat_label,tf.log(flat_out+0.000001)))
    if not self.disable_gcn and self.use_single_loss:
      print("###### use single loss ...")
      self.losses.append(self.seg_loss)

    if self.temporal_len > 1:
      ## seq loss
      if self.seq_label:
        seq_label_map = tf.one_hot(tf.cast(self.targets[:,:-1],tf.int32),K) #self.targets[:,:-1]
        flat_seq_label = tf.reshape(seq_label_map,[-1,K])
        flat_seq_out = tf.reshape(output_softmax_list,[-1,K])
        self.seq_seg_loss = tf.reduce_mean(tf.multiply(-flat_seq_label,tf.log(flat_seq_out+0.000001)))
        
        # if self.seq_label:'
        fw_seq_label_map = tf.one_hot(tf.cast(self.targets[:,1:],tf.int32),K) # self.targets[:,1:]
        flat_fw_seq_label = tf.reshape(fw_seq_label_map,[-1,K])
        flat_fw_seq_out = tf.reshape(fw_output_softmax_list,[-1,K])
        self.fw_seq_seg_loss = tf.reduce_mean(tf.multiply(-flat_fw_seq_label,tf.log(flat_fw_seq_out+0.000001)))
        
        self.losses.append(self.seq_seg_loss)
        self.losses.append(self.fw_seq_seg_loss)
      else:

        flat_seq_out = tf.reshape(output_softmax,[-1,K])
        self.seq_seg_loss = tf.reduce_mean(tf.multiply(-flat_label,tf.log(flat_seq_out+0.000001)))

        self.losses.append(self.seq_seg_loss)

      ## sub seq loss to control temporal-gcn
      if not self.disable_gcn and self.use_coarse_loss:
        print("###### use coarse loss ...")
        sub_K = 3
        if self.seq_label:
          # list to array
          fw_sub_output_softmax_list = tf.concat(fw_sub_output_softmax_list[1:][::-1],axis=1)
          sub_output_softmax_list = tf.concat(sub_output_softmax_list[1:][::-1],axis=1)

          sub_targets = tf.reshape(self.targets,[self.batch_size*self.temporal_len,self.crop_height,self.crop_width,1])
          sub_targets = tf.image.resize_images(sub_targets,self.h_size[0:2],method=tf.image.ResizeMethod.BILINEAR)
          sub_targets_new = tf.where(tf.less(sub_targets,1.0/3.0),x=tf.zeros_like(sub_targets),y=tf.ones_like(sub_targets))
          sub_targets = tf.where(tf.less(sub_targets,2.0/3.0),x=sub_targets_new,y=tf.ones_like(sub_targets)*2)
          sub_targets = tf.reshape(sub_targets,[self.batch_size,self.temporal_len]+self.h_size[0:2])

          seq_sub_label_map = tf.one_hot(tf.cast(sub_targets[:,:-1],tf.int32),sub_K) #self.targets[:,:-1]
          flat_seq_sub_label = tf.reshape(seq_sub_label_map,[-1,sub_K])
          flat_seq_sub_out = tf.reshape(sub_output_softmax_list,[-1,sub_K])
          self.seq_sub_seg_loss = tf.reduce_mean(tf.multiply(-flat_seq_sub_label,tf.log(flat_seq_sub_out+0.000001)))
          
          fw_seq_sub_label_map = tf.one_hot(tf.cast(sub_targets[:,1:],tf.int32),sub_K) # self.targets[:,1:]
          flat_fw_seq_sub_label = tf.reshape(fw_seq_sub_label_map,[-1,sub_K])
          flat_fw_seq_sub_out = tf.reshape(fw_sub_output_softmax_list,[-1,sub_K])
          self.fw_seq_sub_seg_loss = tf.reduce_mean(tf.multiply(-flat_fw_seq_sub_label,tf.log(flat_fw_seq_sub_out+0.000001)))

          self.losses.append(self.seq_sub_seg_loss)
          self.losses.append(self.fw_seq_sub_seg_loss)
        else:
          sub_targets = tf.expand_dims(self.targets,axis=3)
          sub_targets = tf.image.resize_images(sub_targets,self.h_size[0:2],method=tf.image.ResizeMethod.BILINEAR)
          sub_targets_new = tf.where(tf.less(sub_targets,1.0/3.0),x=tf.zeros_like(sub_targets),y=tf.ones_like(sub_targets))
          sub_targets = tf.where(tf.less(sub_targets,2.0/3.0),x=sub_targets_new,y=tf.ones_like(sub_targets)*2)

          sub_label_map = tf.one_hot(tf.cast(sub_targets,tf.int32),sub_K) #self.targets[:,:-1]
          flat_sub_label = tf.reshape(sub_label_map,[-1,sub_K])
          flat_sub_out = tf.reshape(sub_softmax,[-1,sub_K])
          self.seq_sub_seg_loss = tf.reduce_mean(tf.multiply(-flat_sub_label,tf.log(flat_sub_out+0.000001)))

          self.losses.append(self.seq_sub_seg_loss)

      # if self.teacher_mode and not self.disable_gcn:
      #   self.feat_loss = tf.reduce_mean(tf.keras.losses.KLD(teacher_h, h))
      #   self.losses.append(self.feat_loss)

    self.total_loss =tf.reduce_mean(self.losses)

    self.loss_sum = scalar_summary("loss", self.total_loss)
    self.val_loss_sum = scalar_summary("val_loss", self.total_loss)

  def build_seq_model_for_test(self):
    # build_seq_model_for_test
    # ##== the conv*2+pool block for unet==##
    # def conv_conv_pool(x, out_dim, ks, keep_prob=False, pool=True, scope="convx2_pool_"):
    #   with tf.variable_scope(scope) as sp:
    #     x = conv2d(x,out_dim,ksize=ks,stride=1,name=scope+"conv_1a") 
    #     x = tf.nn.relu(batchnorm(x,self.is_train,scope+'bn_1a'))
    #     x = conv2d(x,out_dim,ksize=ks,stride=1,name=scope+"conv_1b") 
    #     x = tf.nn.relu(batchnorm(x,self.is_train,scope+'bn_1b'))

    #     if keep_prob:
    #       x = tf.nn.dropout(x,self.keep_prob,seed = self.seed)
        
    #     x_pool = None
    #     if pool:
    #       x_pool = slim.max_pool2d(x,[2,2])
    #   return x, x_pool
    # def conv_up(x, skip, ks, scope="convx2_up_"):
    #   with tf.variable_scope(scope) as sp:
    #     out_dim = skip.shape[-1].value
    #     x = conv2d(x,out_dim,ksize=ks[0],stride=1,name=scope+"conv_1a") 
    #     x = tf.nn.relu(batchnorm(x,self.is_train,scope+'bn_1a'))

    #     x = tf.image.resize_bilinear(x, tf.shape(skip)[1:3])
    #     x = tf.concat([x,skip],axis=-1)

    #     x,_ = conv_conv_pool(x,out_dim,ks[1],pool=False,scope="convx2_")
    #   return x
    # ##====##

    # build_seq_model_for_test
    def single_model(inputs, prev_h=None):
      # if self.model_type=="unet":
      #   x = inputs - 127.5

      #   x1_skip,x1 = conv_conv_pool(x,32,3,scope="convx2_pool_1_") #skip: 192x192x64
      #   x2_skip,x2 = conv_conv_pool(x1,32,3,scope="convx2_pool_2_") #skip: 96x96x128
      #   x3_skip,x3 = conv_conv_pool(x2,64,3,keep_prob=True,scope="convx2_pool_3_") #skip: 48x48x256
      #   # x4_skip,x4 = conv_conv_pool(x3,512,3,keep_prob=True,scope="convx2_pool_4_") #skip: 24x24x512, output: 12x12x512
      #   # x5,_ = conv_conv_pool(x4,1024,3,pool=False,scope="convx2_pool_5_") #output: 12x12x1024
      #   x4,_ = conv_conv_pool(x3,128,3,pool=False,scope="convx2_pool_4_") #output: 24x24x512

      #   # gcn
      #   # h: batch_size x [h_size]
      #   _h = tf.expand_dims(tf.reshape(x4,[self.batch_size,-1,self.h_chn]),axis=1)
      #   # prev_h = None
      #   if prev_h is not None:
      #     if self.gcn_mode==0:
      #       _h = tf.concat([_h,prev_h],axis=2)
      #       kernel_sp = self.kernel_sp
      #       kernel_tp = None
      #     elif self.gcn_mode==1:
      #       kernel_sp = self.kernel_sp[:self.n,:self.n]
      #       _h = tf.concat([_h,prev_h],axis=2)
      #       kernel_tp = self.kernel_tp
      #     elif self.gcn_mode==2:
      #       kernel_sp = self.kernel_sp[:self.n,:self.n]
      #       _h = tf.concat([prev_h,_h],axis=1)
      #       kernel_tp = None
      #   else:
      #     kernel_sp = self.kernel_sp[:self.n,:self.n]
      #     kernel_tp = None
        
      #   _h,_ = st_conv_block(_h, kernel_sp, kernel_tp, [self.h_chn,self.h_chn,self.h_chn,3], \
      #                             self.keep_prob, h_size=self.h_size[0:2]+[self.h_chn], is_train=self.is_train, scope="gcn", gcn_mode=self.gcn_mode)
      #   _h = _h[:,:,:self.n,:]
      #   # sub_output_softmax = sub_output_softmax[:,0,:self.n,:]
      #   # sub_output_softmax = tf.reshape(sub_output_softmax,[self.batch_size,self.h_size[0],self.h_size[1],3])
      #   x4 = tf.reshape(_h[:,0,:,:],[self.batch_size,self.h_size[0],self.h_size[1],self.h_chn])

      #   # up6 = conv_up(x5, x4_skip, [2,3], scope="convx2_up_6") # 24x24x512
      #   up7 = conv_up(x4, x3_skip, [2,3], scope="convx2_up_7") # 48x48x256
      #   up8 = conv_up(up7, x2_skip, [2,3], scope="convx2_up_8") # 96x96x128
      #   up9 = conv_up(up8, x1_skip, [2,3], scope="convx2_up_9") # 192x192x64

      #   up9 = conv2d(up9,self.num_class,ksize=3,stride=1,name="out_conv") 
      #   up9 = tf.nn.relu(batchnorm(up9,self.is_train,'bn_out'))
      #   #################
      #   output_softmax = tf.nn.softmax(up9)
      #   output = tf.cast(tf.argmax(output_softmax,axis=3),tf.uint8)

      #   return _h, output_softmax, output
      
      # elif "deeplab" in self.model_type:
      h = inputs-127.5
    
      if "resnet" in self.model_type:
        end_points = {}
        with slim.arg_scope([slim.batch_norm],is_training=self.is_train):
            _, end_points = resnet_v2_50(h,
                num_classes=0,
                is_training=self.is_train,
                global_pool=False,
                output_stride=16,
                spatial_squeeze=False,
                reuse=None,
                scope=self.model_name)
        if not self.half_model:
          h = end_points['resnet_v2_50/block4']  
        else:
          h = end_points['resnet_v2_50/block3'] 
        skip = end_points['resnet_v2_50/block1/unit_3/bottleneck_v2/conv1']   
      elif "mobilenet" in self.model_type:
        end_points = {}
        with slim.arg_scope([slim.batch_norm],is_training=self.is_train):
            backbone_scope='MobilenetV1'
            _,end_points = mobilenet_v1(h,
                    num_classes=0,
                    dropout_keep_prob=1.0,
                    is_training=self.is_train,
                    min_depth=8,
                    depth_multiplier=1.0,
                    conv_defs=None,
                    prediction_fn=contrib_layers.softmax,
                    spatial_squeeze=False,
                    output_stride=16,
                    reuse=None,
                    scope=backbone_scope,
                    global_pool=False)
        if not self.half_model:
          h = end_points['Conv2d_13_pointwise'] 
        else:
          h = end_points['Conv2d_8_pointwise']   
        skip = end_points['Conv2d_2_pointwise']  

      if not 'EP' in self.valid_data[0]:    
        h = tf.nn.dropout(h,self.keep_prob,seed = self.seed)
      
      # atrous spatial pyramid pooling
      h = atrous_spatial_pyramid_pooling(h, output_stride=16, depth=self.h_chn,is_train=self.is_train)
      
      # skip connect low level features
      skip = conv2d(skip,32,ksize=1,stride=1,name="conv_skip") 
      skip = tf.nn.relu(batchnorm(skip,self.is_train,'bn_skip'))

      # gcn
      if self.temporal_len == 1:
        _,h = single_gcn(h,prev_h=prev_h)

      # upsample*4
      h = tf.image.resize_bilinear(h, tf.shape(skip)[1:3])
      
      # concate and segment
      h = tf.concat([h,skip],axis=3)    
      
      h = separable_conv2d(h,128,ksize=3,name="conv_out1")
      h = tf.nn.relu(batchnorm(h,self.is_train,'bn_out1'))

      h = separable_conv2d(h,128,ksize=3,name="conv_out2")
      h = tf.nn.relu(h)
      
      h = separable_conv2d(h,self.num_class,ksize=3,name="conv_out3")
      # upsample
      h = tf.image.resize_bilinear(h, [self.crop_height, self.crop_width])
      
      output_softmax = tf.nn.softmax(h)
      output = tf.cast(tf.argmax(output_softmax,axis=3),tf.uint8)
      return output_softmax, output

    # build_seq_model_for_test
    def deeplab_decoder(skip,h):
      # if "unet" == self.model_type:
      #   x1_skip,x2_skip,x3_skip=skip
      #   up7 = conv_up(h, x3_skip, [2,3], scope="convx2_up_7") # 96x96x128
      #   up8 = conv_up(up7, x2_skip, [2,3], scope="convx2_up_8") # 96x96x128
      #   up9 = conv_up(up8, x1_skip, [2,3], scope="convx2_up_9") # 192x192x64

      #   up9 = conv2d(up9,self.num_class,ksize=3,stride=1,name="out_conv") 
      #   up9 = tf.nn.relu(batchnorm(up9,self.is_train,'bn_out'))
      #   output_softmax = tf.nn.softmax(up9)
      #   output = tf.cast(tf.argmax(output_softmax,axis=3),tf.uint8)

      #   return output_softmax, output
      
      # elif "deeplab" in self.model_type:
      # upsample*4
      h = tf.image.resize_bilinear(h, tf.shape(skip)[1:3])
      
      # concate and segment
      h = tf.concat([h,skip],axis=3)    
      
      h = separable_conv2d(h,128,ksize=3,name="conv_out1")
      h = tf.nn.relu(batchnorm(h,self.is_train,'bn_out1'))

      h = separable_conv2d(h,128,ksize=3,name="conv_out2")
      h = tf.nn.relu(h)
      
      h = separable_conv2d(h,self.num_class,ksize=3,name="conv_out3")
      # upsample
      h = tf.image.resize_bilinear(h, [self.crop_height, self.crop_width])
      
      output_softmax = tf.nn.softmax(h)
      output = tf.cast(tf.argmax(output_softmax,axis=3),tf.uint8)
      return output_softmax, output

    # build_seq_model_for_test
    def single_enc(inputs,model_type,teacher=False):
      # if teacher:
      #   is_train=False
      # else:
      #   is_train=self.is_train
      is_train=False

      # if model_type=="unet":
      #   x = inputs - 127.5

      #   x1_skip,x1 = conv_conv_pool(x,64,3,scope="convx2_pool_1_") #skip: 192x192x64
      #   x2_skip,x2 = conv_conv_pool(x1,128,3,scope="convx2_pool_2_") #skip: 96x96x128
      #   x3_skip,x3 = conv_conv_pool(x2,256,3,keep_prob=True,scope="convx2_pool_3_") #skip: 48x48x256
      #   # x4_skip,x4 = conv_conv_pool(x3,512,3,keep_prob=True,scope="convx2_pool_4_") #skip: 24x24x512, output: 12x12x512
      #   # _,x5 = conv_conv_pool(x4,1024,3,pool=False,scope="convx2_pool_5_") #output: 12x12x1024
      #   x4,_ = conv_conv_pool(x3,512,3,pool=False,scope="convx2_pool_4_")

      #   skip = [x1_skip,x2_skip,x3_skip]

      #   return skip, x4
      
      # elif "deeplab" in model_type:
      h = inputs-127.5
    
      if "resnet" in model_type:
        end_points = {}
        with slim.arg_scope([slim.batch_norm],is_training=is_train):
            _, end_points = resnet_v2_50(h,
                num_classes=0,
                is_training=is_train,
                global_pool=False,
                output_stride=16,
                spatial_squeeze=False,
                reuse=None,
                scope="resnet_v2_50")
                # scope=self.model_name)
        
        # if teacher:
        #   h = end_points['teacher/resnet_v2_50/block4']   
        #   skip = end_points['teacher/resnet_v2_50/block1/unit_3/bottleneck_v2/conv1'] 
        # else:
        if not self.half_model:
          h = end_points['resnet_v2_50/block4']
        else:
          h = end_points['resnet_v2_50/block3']      
        skip = end_points['resnet_v2_50/block1/unit_3/bottleneck_v2/conv1']  
      elif "mobilenet" in model_type:
        end_points = {}
        with slim.arg_scope([slim.batch_norm],is_training=is_train):
            backbone_scope='MobilenetV1'
            _,end_points = mobilenet_v1(h,
                    num_classes=0,
                    dropout_keep_prob=1.0,
                    is_training=is_train,
                    min_depth=8,
                    depth_multiplier=1.0,
                    conv_defs=None,
                    prediction_fn=contrib_layers.softmax,
                    spatial_squeeze=False,
                    output_stride=16,
                    reuse=None,
                    scope=backbone_scope,
                    global_pool=False)
        
        # if teacher:
        #   # h = end_points['teacher/Conv2d_13_pointwise']   
        #   h = end_points['teacher/Conv2d_8_pointwise']   
        #   skip = end_points['teacher/Conv2d_2_pointwise']
        # else:
        if not self.half_model:
          h = end_points['Conv2d_13_pointwise']  
        else: 
          h = end_points['Conv2d_8_pointwise']   
        skip = end_points['Conv2d_2_pointwise']

      if not 'EP' in self.valid_data[0]:     
        h = tf.nn.dropout(h,self.keep_prob,seed = self.seed)
      
      # atrous spatial pyramid pooling
      h = atrous_spatial_pyramid_pooling(h, output_stride=16, depth=self.h_chn,is_train=is_train)
      
      # skip connect low level features
      skip = conv2d(skip,32,ksize=1,stride=1,name="conv_skip") 
      skip = tf.nn.relu(batchnorm(skip,is_train,'bn_skip'))
      
      return skip, h

    # build_seq_model_for_test
    def merge_prev(prev_output,prev_h):
      if prev_h is None:
        return None

      prev_output = tf.expand_dims(prev_output,axis=3)
      prev_output = tf.image.resize_images(prev_output,[self.h_size[0],self.h_size[1]],method=tf.image.ResizeMethod.BILINEAR)
      
      prev_output = tf.expand_dims(tf.reshape(prev_output,[self.batch_size,-1,1]),axis=1)
      return prev_output*prev_h

    ## build_seq_model_for_test
    # output_sub=True
    def single_gcn(h,prev_h=None,scope="gcn"):

      # h: batch_size x [h_size]
      # _h = tf.expand_dims(tf.reshape(h,[self.batch_size,-1,self.h_chn]),axis=1)
      _h = tf.reshape(h,[self.batch_size,1,-1,self.h_chn])
      # prev_h = None
      if prev_h is not None:
        # global_map = self.global_matching(_h, prev_h)
        if self.gcn_mode==0:
          _h = tf.concat([_h,prev_h],axis=2)
          kernel_sp = self.kernel_sp
        elif self.gcn_mode==1:
          kernel_sp = self.kernel_sp[:self.n,:self.n]
          _h = tf.concat([_h,prev_h],axis=2)
          kernel_tp = self.kernel_tp
        elif self.gcn_mode==2:
          _h = tf.concat([prev_h,_h],axis=1)
      # else:
      #   # global_map = None
      #   kernel_sp = self.kernel_sp[:self.n,:self.n]
      
      h, _h, _ = st_conv_block(_h, [self.h_chn,self.h_chn,self.h_chn,3], \
                                h_size=self.h_size, is_train=self.is_train, output_sub=False, scope=scope, gcn_mode=self.gcn_mode)
      return _h, h
    #################################

    # placeholders
    ##############
    # inputs is a 5D tensor, BTHWC
    self.keep_prob = tf.placeholder(tf.float32)
    # self.keep_prob = 1.0

    # hidden state
    if self.test_video:
      self.inputs = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims) 
      self.prev_h = tf.placeholder(tf.float32, [self.batch_size,1,self.h_size[0]*self.h_size[1],self.h_size[2]])
    else:
      self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.temporal_len] + self.image_dims)
      prev_h = None
      prev_output = None
    ##############

    with tf.variable_scope('',reuse=tf.AUTO_REUSE) as scope:

      if self.temporal_len == 1 and not self.test_video:
        self.output_softmax, _ = single_model(self.inputs[:,0,:,:,:])
      elif self.test_video:
        skip, h = single_enc(self.inputs,self.model_type)
        if not self.disable_gcn:
          self.output_h_prev,h = single_gcn(h,prev_h=self.prev_h)
        # if "unet" == self.model_type:
        #   self.output_softmax, _ = deeplab_decoder([skip[0],skip[1],skip[2],h])
        # elif "deeplab" in self.model_type:
        self.output_softmax, _ = deeplab_decoder(skip,h)
      else: 
        # if self.pass_hidden:
        #   prev_h_out = None
        #   for i in range(self.temporal_len-1,-1,-1):
        #     prev_h_out,_,_ = single_gcn(hs[:,i],prev_h=prev_h_out)
        #   self.prev_h_out = prev_h_out

        for i in range(self.temporal_len-1,-1,-1):
          # get feature maps
          skip, h = single_enc(self.inputs[:,i],self.model_type)

          if not self.disable_gcn:
            prev_h = merge_prev(prev_output,prev_h)
            prev_h,h = single_gcn(h,prev_h=prev_h)

          # if "unet" == self.model_type:
          #   output_softmax, output = deeplab_decoder([skip[0],skip[1],skip[2],h])
          # elif "deeplab" in self.model_type:
          output_softmax, output = deeplab_decoder(skip,h)
        
          if i > 0:
            prev_output = output

        self.output_softmax = output_softmax
        # if self.pass_hidden:
        #   self.output_h_prev = prev_h

  def inference(self, img, img_prev=None, h_prev=None, prev_output=None, h_list=None, zero_init=False, extra_output=False):
    def merge_prev(prev_output,prev_h):
      prev_output = cv2.resize(prev_output[0].astype(np.uint8),(self.h_size[0],self.h_size[1])) #interpolation=cv2.INTER_NEAREST
      prev_output = prev_output[...,np.newaxis]
      prev_output = np.reshape(prev_output,(self.batch_size,-1,1))[:,np.newaxis,...]
      return prev_output*prev_h

    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    # make sure the input images are in the right size
    if self.rnn_mode==0:
      shape0 = (img.shape[1],img.shape[0])
      img = cv2.resize(img,(self.input_width,self.input_height))
      inputs = np.array([img]).astype(np.float32)
    elif self.rnn_mode==1:
      shape0 = (img[0].shape[1],img[0].shape[0])
      for i in range(self.temporal_len):        
          img[i]= cv2.resize(img[i],(self.input_width,self.input_height))[np.newaxis,np.newaxis,...]
      inputs = np.concatenate(img,axis=1).astype(np.float32)

      # if self.pass_hidden and not self.test_video:
      #   for i in range(self.temporal_len):        
      #     img_prev[i]= cv2.resize(img_prev[i],(self.input_width,self.input_height))[np.newaxis,np.newaxis,...]
      #   inputs_prev = np.concatenate(img_prev,axis=1).astype(np.float32)

    if self.rnn_mode==0:
      t0 = time.time()
      out_softmax = self.sess.run(self.output_softmax,feed_dict={self.inputs:inputs,self.keep_prob:1.0})
      t = time.time()-t0
      # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
      # chrome_trace = fetched_timeline.generate_chrome_trace_format()
      # with open('timeline_mob8.json', 'w') as f:
      #     f.write(chrome_trace)
    elif self.rnn_mode == 1:
      if self.test_video:
        inputs = inputs[:,0,:,:]
        if zero_init:
          h_prev = np.zeros((self.batch_size,1,self.n,self.h_chn))
        else:
          h_prev = merge_prev(prev_output,h_prev)
        t0 = time.time()
        out_softmax,h_prev = self.sess.run([self.output_softmax,self.output_h_prev], \
                             feed_dict={self.inputs:inputs,self.prev_h:h_prev,self.keep_prob:1.0})
                             # options=options, run_metadata=run_metadata
        t = time.time()-t0
        
        # Create the Timeline object, and write it to a json file
        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('timeline_mob8_mffa.json', 'w') as f:
        #     f.write(chrome_trace)
      else:
        t0 = time.time()
        out_softmax = self.sess.run(self.output_softmax,\
                                              feed_dict={self.inputs:inputs,self.keep_prob:1.0})#self.kernel_sp: kernel_sp, self.kernel_tp: kernel_tp,\
        t = time.time()-t0
    
    # if out_softmax is not None:
    outp = out_softmax[0,:,:,:]
    thresh = 0.5
    outp[outp<thresh] = 0
    out = np.argmax(outp,axis=2)
      
    rst=idxmap2colormap(out,self.color_table)
    
    idxmap = cv2.resize(out,shape0,interpolation=cv2.INTER_NEAREST)
    colormap = cv2.resize(rst,shape0,interpolation=cv2.INTER_NEAREST)
    
    if h_prev is None:
      if extra_output:
        return idxmap,colormap,out_softmax[0,:,:,1],t*1000
      else:
        return idxmap,colormap,t*1000
    else:
      if extra_output:
        return idxmap,colormap,h_prev,out_softmax[0,:,:,1],t*1000
      else:
        return idxmap,colormap,h_prev,t*1000
      
  def train(self,config):
    # np.random.seed(self.seed)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                          for v in self.optimize_vars]) # tf.trainable_variables()
    print("parameter_count =", self.sess.run(parameter_count))

    # images w.o. bw (interv=5,seq_len=4):
    # for interval = 5
    # bad_list = ['S01_17520.jpg','S02_32160.jpg','S05_30000.jpg']
    # pass_hidden_bad_list = ['S01_17520.jpg','S02_32160.jpg','S05_30000.jpg',\
    # 'S08_17580.jpg','S09_33060.jpg']
    # for interval = 3
    bad_list = ['S02_32160.jpg','S05_30000.jpg','L01_17580.jpg','L02_83555.jpg','L03_22440.jpg']
    pass_hidden_bad_list = ['S01_17520.jpg','S02_32160.jpg','S05_30000.jpg',\
    'L01_17580.jpg','L01_36429.jpg','L02_98948.jpg','L02_18750.jpg','L02_83555.jpg','L03_22440.jpg']
    # # for interval = 1
    # bad_list = ['L02_83555.jpg']
    # pass_hidden_bad_list = ['S05_30000.jpg','L02_83555.jpg','L03_22440.jpg']

    batch_num = len(self.data) // self.batch_size
    
    ##### config the optimizer #####
    # learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = self.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                          batch_num*config.decay_epoch, 0.5, staircase=True)
      
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for updating moving average of batchnorm
    # with tf.control_dependencies(update_ops):
    #   optim = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1) \
    #           .minimize(self.seg_loss, var_list=self.optimize_vars,global_step=global_step)

    _optim = tf.train.AdamOptimizer(learning_rate, self.beta1)
    # Compute gradients
    with tf.name_scope("compute_gradients"):
        # Get the gradient pairs (Tensor, Variable)
        #depth_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pose_net")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
          grads = _optim.compute_gradients(self.total_loss,var_list=self.optimize_vars)
    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"):
        optim = _optim.apply_gradients(grads, global_step)
    ##### config the optimizer(end) #####

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.train_sum = merge_summary([self.loss_sum])
    # self.writer = SummaryWriter(os.path.join(self.checkpoint_dir,"logs"), self.sess.graph)
    self.writer = SummaryWriter(os.path.join(self.save_checkpoint_dir,"logs"), self.sess.graph)
  
    counter = 1
    start_time = time.time()
    if os.path.exists(self.pretrain_dir):
        could_load, checkpoint_counter = self.load_pretrain(self.pretrain_dir)
        if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")
    elif self.continue_train:
        could_load, checkpoint_counter = self.load(os.path.join(self.checkpoint_dir,"DeepLab_{}_{}_{}".format(self.batch_size,self.input_height,self.input_width)))
        if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")

    # if self.teacher_mode:
    #   could_load, _ = self.load_teacher(os.path.join(self.teacher_checkpoint_dir,"DeepLab_16_240_240"))
    #   if could_load:
    #     print(" [*] Load teacher model SUCCESS")
    #   else:
    #     print(" [!] Load teacher model failed...")

    idxs = np.arange(len(self.data))
    idxv = np.arange(len(self.val_data))
    train_loss_queue = []
    for epoch in xrange(config.epoch):
      
      random.seed(self.seed)
      random.shuffle(idxs)
      random.seed(self.seed)
      random.shuffle(idxv)
      
      for idx in xrange(0, batch_num):
        file_idxs = idxs[idx*self.batch_size:(idx+1)*self.batch_size]
        
        interval = self.frame_interval
        if np.random.random()>0.5:
            interval = -interval

        if not self.seq_label:
          for i in file_idxs:
            data_basename = os.path.basename(self.data[i])
            # if self.pass_hidden:
            #   if data_basename in pass_hidden_bad_list:
            #     interval = self.frame_interval
            #     break
            # else:
            if data_basename in bad_list:
              interval = self.frame_interval
              break

          # if self.temporal_len > 1:
          #   # get the interval value
          #   data_basenames = []
          #   intervals = []
          #   for i in file_idxs:
          #     data_basename = os.path.basename(self.data[i])
          #     data_basenames.append(data_basename)

          #     if data_basename in bad_list:
          #       interval = self.frame_interval
          #     elif np.random.random()>0.5:
          #       interval = self.frame_interval
          #     else:
          #       interval = -self.frame_interval
          #     intervals.append(interval)

          #   batch_images = [np.array(sequence_read(self.data[i], self.frame_dataset, self.temporal_len, interval=intervals[j], 
          #                                   resize_wh=(self.input_width,self.input_height), nearest_interpolate=True, grayscale = False)
          #                                   )[:self.temporal_len] for i,j in zip(file_idxs,range(self.batch_size))]
          # else:
          #   batch_images = [np.array(sequence_read(self.data[i], self.frame_dataset, self.temporal_len, interval=self.frame_interval, 
          #                                     resize_wh=(self.input_width,self.input_height), nearest_interpolate=True, grayscale = False)
          #                                     )[:self.temporal_len] for i in file_idxs]
          batch_images = [np.array(sequence_read(self.data[i], self.frame_dataset, self.temporal_len, interval=interval, 
                                              resize_wh=(self.input_width,self.input_height), nearest_interpolate=True, grayscale = False)
                                              )[:self.temporal_len] for i in file_idxs]
          batch_labels = [imread(self.label[i],resize_wh=(self.input_width,self.input_height),
                                      nearest_interpolate=True,grayscale=True) for i in file_idxs]
          # if self.pass_hidden:
          #   comb_images = []
          #   for i,j in zip(file_idxs,range(self.batch_size)):
          #     data_basename = os.path.basename(self.data[i])
          #     # data_basename = data_basenames[j]
          #     # interval = intervals[j]

          #     prev_name, prev_idx = data_basename[:-4].split("_")
          #     prev_name = prev_name + "_" + str(int(prev_idx)-self.temporal_len*interval) + ".jpg"
          #     prev_name = os.path.join(self.train_dataset,"images",prev_name)
          #     prev_image = np.array(sequence_read(prev_name, self.frame_dataset, self.temporal_len, interval=interval, 
          #                                   resize_wh=(self.input_width,self.input_height), nearest_interpolate=True, grayscale = False)
          #                                   )[:self.temporal_len]
          #     cur_image = batch_images[j]
          #     comb_images.append(np.concatenate([prev_image,cur_image],axis=0))
        else:
          batch_images = []
          batch_labels = []
          for i in file_idxs:
            img_seq, label_seq = full_sequence_read(self.data[i], self.label[i], self.temporal_len, 
                                          resize_wh=(self.input_width,self.input_height))
            batch_images.append(np.array(img_seq))
            batch_labels.append(np.array(label_seq))
            # batch_images.append(img_seq)
            # batch_labels.append(label_seq)
                     
        # augmentaion
        # if self.pass_hidden:
        #   prev_batch_images = []
        #   batch_images = []
        #   for im in comb_images:
        #      aug_img = self.sess.run(self.im_aug,feed_dict={self.im_raw:im})
        #      prev_batch_images.append(aug_img[:self.temporal_len])
        #      batch_images.append(aug_img[self.temporal_len:])
        # else:
        batch_images = [self.sess.run(self.im_aug,feed_dict={self.im_raw:im}) for im in batch_images]
        if self.seq_label:
          batch_labels = [self.sess.run(self.label_aug,feed_dict={self.label_raw:lb[...,np.newaxis]})[:,:,:,0] for lb in batch_labels]
        else:
          batch_labels = [self.sess.run(self.label_aug,feed_dict={self.label_raw:np.reshape(lb,[lb.shape[0],lb.shape[1],1])})[:,:,0] for lb in batch_labels]
        
        batch_images = np.array(batch_images).astype(np.float32)
        batch_labels = np.array(batch_labels).astype(np.int32)
        if 'EP' in self.valid_data[0]:
          batch_labels[batch_labels>0]=1
        # if self.pass_hidden:
        #   prev_batch_images = np.array(prev_batch_images).astype(np.float32)

        # if self.seq_train and self.seq_label:
        if self.seq_label:
          input_idx = self.temporal_len//2
        else:
          input_idx = 0

        # Update gradient
        if self.rnn_mode==0:
          _,train_loss, summary_str,cur_lr = self.sess.run([optim,self.seg_loss, self.loss_sum,learning_rate],
                                                    feed_dict={ self.inputs: batch_images[:,0,:,:,:], self.targets: batch_labels,self.keep_prob:0.5})
        elif self.rnn_mode == 1:
          # if self.pass_hidden:
          #   prev_h = self.sess.run(self.prev_h_out,feed_dict={ self.inputs: prev_batch_images,\
          #                                                       self.keep_prob:0.5})#self.kernel_sp:kernel_sp, \self.kernel_tp:kernel_tp,
          #   _,train_loss, summary_str, cur_lr = self.sess.run([optim,self.total_loss,self.loss_sum,learning_rate],
          #                                                         feed_dict={ self.inputs: batch_images,self.targets: batch_labels,\
          #                                                         self.prev_h:prev_h, \
          #                                                         self.keep_prob:0.5}) # self.kernel_sp:kernel_sp,self.kernel_tp:kernel_tp,
          # else:
          _,train_loss, summary_str, cur_lr = self.sess.run([optim,self.total_loss,self.loss_sum,learning_rate],
                                                                feed_dict={ self.inputs: batch_images,self.targets: batch_labels,\
                                                                self.keep_prob:0.5}) # self.prev_idx: prev_idx, self.kernel_sp:kernel_sp, self.kernel_tp:kernel_tp,\

        self.writer.add_summary(summary_str, counter)

        counter += 1
        train_loss_queue.append(train_loss)
        if len(train_loss_queue)>10:
            train_loss_queue.pop(0)
        train_loss_mean = np.mean(train_loss_queue)
        print("Epoch[%2d/%2d] [%3d/%3d] time:%.2f min, loss:[%.4f], lr: %.5f" \
          % (epoch, config.epoch, idx, batch_num, (time.time() - start_time)/60, train_loss_mean, cur_lr))

        if counter% (batch_num//10) == 0 and self.save_sample:
          file_idx0 = 0#np.random.randint(len(self.val_data)-self.batch_size)
          file_idxs = idxv[file_idx0:self.batch_size+file_idx0]
          if not self.seq_label:
            # val_batch_images = [imread(self.val_data[i],resize_wh=(self.input_width,self.input_height),
            #                            nearest_interpolate=True,grayscale=False) for i in file_idxs]
            val_batch_images = [sequence_read(self.val_data[i], self.frame_dataset, self.temporal_len, interval=self.frame_interval, 
                                                  resize_wh=(self.input_width,self.input_height), nearest_interpolate=True, grayscale = False)[:self.temporal_len]
                                                  for i in file_idxs]
            val_batch_labels = [imread(self.val_label[i],resize_wh=(self.input_width,self.input_height),
                                      nearest_interpolate=True,grayscale=True) for i in file_idxs]
          else:
            val_batch_images = []
            val_batch_labels = []
            for i in file_idxs:
              img_seq, label_seq = full_sequence_read(self.val_data[i], self.val_label[i], self.temporal_len, 
                                            resize_wh=(self.input_width,self.input_height))
              val_batch_images.append(np.array(img_seq))
              val_batch_labels.append(np.array(label_seq))
              # val_batch_images.append(img_seq)
              # val_batch_labels.append(label_seq)

          # augmentaion
          val_batch_images = [self.sess.run(self.im_aug,feed_dict={self.im_raw:im}) for im in val_batch_images]
          if self.seq_label:
            val_batch_labels = [self.sess.run(self.label_aug,feed_dict={self.label_raw:lb[...,np.newaxis]})[:,:,:,0] for lb in val_batch_labels]
          else:
            val_batch_labels = [self.sess.run(self.label_aug,feed_dict={self.label_raw:np.reshape(lb,[lb.shape[0],lb.shape[1],1])})[:,:,0] for lb in val_batch_labels]
                        
          val_batch_images = np.array(val_batch_images).astype(np.float32)
          val_batch_labels = np.array(val_batch_labels).astype(np.int32)
          if 'E' in self.valid_data[0]:
            val_batch_labels[val_batch_labels>0]=1
          # val_batch_images = self.sess.run(self.sequence_aug,feed_dict={self.sequence_raw:val_batch_images[:,np.newaxis,...]})[:,0,:,:,:]
          # val_batch_labels = self.sess.run(self.label_aug,feed_dict={self.label_raw:val_batch_labels[...,np.newaxis]})[:,:,:,0]
          
          if self.rnn_mode==0:
            out, train_loss, summary_str = self.sess.run([self.output,self.seg_loss, self.val_loss_sum],
                                                    feed_dict={ self.inputs: val_batch_images[:,0,:,:,:], self.targets: val_batch_labels,self.keep_prob:1.0})
            disp_idx=(counter//(batch_num//10))%self.batch_size
            output=idxmap2colormap(out[disp_idx,:,:],self.color_table)
            label = idxmap2colormap(val_batch_labels[disp_idx,:,:],self.color_table)
            input = val_batch_images[disp_idx,0,:,:,:]
            # input = 0.9*input+0.1*label
            rst=np.hstack((input,label,output))
            filename = "%08d.png" % (counter)
          elif self.rnn_mode==1:
            if self.temporal_len==1:
              out, train_loss, summary_str = self.sess.run([self.output,self.total_loss,self.val_loss_sum],
                                                  feed_dict={ self.inputs: val_batch_images, self.targets: val_batch_labels,\
                                                  self.keep_prob:1.0})#self.kernel_tp:kernel_tp, self.kernel_sp:kernel_sp,
              disp_idx=(counter//(batch_num//10))%self.batch_size
              output=idxmap2colormap(out[disp_idx,:,:],self.color_table)
              label = idxmap2colormap(val_batch_labels[disp_idx,:,:],self.color_table)
              input = val_batch_images[disp_idx,0,:,:,:]
              # input = 0.9*input+0.1*label
              rst=np.hstack((input,label,output))
            else:
              # if self.seq_label:
              #   prev_idx = []
              #   for k in range(self.temporal_len):
              #     # get the prev seq
              #     _prev_idx = list(np.arange(self.temporal_len))
              #     _prev_idx.pop(k)
              #     random.shuffle(_prev_idx)
              #     _prev_idx.append(k)
              #     prev_idx.append(_prev_idx)
              #   prev_idx = np.array(np.concatenate(prev_idx),dtype=np.int32)
              # else:
              #   prev_idx = list(np.arange(self.temporal_len*2))
              #   random.shuffle(prev_idx)
              #   prev_idx.pop(0)
              #   prev_idx = [pidx%self.temporal_len for pidx in prev_idx]
              #   prev_idx.append(0)
              #   prev_idx = np.array(prev_idx,dtype=np.int32)
              outs, train_loss, summary_str = self.sess.run([self.output_list,self.total_loss,self.val_loss_sum],
                                                                  feed_dict={ self.inputs: val_batch_images,self.targets: val_batch_labels,\
                                                                  self.keep_prob:0.5})# self.prev_idx: prev_idx, self.kernel_sp:kernel_sp, self.kernel_tp:kernel_tp,\
              disp_idx=(counter//(batch_num//10))%self.batch_size
              input = val_batch_images[disp_idx,:,:,:,:]
              if not self.seq_label:
                label = idxmap2colormap(val_batch_labels[disp_idx,:,:],self.color_table)
                input[0] = input[0]*0.9+label*0.1
              input = np.concatenate([input[i] for i in range(self.temporal_len)],axis=1)
              if self.seq_label:
                label = val_batch_labels[disp_idx,:,:,:]
                label = np.concatenate([label[i] for i in range(self.temporal_len)],axis=1)
                label = idxmap2colormap(label,self.color_table)
                input = input*0.9+label*0.1
              # wraps = np.concatenate([wrap for wrap in wraps[disp_idx,:,:,:,:]]+[np.zeros((self.h_size[0],self.h_size[1],3))],axis=1)
              # wraps = cv2.resize(wraps,(self.crop_width*self.temporal_len,self.crop_width))
              outs = idxmap2colormap(outs[disp_idx,:,:],self.color_table)
              rst=np.concatenate([input,outs],axis=0)
            filename = "%08d.png" % (counter)

          self.writer.add_summary(summary_str, counter)
          
          # cv2.imwrite(os.path.join(self.checkpoint_dir,filename),rst)
          cv2.imwrite(os.path.join(self.save_checkpoint_dir,filename),rst)
          
          
        if np.mod(counter, (batch_num//2)) == 0:
          # self.save(self.checkpoint_dir, counter) 
          self.save(self.save_checkpoint_dir, counter)    
    # self.save(self.checkpoint_dir, counter)   
    self.save(self.save_checkpoint_dir, counter)  

          
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        'DeepLab', self.batch_size,
        self.input_height, self.input_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DeepLab.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    try:
      self.sess.run(tf.global_variables_initializer())
    except:
      self.sess.run(tf.initialize_all_variables().run())
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.loader.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

  def load_teacher(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    try:
      self.sess.run(tf.global_variables_initializer())
    except:
      self.sess.run(tf.initialize_all_variables().run())
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.teacher_loader.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      # counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, 0
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
    
  def load_pretrain(self, pretrain_file):
    import re
    print(" [*] Reading checkpoints...")
    try:
      self.sess.run(tf.global_variables_initializer())
    except:
      self.sess.run(tf.initialize_all_variables().run())
    
    if "mobilenet" in self.model_type:
      self.loader.restore(self.sess,os.path.join(self.pretrain_dir,"mobilenet_v1_1.0_224"+".ckpt"))
    else:
      self.loader.restore(self.sess,os.path.join(self.pretrain_dir,self.model_name+".ckpt"))
#    tf.train.init_from_checkpoint(self.pretrain_file,{v.name.split(':')[0]:v for v in self.load_vars})
    counter = 0
    return True,counter
      
