import os
import numpy as np
import cv2
from model import *
from utils import *
import tensorflow as tf

overall_random_seed = 23 # EP:5, sinus:23
np.random.seed(overall_random_seed)
tf.set_random_seed(overall_random_seed)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)
tf.reset_default_graph()
tf.set_random_seed(overall_random_seed)

flags = tf.app.flags
flags.DEFINE_integer("epoch",30, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("seed", overall_random_seed, "random seed")
flags.DEFINE_integer("input_height", 240, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 240, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("crop_height", 192, "The size of image to crop")
flags.DEFINE_integer("crop_width", 192, "")
flags.DEFINE_integer("temporal_len",4,"the number of consecutive frames to input")

# flags.DEFINE_string("train_dataset", "../sinus_data/cadaver", "train dataset direction")
flags.DEFINE_string("train_dataset", "../sinus_data/syn_cadaver", "train dataset direction")
flags.DEFINE_string("frame_dataset", "../sinus_data/cadaver/frame_dataset", "frame dataset direction")
flags.DEFINE_string("video_dir", "../sinus_data/cadaver/videos", "train dataset direction")
flags.DEFINE_string("datasets", "cf1cf2", "")

flags.DEFINE_string("img_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("label_pattern", "*.png", "Glob pattern of filename of input labels [*]")

flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("save_checkpoint_dir", "", "Directory name to save the checkpoints [checkpoint]")
# flags.DEFINE_string("pretrain_dir", "../pretrain/resnet_v2_50_2017_04_14", "")
flags.DEFINE_string("pretrain_dir", "../pretrain/mobilenet_v1_1.0_224", "")

#$$$$ SL
flags.DEFINE_string("model_type", "deeplab_mobilenet", "")#unet, deeplab_mobilenet, deeplab_resnet

flags.DEFINE_integer("continue_train",0,"")
flags.DEFINE_integer("pass_hidden",0,"")
flags.DEFINE_integer("seq_label",0,"")
flags.DEFINE_integer("teacher_mode",0,"")
flags.DEFINE_integer("disable_gcn",0,"")

# flags.DEFINE_integer("fold_id",0, "")

flags.DEFINE_integer("rnn_mode",1, "")
flags.DEFINE_integer("decay_epoch",15, "Epoch to decay learning rate")
flags.DEFINE_float("learning_rate",0.000125,"")
# flags.DEFINE_float("learning_rate",0.0000625,"")

flags.DEFINE_string("gpu", '0', "gpu")
FLAGS = flags.FLAGS


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.save_checkpoint_dir) and not FLAGS.save_checkpoint_dir=="":
    os.makedirs(FLAGS.save_checkpoint_dir)

  # cvt number to bool
  continue_train = False if FLAGS.continue_train==0 else True
  pass_hidden = False if FLAGS.pass_hidden==0 else True
  seq_label = False if FLAGS.seq_label==0 else True
  teacher_mode = False if FLAGS.teacher_mode==0 else True
  disable_gcn = False if FLAGS.disable_gcn==0 else True

  color_table = load_color_table('./labels.json')
  
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  # run_config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  tf.reset_default_graph()
  tf.set_random_seed(overall_random_seed)
  with tf.Session(config=run_config) as sess:

    net = DeepLab(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          crop_width=FLAGS.crop_width,
          crop_height=FLAGS.crop_height,
          batch_size=FLAGS.batch_size,
          seed=FLAGS.seed,
          temporal_len=FLAGS.temporal_len,
          img_pattern=FLAGS.img_pattern,
          label_pattern=FLAGS.label_pattern,
          checkpoint_dir=FLAGS.checkpoint_dir,
          save_checkpoint_dir=FLAGS.save_checkpoint_dir,
          pretrain_dir=FLAGS.pretrain_dir,
          datasets=FLAGS.datasets,
          train_dataset=FLAGS.train_dataset,
          frame_dataset=FLAGS.frame_dataset,
          video_dir=FLAGS.video_dir,
          continue_train=continue_train, ###
          pass_hidden=pass_hidden,
          seq_label=seq_label,
          teacher_mode=teacher_mode,
          disable_gcn=disable_gcn,
          model_type=FLAGS.model_type,
          rnn_mode=FLAGS.rnn_mode,
          learning_rate=FLAGS.learning_rate, 
          # fold_id=FLAGS.fold_id, ###
          num_class=2,
          color_table=color_table,
          test_video=False,is_train=True)

    net.train(FLAGS)
      
      

    
if __name__ == '__main__':
  tf.app.run()
