import os
import numpy as np
import cv2
from model import *
from utils import *
import tensorflow as tf
from PIL import Image



def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)
tf.reset_default_graph()


flags = tf.app.flags
flags.DEFINE_integer("input_height", 240, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 240, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("crop_height", 240, "The size of image to crop")
flags.DEFINE_integer("crop_width", 240, "")
flags.DEFINE_integer("temporal_len",4,"the number of consecutive frames to input")

flags.DEFINE_string("datasets", "", "")
flags.DEFINE_string("train_dataset", "", "train dataset direction")
flags.DEFINE_string("val_dataset", "", "val dataset direction")
flags.DEFINE_string("frame_dataset", "../../sinus_data/cadaver/frame_dataset", "frame dataset direction")
flags.DEFINE_string("video_dir", "", "train dataset direction")
flags.DEFINE_string("checkpoint_dir", "", "checkpoint")
flags.DEFINE_string("img_dir", "", "img_dir")
flags.DEFINE_string("rst_dir", "", "rst_dir")
flags.DEFINE_string("gt_dir", "", "gt_dir")
flags.DEFINE_string("rst_file", "tets_rst.txt", "gt_dir")

#$$$$ SL
flags.DEFINE_string("model_type", "deeplab_mobilenet", "")#unet, deeplab_mobilenet, deeplab_resnet
flags.DEFINE_integer("rnn_mode",1, "")

flags.DEFINE_integer("continue_train",0,"")
flags.DEFINE_integer("pass_hidden",0,"")
flags.DEFINE_integer("seq_label",0,"")
# flags.DEFINE_integer("teacher_mode",0,"")

flags.DEFINE_integer("test_video",0,"")
flags.DEFINE_integer("spec_test",0,"") # ******

# flags.DEFINE_integer("fold_id",0, "")

flags.DEFINE_string("gpu", '0', "gpu")
FLAGS = flags.FLAGS
 

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
def main(_):
    continue_train = False if FLAGS.continue_train==0 else True
    pass_hidden = False if FLAGS.pass_hidden==0 else True
    seq_label = False if FLAGS.seq_label==0 else True
    test_video = False if FLAGS.test_video==0 else True
    spec_test = False if FLAGS.spec_test==0 else True # ******
    # teacher_mode = False if FLAGS.teacher_mode==0 else True

    color_table = load_color_table('./labels.json')
    run_config = tf.ConfigProto()
    sess=tf.Session(config=run_config)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        net = DeepLab(
              sess,
              input_width=FLAGS.input_width,
              input_height=FLAGS.input_height,
              crop_width=FLAGS.crop_width,
              crop_height=FLAGS.crop_height,
              batch_size=1,
              seed=23,
              temporal_len=FLAGS.temporal_len,
              img_pattern="*.jpg",
              label_pattern="*.png",
              checkpoint_dir=FLAGS.checkpoint_dir,
              save_checkpoint_dir="",
              pretrain_dir='',
              datasets=FLAGS.datasets,
              train_dataset=FLAGS.train_dataset,
              frame_dataset=FLAGS.frame_dataset,
              video_dir=FLAGS.video_dir,
              continue_train=continue_train, ###
              pass_hidden=pass_hidden,
              seq_label=seq_label,
              teacher_mode=False,
              disable_gcn=False,
              model_type=FLAGS.model_type,
              rnn_mode=FLAGS.rnn_mode,
              learning_rate=0.0005,
            #   fold_id=FLAGS.fold_id,
              num_class=2,
              color_table=color_table,
              test_video=FLAGS.test_video,is_train=False)
        if not net.load(net.checkpoint_dir)[0]:
            raise Exception("Cannot find checkpoint!")
        
        if not net.test_video:
        
            #test on train
            img_dir = FLAGS.img_dir
            frames_dir = FLAGS.frame_dataset
            rst_dir = FLAGS.rst_dir
            gt_dir = FLAGS.gt_dir
            
            if not os.path.exists(rst_dir):
                os.makedirs(rst_dir)
    
            # ******
            if spec_test:
                output_dir = os.path.join(rst_dir,"output")
                os.makedirs(output_dir)
                softmax_dir = os.path.join(rst_dir,"softmax")
                os.makedirs(softmax_dir)
                if FLAGS.rnn_mode==1: 
                    attention_dir = os.path.join(rst_dir,"attention")
                    os.makedirs(attention_dir)
            # ******
            
            files=os.listdir(img_dir)
            cum_time = []
            for i,file in enumerate(files):
                if not file.endswith(".jpg"):
                    continue

                if file[:3] not in net.valid_data and not 'ep' in net.datasets:
                    continue
                elif (file[:3] not in net.valid_data and file[:4] not in net.valid_data) and 'ep' in net.datasets:
                    continue
                
                if net.rnn_mode==0:
                    img = cv2.imread(os.path.join(img_dir,file))
                    # ******
                    if spec_test:
                        idxmap,colormap,softmax,t = net.inference(img, extra_output=True)
                    else:
                        idxmap,colormap,t = net.inference(img)
                    # ******
                elif net.rnn_mode==1:
                    seq = sequence_read(os.path.join(img_dir,file),frames_dir,net.temporal_len,net.frame_interval,
                                resize_wh=(net.input_width,net.input_height),nearest_interpolate=False)
                    # if net.pass_hidden:
                    #     prev_name, prev_idx = file[:-4].split("_")
                    #     prev_name = prev_name + "_" + str(int(prev_idx)-net.temporal_len*net.frame_interval) + ".jpg"
                    #     seq_prev = sequence_read(os.path.join(img_dir,prev_name),frames_dir,net.temporal_len,net.frame_interval,
                    #             resize_wh=(net.input_width,net.input_height),nearest_interpolate=False)
                    #     idxmap,colormap,h_prev,t = net.inference(seq,img_prev=seq_prev)
                    # else:
                    # ******
                    if spec_test:
                        idxmap,colormap,softmax,t = net.inference(seq, extra_output=True)
                    else:
                        idxmap,colormap,t = net.inference(seq)
                    # ******
                
                colormap=cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
                # ******
                if spec_test:
                    cv2.imwrite(os.path.join(output_dir,file[:-4]+'.png'),colormap) 
                    cv2.imwrite(os.path.join(softmax_dir,file[:-4]+'.png'),softmax) 
                else:
                    cv2.imwrite(os.path.join(rst_dir,file[:-4]+'.png'),colormap) 
                # ******

            if not spec_test:# ******
                cum_time.append(t)
                evaluate_seg_result(rst_dir, gt_dir, save_name=FLAGS.rst_file, cum_time=cum_time)
        
        elif not 'L' in net.valid_data[0] and not 'ep' in net.datasets:
            img_dir = FLAGS.img_dir
            rst_dir = FLAGS.rst_dir
            gt_dir = FLAGS.gt_dir

            seg_video_path = "seg_videos"

            if not os.path.exists(rst_dir):
                os.makedirs(rst_dir)
            
            files=os.listdir(gt_dir)
            cum_time = []
            for videofile in net.valid_data:
                # vw = cv2.VideoWriter(os.path.join(seg_video_path, videofile+'.avi'),cv2.VideoWriter_fourcc('P','I','M','1'),10,(240,240))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                vw = cv2.VideoWriter(os.path.join(seg_video_path, videofile+'.avi'),fourcc, 20.0,(240,240))

                prev_output = np.zeros((net.batch_size,net.crop_height,net.crop_width),dtype=np.float32)
                h_prev = np.zeros((net.batch_size,1,net.h_size[0]*net.h_size[1],net.h_size[2]),dtype=np.float32)
                h_list = np.zeros((net.batch_size,net.temporal_len,net.h_size[0]*net.h_size[1],net.h_size[2]),dtype=np.float32)

                cap = cv2.VideoCapture(os.path.join(net.video_dir,videofile+'.mp4'))
                count = 0
                while True:
                    if count == 0:
                        idxt = 1
                    else:
                        idxt = count*net.frame_interval
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idxt-1)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if idxt-1 == 0:
                        zero_init=True 
                    else:
                        zero_init=False

                    labelfile = videofile+"_"+str(idxt)+".png"
  
                    idxmap,colormap,h_prev,t = net.inference([frame[:,40:280,:]],h_prev=h_prev,prev_output=prev_output,zero_init=zero_init)
                    
                    prev_output = idxmap[np.newaxis,...]
                    if videofile==net.valid_data[0] and idxt>50 and t<50:
                        cum_time.append(t)
                    
                    colormap=cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
                    if labelfile in files:
                        # print(labelfile)
                        cv2.imwrite(os.path.join(rst_dir,labelfile),colormap)
                    count += 1

                    vw.write(colormap)
                vw.release()

            if not spec_test:
                evaluate_seg_result(rst_dir, gt_dir, save_name=FLAGS.rst_file, cum_time=cum_time)
                print("mean_time:{}ms".format(np.mean(cum_time)))

        elif 'L' in net.valid_data[0]:
            img_dir = FLAGS.img_dir
            rst_dir = FLAGS.rst_dir
            gt_dir = FLAGS.gt_dir
            data_list = {'L01':['L01_0','L01_17589'],'L02':['L02_0','L02_18771','L02_39613','L02_61235','L02_83558'],'L03':['L03_0']}
            frame_offsets = {'L01':[[156,516],[154,514]],'L02':[[156,516],[156,516],[158,518],[126,486],[151,511]],'L03':[[154,514]]}

            if not os.path.exists(rst_dir):
                os.makedirs(rst_dir)
            
            files=os.listdir(gt_dir)
            cum_time = []
            # for videofile in net.valid_data:
            #     print(videofile)
            #     prev_output = np.zeros((net.batch_size,net.crop_height,net.crop_width),dtype=np.float32)
            #     h_prev = np.zeros((net.batch_size,1,net.h_size[0]*net.h_size[1],net.h_size[2]),dtype=np.float32)
            #     h_list = np.zeros((net.batch_size,net.temporal_len,net.h_size[0]*net.h_size[1],net.h_size[2]),dtype=np.float32)

            #     vcap_list = []
            #     for vname in data_list[videofile]:
            #         vcap_list.append( cv2.VideoCapture(os.path.join(net.video_dir,vname+'.mp4')) )
                
            #     count = 0
            #     while True:
            #         if count == 0:
            #             full_idx = 1
            #         else:
            #             full_idx = count*net.frame_interval

            #         # select corresponding videos
            #         if videofile == 'L01':
            #             if full_idx < 17589:
            #                 offset_idx = 0
            #                 cap = vcap_list[0]
            #                 x1,x2 = frame_offsets[videofile][0]
            #             else:
            #                 offset_idx = 17589
            #                 cap = vcap_list[1]
            #                 x1,x2 = frame_offsets[videofile][1]
            #         elif videofile == 'L02':
            #             if full_idx < 18771:
            #                 offset_idx = 0
            #                 cap = vcap_list[0]
            #                 x1,x2 = frame_offsets[videofile][0]
            #             elif full_idx < 39613:
            #                 offset_idx = 18771
            #                 cap = vcap_list[1]
            #                 x1,x2 = frame_offsets[videofile][1]
            #             elif full_idx < 61235:
            #                 offset_idx = 39613
            #                 cap = vcap_list[2]
            #                 x1,x2 = frame_offsets[videofile][2]
            #             elif full_idx < 83558:
            #                 offset_idx = 61235 
            #                 cap = vcap_list[3]
            #                 x1,x2 = frame_offsets[videofile][3]
            #             else:
            #                 offset_idx = 83558 
            #                 cap = vcap_list[4]
            #                 x1,x2 = frame_offsets[videofile][4]
            #         else:
            #             offset_idx = 0
            #             cap = vcap_list[0]
            #             x1,x2 = frame_offsets[videofile][0]

            #         labelfile = videofile+"_"+str(full_idx)+".png"

            #         idxt = full_idx-offset_idx
            #         cap.set(cv2.CAP_PROP_POS_FRAMES, idxt)
            #         ret, frame = cap.read()
            #         if labelfile in files:
            #             print(labelfile)
            #             frame = cv2.imread(os.path.join(img_dir,videofile+"_"+str(full_idx)+".jpg"))
            #         elif not ret:
            #             break
            #         else:
            #             frame = frame[:,x1:x2]
            #             frame = cv2.resize(frame,(240,240))

            #         if full_idx-1 == 0:
            #             zero_init=True 
            #         else:
            #             zero_init=False

            #         idxmap,colormap,h_prev,t = net.inference([frame],h_prev=h_prev,prev_output=prev_output,zero_init=zero_init)
            #         prev_output = idxmap[np.newaxis,...]
            #         cum_time.append(t)

                    
            #         if labelfile in files:
            #             # print(labelfile)
            #             colormap=cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
            #             cv2.imwrite(os.path.join(rst_dir,labelfile),colormap)
            #         count += 1

            if not spec_test:
                # evaluate_seg_result(rst_dir, gt_dir, FLAGS.rst_file)
                evaluate_seg_result(rst_dir, gt_dir, save_name=FLAGS.rst_file, cum_time=cum_time)
                print("mean_time:{}ms".format(np.mean(cum_time)))

        elif 'ep' in net.datasets:
            img_dir = FLAGS.img_dir
            rst_dir = FLAGS.rst_dir
            gt_dir = FLAGS.gt_dir

            if not os.path.exists(rst_dir):
                os.makedirs(rst_dir)
            
            files=os.listdir(gt_dir)
            cum_time = []
            for videofile in net.valid_data:
                prev_output = np.zeros((net.batch_size,net.crop_height,net.crop_width),dtype=np.float32)
                h_prev = np.zeros((net.batch_size,1,net.h_size[0]*net.h_size[1],net.h_size[2]),dtype=np.float32)
                h_list = np.zeros((net.batch_size,net.temporal_len,net.h_size[0]*net.h_size[1],net.h_size[2]),dtype=np.float32)

                cap = cv2.VideoCapture(os.path.join(net.video_dir,'Proctocolectomy'+str(int(videofile[2:]))+'.avi'))
                
                count = 0
                while True:
                    if count == 0:
                        idxt = 1
                    else:
                        idxt = count*net.frame_interval
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idxt-1)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if idxt-1 == 0:
                        zero_init=True 
                    else:
                        zero_init=False

                    labelfile = videofile+"_"+str(idxt)+".png"
                    print(labelfile)
                    frame = cv2.imread(os.path.join(img_dir,labelfile[:-4]+".jpg"))
  
                    idxmap,colormap,h_prev,t = net.inference([frame],h_prev=h_prev,prev_output=prev_output,zero_init=zero_init)
                    prev_output = idxmap[np.newaxis,...]
                    if videofile==net.valid_data[0] and idxt>50 and t<50:
                        cum_time.append(t)
                    
                    if labelfile in files:
                        # print(labelfile)
                        colormap=cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(rst_dir,labelfile),colormap)
                    count += 1

            if not spec_test:
                evaluate_seg_result(rst_dir, gt_dir, save_name=FLAGS.rst_file, cum_time=cum_time)
                print("mean_time:{}ms".format(np.mean(cum_time)))
if __name__ == '__main__':
  tf.app.run()
