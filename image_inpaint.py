import os
import cv2 
import numpy as np
# import random
import imutils
import copy
# import tensorflow.compat.v1 as tf
# np.random.seed(0)

# #########
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# def gen_tf_ex(imgs,segs,ex_name,gt_valid_id,writer,size,img_num):
#     W,H = size

#     concat_view = np.concatenate(imgs,axis=1)
#     if segs is not None:
#         concat_seg = np.concatenate(segs,axis=1)
#     else:
#         concat_seg = np.zeros((H,W*img_num,1))
    
#     # gt_valid_id = 0
#     good_seg = np.zeros((H,W,1))
    
#     h, w, c = concat_view.shape
#     if not (h==H and w==W*img_num and c==3): 
#         print('STOP:',concat_view.shape,concat_depth.shape)
#         return writer
    
#     # CONVERT THE VARIABLES TO THE TARGET TYPE: IMPORTANT!!!
#     concat_view = concat_view.astype(np.uint8)
#     good_seg = good_seg.astype(np.float32)
#     concat_seg = concat_seg.astype(np.float32)
#     gt_valid_id = np.array([gt_valid_id]).astype(np.int32)

#     bbox_seq = np.zeros(img_num*4)
#     # print(concat_seg.shape,concat_view.shape)
    
#     example = tf.train.Example(features=tf.train.Features(feature={
#                 'image_seq': _bytes_feature(concat_view.tostring()),
#                 'good_seg': _bytes_feature(good_seg.tostring()),
#                 'seg_seq': _bytes_feature(concat_seg.tostring()),
#                 'gt_valid_id': _bytes_feature(gt_valid_id.tostring()),
#                 'bbox_seq': _bytes_feature(bbox_seq.tostring()),
#                 # 'bbox_segs': _bytes_feature(bbox_segs.tostring()),
#                 # 'edge_seq': _bytes_feature(edge_seq.tostring()),
#                 'seq_name': _bytes_feature(str.encode(ex_name)),
#                 }))
    
#     writer.write(example.SerializeToString())
#     return writer
# #########

def distance(p1,p2): 
    p1 = np.squeeze(p1)
    p2 = np.squeeze(p2)
    
    diff = p1-p2
    if len(diff.shape) == 2:
        return np.linalg.norm(diff,axis=1)
    else:
        return np.linalg.norm(diff)

def ext_endo_pos(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    endo_pos = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,param1=50,param2=30,minRadius=100,maxRadius=150)
    if endo_pos is None:
        return (np.nan,np.nan,np.nan)
    else:
        endo_pos = endo_pos[0][0]
        x,y,r = endo_pos
        return (x,y,r)

def genMask(endo_pos,img):
    cx, cy, r = endo_pos
    mask = np.zeros_like(img)
    if np.isnan(cx):
        return 1-mask
    cv2.circle(mask,(int(cx),int(cy)), int(r), (1,1,1), -1)
    return mask

def shift_image(img,dx,dy):
    rows, cols, _ = img.shape
    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

#####

def postproc_image(image,resize_wh=None, nearest_interpolate=True):
  if resize_wh is not None:
      if nearest_interpolate:
          image = cv2.resize(image,resize_wh,interpolation=cv2.INTER_NEAREST)
      else:
          image = cv2.resize(image,resize_wh)
  return image

# def inpaint_image(imagefile, labelfile, inpaint_dir, seg_length, resize_wh=None, angle_range=(30,60),shift_range=(10,30,30,80),bg_max_shift=40):
def inpaint_image(imagefile, labelfile, inpaint_dir, seg_length, resize_wh=None, angle_range=(30,40),shift_range=(10,30,20,60),bg_max_shift=40):
    if 'EP' in os.path.basename(imagefile):
        angle_range=(30,40)
        shift_range=(10,30,20,60)
        bg_max_shift=40

    sample_num = seg_length//2
    # sample_num = 0 # $$$$$$$$$
    # shuffle_idx = np.random.choice(seg_length,seg_length,replace=False)

    image = cv2.imread(imagefile)

    basename = os.path.basename(imagefile)[:-4]
    inpaint_img = cv2.imread(os.path.join(inpaint_dir,basename+"_bg.jpg"))
    height,width,_ = inpaint_img.shape
    width = int(width/2)
    inpaint_img,edge_mask = inpaint_img[:,:width],inpaint_img[:,width:]

    # shift bg
    _bg_seq = []
    # generate bw and fw translation (max: 0~20 pixels)
    max_shift = np.random.random()*bg_max_shift
    dx,dy = np.random.random(2)*max_shift-max_shift/2
    dx_list = np.linspace(-dx,dx,seg_length)
    dy_list = np.linspace(-dy,dy,seg_length)
    for dx,dy in zip(dx_list,dy_list):
        shift_bg = shift_image(inpaint_img,dx,dy)
        _bg_seq.append(shift_bg*edge_mask)
    # # randomly shuffle bg seq
    # shuffle_idx = np.random.choice(seg_length,seg_length,replace=False)
    # _bg_seq = [_bg_seq[i] for i in shuffle_idx]

    # cv2.imwrite(str(file_id)+"_bg_seq.jpg",np.concatenate(_bg_seq,axis=1))
        
    new_label = cv2.imread(os.path.join(inpaint_dir,basename+"_label.png"))
    inpaint_inst = cv2.imread(os.path.join(inpaint_dir,basename+"_inst.jpg"))

    if new_label is None:
        # read the label image
        label = cv2.imread(labelfile)
        if np.sum(label)>0:
            # print("## bad image ##",os.path.basename(imagefile))
            return None, None, None

        _bg_seq[sample_num] = image*edge_mask
        
        _label_seq = [np.zeros((height,width,1))]*seg_length 
        #[np.zeros((height,width,1)) for i in range(seg_length)]
        _new_label = label*edge_mask
        _label_seq[sample_num] = _new_label[:,:,0][...,np.newaxis]
        
        _bg_seq = [postproc_image(_bg_seq[i],resize_wh=resize_wh) for i in range(seg_length)]
        _label_seq = [postproc_image(_label_seq[i],resize_wh=resize_wh) for i in range(seg_length)]
        # _bg_seq = [postproc_image(_bg_seq[i],resize_wh=resize_wh) for i in shuffle_idx]
        # _label_seq = [postproc_image(_label_seq[i],resize_wh=resize_wh) for i in shuffle_idx]
        # gt_valid_id = np.squeeze(np.argwhere(np.array(shuffle_idx)==sample_num))
        gt_valid_id = sample_num
        return _bg_seq, _label_seq, gt_valid_id

    cv2.imwrite(os.path.join("./samples",basename+".png"),new_label/np.max(new_label)*255)

    if 'EP' in os.path.basename(imagefile):
        new_width = width*1.2
        new_height = height*1.2
    else:
        new_width = width*2
        new_height = height*2
    # randomly rotate and shift instrument
    if np.random.random() < 0.0:
        _angle_range = angle_range[0]
        _shift_min, _shift_max = shift_range[0], shift_range[1]
    else:
        _angle_range = angle_range[1]
        _shift_min, _shift_max = shift_range[2], shift_range[3]

    # randomly shift every instruments
    new_label_all = copy.deepcopy(new_label)
    inst_ids = np.unique(new_label_all[new_label_all>0])
    _img_seq = copy.deepcopy(_bg_seq)
    _label_seq = [np.zeros((height,width))]*seg_length
    for inst_id in inst_ids:
        new_label = np.zeros_like(new_label_all)
        new_label[new_label_all==inst_id] = 1

        # generate bw and fw angle (max: ranges -40~40 degree)
        max_degree = np.random.random()*_angle_range-_angle_range/2
        ang_bw = np.random.random()*max_degree-max_degree 
        ang_fw = np.random.random()*max_degree
        ang_list = list(np.linspace(ang_bw,0,sample_num+1)[:-1])+list(np.linspace(0,ang_fw,sample_num+1))
        # generate bw and fw translation (max: 10~50 pixels)
        max_shift = np.random.random()*(_shift_max-_shift_min)+_shift_min
        dx,dy = np.random.random(2)*max_shift-max_shift/2
        dx_list = np.linspace(-dx,dx,seg_length)
        dy_list = np.linspace(-dy,dy,seg_length)
    #
    
        _frame_id = 0
        for angle,dx,dy,bg_img in zip(ang_list,dx_list,dy_list,_img_seq):

            _inpaint_inst = imutils.rotate(inpaint_inst, angle)
            _inpaint_inst = shift_image(_inpaint_inst,dx,dy)
            # cv2.imwrite("inpaint_inst.jpg",_inpaint_inst)

            _new_label = imutils.rotate(new_label, angle)
            _new_label = shift_image(_new_label,dx,dy)
            label_dx,label_dy = int((new_width-width)/2),int((new_height-height)/2)
            _new_label = _new_label[label_dy:label_dy+height,label_dx:label_dx+width]
            _new_label = _new_label*edge_mask
            # cv2.imwrite("new_label.jpg",_new_label*255)

            # if np.sum(_new_label[:,:,0]) < 8000 and np.random.random() < 0.4 and not _frame_id == sample_num:
            #     _new_label = np.zeros_like(_new_label)
            #     _inpaint_img = bg_img
            # else:
            #     _inpaint_inst = _inpaint_inst[label_dy:label_dy+height,label_dx:label_dx+width]
            #     _smooth_new_label = cv2.GaussianBlur(_new_label.astype(np.float32),(3,3),3)
            #     _inpaint_img = bg_img * (1-_smooth_new_label) + _inpaint_inst * _smooth_new_label
            _inpaint_inst = _inpaint_inst[label_dy:label_dy+height,label_dx:label_dx+width]
            _smooth_new_label = cv2.GaussianBlur(_new_label.astype(np.float32),(3,3),3)
            _inpaint_img = bg_img * (1-_smooth_new_label) + _inpaint_inst * _smooth_new_label
            # sample = np.concatenate([image,inpaint_img],axis=1)
            # cv2.imwrite("inpaint_results.jpg",sample)

            if _frame_id == sample_num:
                _inpaint_img = image*edge_mask
                _new_label = cv2.imread(labelfile)*edge_mask
                # _new_label = _new_label*edge_mask
            
            if np.random.random() < 0.6:
                _inpaint_img = np.clip(_inpaint_img + np.random.randint(10,60)*1.0,0.,255.)
                _inpaint_img = _inpaint_img*edge_mask
            _img_seq[_frame_id] = _inpaint_img#*final_edge_mask

            _new_label = _new_label[:,:,0] 
            _new_label = _new_label+_label_seq[_frame_id]
            _new_label[_new_label>0]=1
            _label_seq[_frame_id] = _new_label#*final_edge_mask[:,:,0][...,np.newaxis]

            # if np.sum((np.mean(_inpaint_img,axis=2)*_new_label[:,:,0])>240)/(np.sum(_new_label[:,:,0])+0.00000001) >= 0.6:
            #     save_seq += 1
            _frame_id+=1

    # randomly shuffle img and seg seq
    _img_seq = [postproc_image(_img_seq[i],resize_wh=resize_wh) for i in range(seg_length)]
    _label_seq = [postproc_image(_label_seq[i][...,np.newaxis],resize_wh=resize_wh) for i in range(seg_length)]
    # _img_seq = [postproc_image(_img_seq[i],resize_wh=resize_wh) for i in shuffle_idx]
    # _label_seq = [postproc_image(_label_seq[i],resize_wh=resize_wh) for i in shuffle_idx]
    # gt_valid_id = np.squeeze(np.argwhere(np.array(shuffle_idx)==sample_num))
    gt_valid_id = sample_num
    return _img_seq, _label_seq, gt_valid_id