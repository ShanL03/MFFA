model=deeplab_resnet  
pretrain_path=../pretrain/resnet_v2_50_2017_04_14
img_dir=../sinus_data/cadaver/images
gt_dir=../sinus_data/cadaver/labels
video_dir=../sinus_data/cadaver/videos

############ mffa-cc1 ############
rst_subfix=-lc1
datasets=lc1

subfix3=res2_ts_noCoarse_20syn_20real-lc1
subfix4=res2_ts_noCoarse_20real_20real-lc1

# python test.py --test_video 1 --pass_hidden 0 --img_dir $img_dir --gt_dir $gt_dir --video_dir $video_dir --datasets $datasets \
# --model_type $model --checkpoint_dir ./rsts/ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 \
# --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt

python test.py --test_video 1 --pass_hidden 0 --img_dir $img_dir --gt_dir $gt_dir --video_dir $video_dir --datasets $datasets \
--model_type $model --checkpoint_dir ./rsts/ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 \
--rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt