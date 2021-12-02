epoch_num=20
epoch_decay_num1=20
epoch_decay_num2=5
cont_learning_rate=0.0005

model=deeplab_resnet  
pretrain_path=../pretrain/resnet_v2_50_2017_04_14
train_dataset=../sinus_data/syn
frame_dataset=../sinus_data/frame_dataset
video_dataset=../sinus_data/cadaver/videos

############ mffa-cc1 ############
rst_subfix=-lcfull
datasets=lc1lc2lc3

subfix3=res2_ts_noCoarse_noSingle_20syn_20real$rst_subfix
subfix4=res2_ts_noCoarse_noSingle_20real_20real$rst_subfix

# # 20 syn
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num \
# --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path \
# --train_dataset $train_dataset --frame_dataset $frame_dataset video_dataset $video_dataset \
# --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --datasets $datasets
# # + 20 real
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num \
# --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model \
# --train_dataset $train_dataset --frame_dataset $frame_dataset video_dataset $video_dataset \
# --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --datasets $datasets

# 20 real + 20 real
python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num \
--decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path \
--train_dataset $train_dataset --frame_dataset $frame_dataset video_dataset $video_dataset \
--checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --datasets $datasets
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num \
# --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model \
# --train_dataset $train_dataset --frame_dataset $frame_dataset video_dataset $video_dataset \
# --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --datasets $datasets