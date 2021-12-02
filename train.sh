epoch_num=20
epoch_decay_num1=20
epoch_decay_num2=5
cont_learning_rate=0.0005

model=deeplab_resnet  
pretrain_path=../pretrain/resnet_v2_50_2017_04_14

############ mffa-cc1 ############
rst_subfix=-lcfull
datasets=lc1lc2lc3

subfix3=res2_ts_noCoarse_noSingle_20syn_20real$rst_subfix
subfix4=res2_ts_noCoarse_noSingle_20real_20real$rst_subfix

# # 20 syn
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num \
# --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path \
# --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --datasets $datasets
# # + 20 real
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num \
# --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model \
# --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --datasets $datasets

# 20 real + 20 real
python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num \
--decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path \
--checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --datasets $datasets
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num \
# --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model \
# --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --datasets $datasets

#######################################
############ cc2-fam ############
rst_subfix=-cc2 #$$$$
fid=1 #$$$$
# cc1,0 cc2,1 cc3,2 lc1,3 lc2,4, lc3,5

# seq_len=2
# subfix=mob8_ts2_noCoarse_20real_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=2
# subfix=mob8_ts2_noCoarse_20syn_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=3
# subfix=mob8_ts3_noCoarse_20real_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=3
# subfix=mob8_ts3_noCoarse_20syn_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=5
# subfix=mob8_ts5_noCoarse_20real_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=5
# subfix=mob8_ts5_noCoarse_20syn_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# ###
# # subfix1=mob8_ts_20syn-cc2
# # subfix2=mob8_ts_20syn_20syn-cc2
# # subfix3=mob8_ts_20syn_20real-cc2
# # subfix4=mob8_ts_20real_20real-cc2
# # ## 20 syn ##
# # python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
# # ## +20 syn ##
# # # python train.py --continue_train 1 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix2 --rnn_mode 1 --fold_id $fid
# # # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix2/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix2.txt --fold_id $fid
# # ## +20 real ##
# # python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# # ## 20 real + 20 real ##
# # python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# # python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
# ###
# subfix1=res2_ts_20syn-cc2
# # subfix2=res2_ts_20syn_20syn-cc2
# subfix3=res2_ts_20syn_20real-cc2
# subfix4=res2_ts_20real_20real-cc2
# ## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# ## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
### ablation study
# subfix=mob8_ts_noSingle_20syn_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noCoarse_20syn_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20syn_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
## real+real
# subfix=mob8_ts_noSingle_20real_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noCoarse_20real_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20real_20real-cc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
## res2
# subfix3=res2_ts_noCoarse_20syn_20real-cc2
# subfix4=res2_ts_noCoarse_20real_20real-cc2
## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid

# #######################################
# ############ cc3-fam ############
rst_subfix=-cc3 #$$$$
fid=2 #$$$$
# cc1,0 cc2,1 cc3,2 lc1,3 lc2,4, lc3,5

# seq_len=2
# subfix=mob8_ts2_noCoarse_20real_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=2
# subfix=mob8_ts2_noCoarse_20syn_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=3
# subfix=mob8_ts3_noCoarse_20real_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=3
# subfix=mob8_ts3_noCoarse_20syn_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=5
# subfix=mob8_ts5_noCoarse_20real_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=5
# subfix=mob8_ts5_noCoarse_20syn_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# # ###
# # subfix1=mob8_ts_20syn-cc3
# # subfix2=mob8_ts_20syn_20syn-cc3
# # subfix3=mob8_ts_20syn_20real-cc3
# # subfix4=mob8_ts_20real_20real-cc3
# # ## 20 syn ##
# # python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
# # ## +20 syn ##
# # # python train.py --continue_train 1 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix2 --rnn_mode 1 --fold_id $fid
# # # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix2/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix2.txt --fold_id $fid
# # ## +20 real ##
# # python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# # ## 20 real + 20 real ##
# # python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# # python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
# ###
# subfix1=res2_ts_20syn-cc3
# # subfix2=res2_ts_20syn_20syn-cc3
# subfix3=res2_ts_20syn_20real-cc3
# subfix4=res2_ts_20real_20real-cc3
# ## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# ## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
### ablation study
# subfix=mob8_ts_noSingle_20syn_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noCoarse_20syn_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20syn_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
## real+real
# subfix=mob8_ts_noSingle_20real_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noCoarse_20real_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20real_20real-cc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
## res2
# subfix3=res2_ts_noCoarse_20syn_20real-cc3
# subfix4=res2_ts_noCoarse_20real_20real-cc3
## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid

# #######################################
# ############ lc1-fam ############
rst_subfix=-lc1 #$$$$
fid=3 #$$$$
# cc1,0 cc2,1 cc3,2 lc1,3 lc2,4, lc3,5

# ###
# subfix1=mob8_ts_20syn-lc1
# subfix2=mob8_ts_20syn_20syn-lc1
# subfix3=mob8_ts_20syn_20real-lc1
# subfix4=mob8_ts_20real_20real-lc1
# ## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
# ## +20 syn ##
# # python train.py --continue_train 1 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix2 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix2/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix2.txt --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# ## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
###
# subfix1=res2_ts_20syn-lc1
# # subfix2=res2_ts_20syn_20syn-lc1
# subfix3=res2_ts_20syn_20real-lc1
# subfix4=res2_ts_20real_20real-lc1
# ## 20 syn ##
# # python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
# ## +20 real ##
# # python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# ## 20 real + 20 real ##
# # python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# # python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
### ablation study
# subfix=mob8_ts_noSingle_20syn_20real-lc1
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noCoarse_20syn_20real-lc1
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20syn_20real-lc1
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
##
# subfix=mob8_ts_noSingle_20real_20real-lc1
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# real+real
# subfix=mob8_ts_noCoarse_20real_20real-lc1
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20real_20real-lc1
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
## res2
# subfix3=res2_ts_noCoarse_20syn_20real-lc1
# subfix4=res2_ts_noCoarse_20real_20real-lc1
## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid

#######################################
############ lc2-fam ############
rst_subfix=-lc2 #$$$$
fid=4 #$$$$
# cc1,0 cc2,1 cc3,2 lc1,3 lc2,4, lc3,5

# seq_len=2
# subfix=mob8_ts2_noCoarse_20real_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 40 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=2
# subfix=mob8_ts2_noCoarse_20syn_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch 40 --decay_epoch 40 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=3
# subfix=mob8_ts3_noCoarse_20real_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 40 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=3
# subfix=mob8_ts3_noCoarse_20syn_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch 40 --decay_epoch 40 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# ###
# subfix1=mob8_ts_20syn-lc2
# subfix2=mob8_ts_20syn_20syn-lc2
# subfix3=mob8_ts_20syn_20real-lc2
# subfix4=mob8_ts_20real_20real-lc2
# ## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch 40 --decay_epoch 40 --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
# ## +20 syn ##
# # python train.py --continue_train 1 --pass_hidden 0 --seq_label 1 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix2 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix2/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix2.txt --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# ## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 40 --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
###
# subfix1=res2_ts_20syn-lc2
# # subfix2=res2_ts_20syn_20syn-lc2
# subfix3=res2_ts_20syn_20real-lc2
# subfix4=res2_ts_20real_20real-lc2
# ## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch 40 --decay_epoch 40 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# ## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 40 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
### ablation study
# subfix=mob8_ts_noSingle_20syn_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch 40 --decay_epoch 40 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noCoarse_20syn_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch 40 --decay_epoch 40 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20syn_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch 40 --decay_epoch 40 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
## real+real
# subfix=mob8_ts_noSingle_20real_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 40 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noCoarse_20real_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 40 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20real_20real-lc2
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 40 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# ## res2
# subfix3=res2_ts_noCoarse_20syn_20real-lc2
# subfix4=res2_ts_noCoarse_20real_20real-lc2
# ## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch 40 --decay_epoch 40 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 40 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch 40 --decay_epoch 10 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid

#######################################
############ lc3-fam ############
rst_subfix=-lc3 #$$$$
fid=5 #$$$$
# cc1,0 cc2,1 cc3,2 lc1,3 lc2,4, lc3,5

# seq_len=2
# subfix=mob8_ts2_noCoarse_20real_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=2
# subfix=mob8_ts2_noCoarse_20syn_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=3
# subfix=mob8_ts3_noCoarse_20real_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# seq_len=3
# subfix=mob8_ts3_noCoarse_20syn_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --temporal_len $seq_len --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --temporal_len $seq_len --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file vid_test_rst_$subfix.txt --fold_id $fid

# ###
# subfix1=mob8_ts_20syn-lc3
# # subfix2=mob8_ts_20syn_20syn-lc3
# subfix3=mob8_ts_20syn_20real-lc3
# subfix4=mob8_ts_20real_20real-lc3
## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
## +20 syn ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix2 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix2/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix2.txt --fold_id $fid
## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
###
# subfix1=res2_ts_20syn-lc3
# # subfix2=res2_ts_20syn_20syn-lc3
# subfix3=res2_ts_20syn_20real-lc3
# subfix4=res2_ts_20real_20real-lc3
# ## 20 syn ##
# # python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix1 --rnn_mode 1 --fold_id $fid
# # python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix1/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix1.txt --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix1 --save_checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# ## 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
### ablation study
# subfix=mob8_ts_noSingle_20syn_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noCoarse_20syn_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20syn_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
## real+real
# subfix=mob8_ts_noSingle_20real_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noCoarse_20real_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
#
# subfix=mob8_ts_noSingle_noCoarse_20real_20real-lc3
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --checkpoint_dir ./ck_$subfix --rnn_mode 1 --fold_id $fid
# python test.py --test_video 0 --pass_hidden 0 --checkpoint_dir ./ck_$subfix/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix.txt --fold_id $fid
## res2
# subfix3=res2_ts_noCoarse_20syn_20real-lc3
# # subfix4=res2_ts_noCoarse_20real_20real-lc3
# # ## 20 syn ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 1 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# ## +20 real ##
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix3 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix3/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix3.txt --fold_id $fid
# 20 real + 20 real ##
# python train.py --continue_train 0 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num1 --model_type $model --pretrain_dir $pretrain_path --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python train.py --continue_train 1 --pass_hidden 0 --seq_label 0 --epoch $epoch_num --decay_epoch $epoch_decay_num2 --learning_rate $cont_learning_rate --model_type $model --checkpoint_dir ./ck_$subfix4 --rnn_mode 1 --fold_id $fid
# python test.py --test_video 1 --pass_hidden 0 --model_type $model --checkpoint_dir ./ck_$subfix4/DeepLab_16_240_240 --rnn_mode 1 --rst_dir ./test-rsts$rst_subfix --rst_file test_rst_$subfix4.txt --fold_id $fid
