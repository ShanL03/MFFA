# @Time     : Jan. 12, 2019 17:45
# @Author   : Veritas YIN
# @FileName : layers.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import tensorflow as tf
import numpy as np

try:
    from .ops import *
    from .utils import *
except:    
    from ops import *
    from utils import *

def normalize_adj(feat_array,h_size,is_train,kernel=None,mode="attn",scope=""):
# def normalize_adj(kernel,prev_output,feat_array,h_size,is_train,mode="attn",scope=""):
    # if prev_output is not None:
    #     prev_output = tf.cast(prev_output,tf.float32)[:,0,:,:]
    #     prev_output = tf.reshape(prev_output,[-1,h_size[0]*h_size[1],1])
    #     prior_kernel = tf.matmul(prev_output,prev_output,transpose_b=True) \
    #                    + tf.matmul(1-prev_output,1-prev_output,transpose_b=True)
    #     # print("#########",prev_output)

    # n = kernel.get_shape()[1].value
    batch_size = feat_array.get_shape()[0].value
    # feat_array = feat_array[:,:,:n,:]
    if mode=="attn" or mode=="comb":
        with tf.variable_scope(scope) as scope:
            f_chn = int(h_size[2]/2)
            # feature array size: batch_siz x time x pts num x feat_chn
            # conv feature array
            f1 = tf.reshape(feat_array[:,0,:,:],[batch_size,h_size[0],h_size[1],-1]) # h_size[2]+1
            f1 = separable_conv2d(f1,f_chn,ksize=1,name="norm_conv_out1")
            f1 = tf.nn.relu(batchnorm(f1,is_train,'norm_bn_out1'))
            f1 = tf.reshape(f1,[batch_size,-1,f_chn])
            # f1 = tf.reshape(feat_array[:,0,:,:],[batch_size,-1,h_size[2]])
            #
            f2 = tf.reshape(feat_array[:,0,:,:],[batch_size,h_size[0],h_size[1],-1]) # h_size[2]+1
            f2 = separable_conv2d(f2,f_chn,ksize=1,name="norm_conv_out2")
            f2 = tf.nn.relu(batchnorm(f2,is_train,'norm_bn_out2'))
            f2 = tf.reshape(f2,[batch_size,-1,f_chn])
            # f2 = tf.reshape(feat_array[:,0,:,:],[batch_size,-1,h_size[2]])
            #
            sp_attn = tf.matmul(f1,f2,transpose_b=True) # size: batch_size x num x num
            # if prev_output is not None:
            #     sp_attn = sp_attn*prev_output
            # if mode=="comb":
            #     kernel = tf.math.multiply(sp_attn,kernel)
            # else:
            #     kernel = sp_attn
            # kernel = tf.nn.softmax(kernel,axis=1)
            kernel = tf.nn.softmax(sp_attn,axis=1)
    elif mode=="dist":
        kernel = tf.tile(tf.expand_dims(kernel,axis=0),[batch_size,1,1])

    D_minus_sqrt = tf.matrix_diag(tf.math.sqrt(1.0/tf.math.reduce_sum(kernel,axis=-1)))
    kernel = tf.linalg.matmul(tf.linalg.matmul(D_minus_sqrt,kernel),D_minus_sqrt)
    
    return kernel#,feat_array

# def normalize_tp_adj(kernel,feat_array,h_size,is_train,mode="attn",scope=""):
#     T = feat_array.get_shape()[1].value
#     n_route = feat_array.get_shape()[2].value
#     feat_array = tf.concat([feat_array[:,i] for i in range(T)],axis=1)

#     # n = kernel.get_shape()[1].value
#     batch_size = feat_array.get_shape()[0].value
#     # feat_array = feat_array[:,:n,:]
#     if mode=="attn" or mode=="comb":
#         with tf.variable_scope(scope) as scope:
#             # feature array size: batch_siz x time x pts num x feat_chn
#             # conv feature array
#             f1 = tf.reshape(feat_array,[batch_size,h_size[0]*2,h_size[1],-1]) # h_size[2]+1
#             f1 = separable_conv2d(f1,64,ksize=1,name="norm_conv_out1")
#             f1 = tf.nn.relu(batchnorm(f1,is_train,'norm_bn_out1'))
#             f1 = tf.reshape(f1,[batch_size,-1,64])
#             #
#             f2 = tf.reshape(feat_array,[batch_size,h_size[0]*2,h_size[1],-1]) # h_size[2]+1
#             f2 = separable_conv2d(f2,64,ksize=1,name="norm_conv_out2")
#             f2 = tf.nn.relu(batchnorm(f2,is_train,'norm_bn_out2'))
#             f2 = tf.reshape(f2,[batch_size,-1,64])
#             #
#             sp_attn = tf.matmul(f1,f2,transpose_b=True) # size: batch_size x num x num
#             if mode=="comb":
#                 kernel = tf.math.multiply(sp_attn,kernel)
#             else:
#                 kernel = sp_attn
#             kernel = tf.nn.softmax(kernel,axis=1)
#     elif mode=="dist":
#         kernel = tf.tile(tf.expand_dims(kernel,axis=0),[batch_size,1,1])

#     D_minus_sqrt = tf.matrix_diag(tf.math.sqrt(1.0/tf.math.reduce_sum(kernel,axis=-1)))
#     kernel = tf.linalg.matmul(tf.linalg.matmul(D_minus_sqrt,kernel),D_minus_sqrt)
    
#     feat_array = tf.transpose(feat_array, [0, 2, 1])
#     feat_array = tf.matmul(feat_array, kernel)
#     # x_ker -> [batch_size, n_route, c_in]
#     feat_array = tf.transpose(feat_array, [0, 2, 1])
#     feat_array = tf.concat([tf.expand_dims(feat_array[:,i*n_route:(i+1)*n_route,:],axis=1) for i in range(T)],axis=1)

#     return feat_array

def gconv(x, kernel, theta, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    # kernel = tf.get_collection('graph_kernel')[0]
    # n = tf.shape(kernel)[0]
    n = tf.shape(x)[1]
    # x -> [batch_size, c_in, n_route] -> [batch_size, c_in, n_route]
    x_tmp = tf.transpose(x, [0, 2, 1])
    # x_mul = x_tmp * ker -> [batch_size, c_in, n_route] -> [batch_size, c_in, n_route]
    x_mul = tf.matmul(x_tmp, kernel)
    # x_ker -> [batch_size, n_route, c_in] -> [batch_size*n_route, c_in]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 2, 1]), [-1, c_in])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv

def layer_norm(x, scope):
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def temporal_conv_layer(x, c_in, c_out, act_func='GLU'):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()
    Kt = T
    
    # if T==1:
    #     return x

    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]

    if act_func == 'GLU':
        # gated liner unit
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        # return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
        return tf.nn.relu((x_conv[:, :, :, 0:c_out]*tf.nn.sigmoid(x_conv[:, :, :, -c_out:]) + x_input))
    elif act_func == 'conf_map':
        # gated liner unit
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out+1], dtype=tf.float32)
        # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out+1]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        # return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
        return tf.nn.relu((x_conv[:, :, :, 0:c_out]*tf.nn.sigmoid(x_conv[:, :, :, -1:]) + x_input))
    else:
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')

def suboutput_conv_layer(x, c_in, c_out):
    _, T, n, _ = x.get_shape().as_list()
    Kt = T

    wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
    bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
    x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
    # return tf.nn.relu(x_conv)
    return tf.nn.softmax(x_conv)


def spatio_conv_layer(x, kernel, c_in, c_out):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x
    # c_in = c_out

    ws = tf.get_variable(name='ws', shape=[c_in, c_out], dtype=tf.float32)
    # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    # variable_summaries(ws, 'theta')
    bs = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
    # x_gconv = gconv(tf.reshape(x_input, [-1, n, c_in]), kernel, ws, c_in, c_out) + bs
    x_gconv = gconv(x_input[:,-1,:,:], kernel, ws, c_in, c_out) + bs
    # x_g -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
    return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)


def st_conv_block(x, channels, kernel_sp=None, kernel_tp=None, h_size=None, glb_map=None, is_train=True, output_sub=True, scope="gcn", gcn_mode=0, act_func='GLU'):#Ks, Kt
    '''
    Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    :param x: tensor, batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param channels: list, channel configs of a single st_conv block.
    :param scope: str, variable scope.
    :param keep_prob: placeholder, prob of dropout.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    with tf.variable_scope('',reuse=tf.AUTO_REUSE) as sp:
        if gcn_mode==0:
            for i in range(1,len(channels)):
                # norm_kernel_sp,x = normalize_adj(kernel_sp,x,h_size,is_train,scope="norm_sp")
                norm_kernel_sp,x = normalize_adj(x,h_size,is_train,scope="norm_sp")
                with tf.variable_scope(f'stn_block_{scope}_'+str(i)):
                    x = spatio_conv_layer(x, norm_kernel_sp, channels[i-1], channels[i])
                    # x = spatio_conv_layer(x, kernel_sp, channels[i-1], channels[i])

                    if i==len(channels)-1:
                        x_t = x

            return x_t, tf.nn.softmax(x)
        # elif gcn_mode==1:
        #     n = kernel_sp.get_shape()[1].value
        #     # try old code:
        #     # norm_kernel_sp,_ = normalize_adj(kernel_sp,x,h_size,is_train,scope="norm_sp")
        #     if kernel_tp is not None:
        #         norm_kernel_tp,_ = normalize_adj(kernel_tp,x,h_size,is_train,scope="norm_tp")
        #         with tf.variable_scope(f'stn_block_{scope}_temperal'):
        #             # norm_kernel_tp,x = normalize_adj(kernel_tp,x,h_size,is_train)
        #             x = spatio_conv_layer(x, norm_kernel_tp, channels[0], channels[1])
        #             # x = spatio_conv_layer(x, kernel_tp, channels[0], channels[1])
        #             # x = x[:,:,:n,:]
        #     with tf.variable_scope(f'stn_block_{scope}_spatial'):
        #         norm_kernel_sp,x = normalize_adj(kernel_sp,x,h_size,is_train,scope="norm_sp")
        #         # norm_kernel_sp,x = normalize_adj(kernel_sp,x,h_size,is_train)
        #         x = spatio_conv_layer(x, norm_kernel_sp, channels[1], channels[2])
        #         # x = spatio_conv_layer(x, kernel_sp, channels[1], channels[2])
        #         x_t = x
        #     return x_t #s, tf.nn.softmax(x)
        elif gcn_mode==2:
            sub_softmax = None
            
            batch_size, T, _, _ = x.get_shape().as_list()
            if T > 1:
                # # global map
                # _x = tf.concat([x[:,:1],glb_map],axis=-1)
                # prev_x = tf.concat([x[:,1:],glb_map],axis=-1)
                # with tf.variable_scope(f'stn_block_{scope}_inst_temperal'):
                #     x = tf.concat([prev_inst,x],axis=1)
                #     x = temporal_conv_layer(x, channels[0], channels[1])

                with tf.variable_scope(f'stn_block_{scope}_temperal'):
                    # x = normalize_tp_adj(kernel_tp,x,h_size,is_train,scope="norm_tp")
                    x = temporal_conv_layer(x, channels[0], channels[1], act_func='conf_map') # act_func='conf_map', act_func='relu',channels[0]+1
                if output_sub:
                    with tf.variable_scope(f'stn_block_{scope}_suboutput'):
                        sub_softmax = suboutput_conv_layer(x, channels[1], 3)
            with tf.variable_scope(f'stn_block_{scope}_spatial_a'):
                # x = tf.concat([x,prev_output],axis=-1)
                # norm_kernel_sp,x = normalize_adj(kernel_sp,prev_output,x,h_size,is_train,scope="norm_sp")
                norm_kernel_sp = normalize_adj(x,h_size,is_train,scope="norm_sp")
                x = spatio_conv_layer(x, norm_kernel_sp, channels[1], channels[2])
                _sub_softmax = suboutput_conv_layer(x, channels[2], 3)
            with tf.variable_scope(f'stn_block_{scope}_spatial_refine'):
                x = tf.reshape(x,[batch_size]+h_size)
                x_inputs = x

                _sub_softmax = tf.reshape(_sub_softmax,[batch_size]+h_size[:2]+[3])
                x = tf.concat([x,_sub_softmax],axis=3)
                x = conv2d(x,channels[2],ksize=3,stride=1,name=scope+"res_conv_1") 
                x = tf.nn.relu(x) #tf.nn.relu(batchnorm(x,is_train,scope+'res_bn_1'))
                x = conv2d(x,channels[2],ksize=3,stride=1,name=scope+"res_conv_2") 
                x = tf.nn.relu(x) #tf.nn.relu(batchnorm(x,is_train,scope+'res_bn_2'))
                x = tf.nn.relu(x_inputs+x)
                _x = tf.reshape(x,[batch_size,1,h_size[0]*h_size[1],channels[2]])

            return x, _x, sub_softmax

def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, T, scope, act_func='GLU'):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param T: int, kernel size of temporal convolution.
    :param scope: str, variable scope.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()

    # maps multi-steps to one.
    with tf.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    # maps multi-channels to one.
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc


def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    :param var: tf.Variable().
    :param v_name: str, name of the variable.
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{v_name}', mean)

        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{v_name}', stddev)

        tf.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{v_name}', tf.reduce_min(var))

        tf.summary.histogram(f'histogram_{v_name}', var)