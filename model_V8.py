# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

l2 = tf.keras.regularizers.l2

def fix_adaIN(inputs, style, epsilon=1e-5):
    
    _, H, W, C = inputs.get_shape()
    C_num = C // 3
    img_buf = []

    in_mean_var = [tf.nn.moments(inputs[:, :, :, i*3:(i+1)*3], axes=[1,2], keepdims=True) for i in range(C_num)]
    in_mean = [in_mean_var[i][0] for i in range(C_num)]
    in_var = [in_mean_var[i][1] for i in range(C_num)]
    st_mean, st_var = tf.nn.moments(style, axes=[1,2], keepdims=True)

    in_std = [tf.sqrt(in_var[i] + epsilon) for i in range(C_num)]
    st_std = tf.sqrt(st_var + epsilon)

    img = [st_std * (style - in_mean[i]) / in_std[i] + st_mean for i in range(C_num)]
    img = tf.concat(img, -1)

    return img

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

# Version 7에서 짰었던 모델을 조금만더 고쳐보자 --> 메모리차지를 너무 많이한다. 코드 최적화가 필요

def conv_block(input, filters, dilation_rate, weight_decay):
    h = tf.keras.layers.Conv2D(filters=filters // 2,
                               kernel_size=1,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(input)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation_rate,dilation_rate],[dilation_rate,dilation_rate],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=filters // 2,
                               kernel_size=3,
                               strides=1,
                               dilation_rate=dilation_rate,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h += input
    h = tf.keras.layers.ReLU()(h)

    return h

def V8_generator(input_shape=(256, 256, 3), # Adaptive Instance normalization을 시키기에는 중간 layer에 너무많은 메모리를 차지한다.. 어떻게 다루어야하나...
                 style_shape=(256, 256, 3),
                 weight_decay=0.000002,
                 repeat=3):

    h = inputs = tf.keras.Input(input_shape)
    s = style = tf.keras.Input(style_shape)
    s1 = tf.image.resize(s, [128, 128])
    s2 = tf.image.resize(s, [64, 64])

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=72,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 72]

    ###################################################
    _, H, W, C = h.get_shape()
    first_h = h[:, :, :, 0:C//3]
    second_h = h[:, :, :, C//3:C*2//3]
    third_h = h[:, :, :, C*2//3:]

    h_list = (first_h, second_h, third_h)
    tf.random.shuffle(h_list)
    buf_1 = []
    for i in range(3):
        h = h_list[i]
        for _ in range(repeat):
            h = conv_block(h, (C//3), i + 1, weight_decay)

        buf_1.append(h)
    h = tf.concat(buf_1, 3)
    ###################################################

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=132,
                               kernel_size=3,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 132]

    ###################################################
    _, H, W, C = h.get_shape()
    first_h = h[:, :, :, 0:C//3]
    second_h = h[:, :, :, C//3:C*2//3]
    third_h = h[:, :, :, C*2//3:]

    h_list = (first_h, second_h, third_h)
    tf.random.shuffle(h_list)
    buf_1 = []
    for i in range(3):
        h = h_list[i]
        for _ in range(repeat):
            h = conv_block(h, (C//3), i + 1, weight_decay)
        buf_1.append(h)
    h = tf.concat(buf_1, 3)
    ###################################################

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=264,
                               kernel_size=3,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = fix_adaIN(h, s2)    #  이 부분만 추가하면된다.
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 264]

    ###################################################
    _, H, W, C = h.get_shape()
    first_h = h[:, :, :, 0:C//3]
    second_h = h[:, :, :, C//3:C*2//3]
    third_h = h[:, :, :, C*2//3:]

    h_list = (first_h, second_h, third_h)
    tf.random.shuffle(h_list)
    buf_1 = []
    for i in range(3):
        h = h_list[i]
        for _ in range(repeat - 2):
            h = conv_block(h, (C//3), i + 1, weight_decay)
        buf_1.append(h)
    h = tf.concat(buf_1, 3)
    ###################################################

    h = tf.keras.layers.Conv2DTranspose(filters=132,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = fix_adaIN(h, s1)    #  이 부분만 추가하면된다.

    ###################################################
    _, H, W, C = h.get_shape()
    first_h = h[:, :, :, 0:C//3]
    second_h = h[:, :, :, C//3:C*2//3]
    third_h = h[:, :, :, C*2//3:]

    h_list = (first_h, second_h, third_h)
    tf.random.shuffle(h_list)
    buf_1 = []
    for i in range(3):
        h = h_list[i]
        for _ in range(repeat - 2):
            h = conv_block(h, (C//3), i + 1, weight_decay)
        buf_1.append(h)
    h = tf.concat(buf_1, 3)
    ###################################################

    h = tf.keras.layers.Conv2DTranspose(filters=72,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = fix_adaIN(h, s)    #  이 부분만 추가하면된다.

    ###################################################
    _, H, W, C = h.get_shape()
    first_h = h[:, :, :, 0:C//3]
    second_h = h[:, :, :, C//3:C*2//3]
    third_h = h[:, :, :, C*2//3:]

    h_list = (first_h, second_h, third_h)
    tf.random.shuffle(h_list)
    buf_1 = []
    for i in range(3):
        h = h_list[i]
        for _ in range(repeat - 2):
            h = conv_block(h, (C//3), i + 1, weight_decay)
        buf_1.append(h)
    h = tf.concat(buf_1, 3)
    ###################################################

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7,
                               strides=1,
                               padding="valid")(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=[inputs, style], outputs=h)

def discriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):

    dim_ = dim
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)


    return tf.keras.Model(inputs=inputs, outputs=h)

def style_network(input_shape=(256, 256, 3),
                  style_shape=(256, 256, 3),
                  weight_decay=0.000002):

    h = inputs = tf.keras.Input(input_shape)
    s = style = tf.keras.Input(style_shape)
    s1 = tf.image.resize(s, [128, 128])
    s2 = tf.image.resize(s, [64, 64])

    h = tf.keras.layers.Conv2D(filters=72,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.AveragePooling2D((3,3), strides=2, padding="same")(h)   # 128

    ####################################################################
    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=132,
                                kernel_size=3,
                                strides=1,
                                padding="valid",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h)
    h = fix_adaIN(h, s1)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
    ####################################################################

    h = tf.keras.layers.AveragePooling2D((3,3), strides=2, padding="same")(h)   # 64

    ####################################################################
    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=264,
                                kernel_size=3,
                                strides=1,
                                padding="valid",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h)
    h = fix_adaIN(h, s2)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
    ####################################################################

    h = tf.keras.layers.AveragePooling2D((3,3), strides=2, padding="same")(h) # 32  

    ####################################################################
    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], "SYMMETRIC")
    h = tf.keras.layers.Conv2D(filters=264,
                                kernel_size=3,
                                strides=1,
                                padding="valid")(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
    ####################################################################

    #h = tf.keras.layers.AveragePooling2D((3,3), strides=2, padding="same")(h) # 16

    #####################################################################
    #h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    #h = tf.keras.layers.Conv2D(filters=540,
    #                            kernel_size=3,
    #                            strides=1,
    #                            padding="valid",
    #                            use_bias=False,
    #                            kernel_regularizer=l2(weight_decay))(h)
    #h = InstanceNormalization()(h)
    #h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
    #####################################################################

    #h = tf.keras.layers.AveragePooling2D((3,3), strides=2, padding="same")(h) # 8

    #h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")
    #h = tf.keras.layers.Conv2D(filters=540,
    #                           kernel_size=3,
    #                           strides=2,
    #                           use_bias=False,
    #                           kernel_regularizer=l2(weight_decay))(h)
    #h = InstanceNormalization()(h)
    #h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    #h = tf.keras.layers.Conv2D(filters=128,
    #                           kernel_size=4,
    #                           use_bias=False,
    #                           kernel_regularizer=l2(weight_decay))(h)
    #h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)


    return tf.keras.Model(inputs=[inputs, style], outputs=h)

#model = style_network()
#model.summary()