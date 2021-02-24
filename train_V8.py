# -*- coding:utf-8 -*-
from model_V8 import *
from random import random

import matplotlib.pyplot as plt
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256,
                           
                           "load_size": 266,
                           
                           "batch_size": 1,
                           
                           "epochs": 200,
                           
                           "lr": 0.0002,

                           "n_classes": 24,
                           
                           "in_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/male_16_39_train.txt",
                           
                           "in_img_path": "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/",
                           
                           "st_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/female_40_63_train.txt",
                           
                           "st_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/female_40_63/",
                           
                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": "",
                           
                           "save_images": "C:/Users/Yuhwan/Pictures/test_sample",
                           
                           "graphs": ""})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.5)
s_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.5)

def input_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.load_size, FLAGS.load_size])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3]) / 127.5 - 1.

    if random() < 0.5:
        img = tf.image.flip_left_right(img)

    lab = lab_list - 16 + 1
    nor = lab / FLAGS.n_classes

    #noise = tf.random.uniform([256, 256, 3], dtype=tf.float32)
    noise = tf.random.truncated_normal([256, 256, 3], dtype=tf.float32)
    return img, lab, nor, noise

def style_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.load_size, FLAGS.load_size])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3]) / 127.5 - 1.

    if random() < 0.5:
        img = tf.image.flip_left_right(img)

    lab = lab_list - 40 + 1
    nor = lab / FLAGS.n_classes
    

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(images, style_images, noise, 
             g_model_A2B, g_model_B2A, 
             s_model_real, s_model_fake, 
             d_model):
    # 라벨넣는것으로 조금만 추가해보자! --> 원래는 이것까지 하는게 내 목표였다 (오늘 논문 할당양 쓰고 수정하자)
    with tf.GradientTape() as g_tape, tf.GradientTape() as s_tape, tf.GradientTape() as d_tape:
        fake_img = run_model(g_model_A2B, [images, style_images], True)
        cycle_img = run_model(g_model_B2A, [fake_img, images], True)
        fake_style_logits = run_model(s_model_fake, [noise, fake_img], True)
        real_style_logits = run_model(s_model_real, [noise, style_images], True)

        real_dis = run_model(d_model, images, True)
        fake_dis = run_model(d_model, fake_img, True)

        g_style_recon = tf.reduce_mean(tf.abs(fake_img - style_images))
        g_style_diver = tf.reduce_mean(tf.abs(fake_style_logits - real_style_logits))
        g_cycle = tf.reduce_mean(tf.abs(images - cycle_img))
        g_ad = tf.reduce_mean((fake_dis - tf.ones_like(fake_dis))**2)
        g_loss = g_style_recon + g_style_diver + g_cycle + g_ad

        d_loss = (tf.reduce_mean((real_dis - tf.ones_like(real_dis))**2) + tf.reduce_mean((fake_dis - tf.zeros_like(fake_dis))**2)) / 2

    g_grads = g_tape.gradient(g_loss, g_model_A2B.trainable_variables + g_model_B2A.trainable_variables)
    s_grads = s_tape.gradient(g_loss, s_model_real.trainable_variables + s_model_fake.trainable_variables)
    d_grads = d_tape.gradient(d_loss, d_model.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, g_model_A2B.trainable_variables + g_model_B2A.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, d_model.trainable_variables))
    s_optim.apply_gradients(zip(s_grads, s_model_real.trainable_variables + s_model_fake.trainable_variables))

    return g_loss, d_loss

def main():
    g_model_A2B = V8_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                               style_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    g_model_B2A = V8_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                               style_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    s_model_real = style_network(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                                 style_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    s_model_fake = style_network(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                                 style_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    d_model = discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    g_model_A2B.summary()
    s_model_fake.summary()
    d_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(g_model=g_model, s_model=s_model, d_model=d_model,
                                   g_optim=g_optim, s_optim=s_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)


    if FLAGS.train:
        count = 0

        input_img = np.loadtxt(FLAGS.in_txt_path, dtype="<U100", skiprows=0, usecols=0)
        input_img = [FLAGS.in_img_path + img for img in input_img]
        input_lab = np.loadtxt(FLAGS.in_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        style_img = np.loadtxt(FLAGS.st_txt_path, dtype="<U100", skiprows=0, usecols=0)
        style_img = [FLAGS.st_img_path + img for img in style_img]
        style_lab = np.loadtxt(FLAGS.st_txt_path, dtype=np.int32, skiprows=0, usecols=1)


        for epoch in range(FLAGS.epochs):

            input_gener = tf.data.Dataset.from_tensor_slices((input_img, input_lab))
            input_gener = input_gener.shuffle(len(input_img))
            input_gener = input_gener.map(input_func)
            input_gener = input_gener.batch(FLAGS.batch_size)
            input_gener = input_gener.prefetch(tf.data.experimental.AUTOTUNE)

            style_gener = tf.data.Dataset.from_tensor_slices((style_img, style_lab))
            style_gener = style_gener.shuffle(len(style_img))
            style_gener = style_gener.map(style_func)
            style_gener = style_gener.batch(FLAGS.batch_size)
            style_gener = style_gener.prefetch(tf.data.experimental.AUTOTUNE)

            input_iter = iter(input_gener)
            style_iter = iter(style_gener)

            train_idx = min(len(input_gener), len(style_img)) // FLAGS.batch_size
            for step in range(train_idx):

                images, labels, _, noise = next(input_iter)

                style_images, style_labels = next(style_iter)

                g_loss, d_loss = cal_loss(images, style_images, noise, 
                                          g_model_A2B, g_model_B2A, 
                                          s_model_real, s_model_fake, 
                                          d_model)

                print(g_loss, d_loss, count)

                if count % 100 == 0:
                    fake_img = run_model(g_model_A2B, [images, style_images], False)

                    plt.imsave(FLAGS.save_images + "/fake_1_{}.jpg".format(count), fake_img[0].numpy() * 0.5 + 0.5)

                    plt.imsave(FLAGS.save_images + "/input_1_{}.jpg".format(count), images[0].numpy() * 0.5 + 0.5)

                    plt.imsave(FLAGS.save_images + "/style_1_{}.jpg".format(count), style_images[0].numpy() * 0.5 + 0.5)

                count += 1

if __name__ == "__main__":
    main()