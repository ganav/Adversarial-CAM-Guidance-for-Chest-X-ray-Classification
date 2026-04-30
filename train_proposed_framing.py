# train_proposed_framing.py
from __future__ import print_function

import os
import csv
import numpy as np

import keras
import tensorflow as tf
from keras import backend as K
from keras import layers, models

from model_backbones import build_backbone
from paired_xray_mask_loader import build_pairs_data_proposed, PairedXrayMaskSequence


def build_discriminator_simple(img_size=299, name="disc"):
    """
    Simple discriminator, NO BatchNorm, NO Dropout, so it stays stable.
    Input is a map (mask or CAM), shape (H,W,1), output is real/fake sigmoid.
    """
    inp = keras.Input(shape=(img_size, img_size, 1), name=name + "_in")
    x = inp

    x = layers.Conv2D(32, 4, strides=2, padding="same", name=name + "_c1")(x)
    x = layers.LeakyReLU(alpha=0.2, name=name + "_a1")(x)

    x = layers.Conv2D(64, 4, strides=2, padding="same", name=name + "_c2")(x)
    x = layers.LeakyReLU(alpha=0.2, name=name + "_a2")(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same", name=name + "_c3")(x)
    x = layers.LeakyReLU(alpha=0.2, name=name + "_a3")(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same", name=name + "_c4")(x)
    x = layers.LeakyReLU(alpha=0.2, name=name + "_a4")(x)

    x = layers.GlobalAveragePooling2D(name=name + "_gap")(x)
    out = layers.Dense(1, activation="sigmoid", name=name + "_out")(x)

    return models.Model(inp, out, name=name)


def build_cam_from_true_class(fmap, dense_kernel, y_true, img_size):
    """
    fmap: (B,Hf,Wf,C)
    dense_kernel: (C,num_classes)
    y_true: (B,) int32
    returns cam: (B,img_size,img_size,1) in [0,1]
    """
    b = tf.shape(fmap)[0]
    c = tf.shape(fmap)[-1]

    # W: (C,K) -> Wt: (K,C)
    Wt = tf.transpose(dense_kernel)

    # gather weights for the true class per sample -> (B,C)
    w = tf.gather(Wt, y_true)  # y_true indexes rows in Wt
    w = tf.reshape(w, [b, 1, 1, c])

    cam = tf.reduce_sum(fmap * w, axis=-1, keepdims=True)  # (B,Hf,Wf,1)
    cam = tf.nn.relu(cam)

    # normalize per sample to [0,1]
    cam_min = tf.reduce_min(cam, axis=[1, 2, 3], keepdims=True)
    cam_max = tf.reduce_max(cam, axis=[1, 2, 3], keepdims=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    cam = tf.image.resize_bilinear(cam, [img_size, img_size])
    return cam


def make_train_functions(class_model, backbone_model, disc_model, img_size, num_classes,
                         lr_cls=1e-4, lr_disc=1e-4, lambda_adv=1.0):
    """
    Creates two TF1/Keras backend training functions:
      train_disc_fn: updates D only
      train_cls_fn : updates C only, with L_cls + lambda_adv*L_adv

    Also returns predict_fn for quick val accuracy.
    """

    x_in = class_model.input
    pred = class_model.output
    fmap = backbone_model.output

    # placeholders
    y_true = K.placeholder(shape=(None,), dtype="int32", name="y_true")
    m_real = K.placeholder(shape=(None, img_size, img_size, 1), dtype="float32", name="m_real")

    # last dense layer kernel (C,K)
    last = class_model.layers[-1]
    if not hasattr(last, "kernel"):
        raise ValueError("Last layer of class_model must be Dense with .kernel, got: {}".format(last.name))
    W = last.kernel

    # CAM from true class, (B,img,img,1)
    cam = build_cam_from_true_class(fmap, W, y_true, img_size)

    # Discriminator outputs
    d_real = disc_model(m_real)
    d_fake_for_d = disc_model(tf.stop_gradient(cam))  # stop grad to classifier
    d_fake_for_g = disc_model(cam)                    # grad flows to classifier

    # losses
    # classification
    ce = keras.losses.sparse_categorical_crossentropy(y_true, pred)  # (B,)
    loss_cls = K.mean(ce)

    # discriminator BCE
    ones = K.ones_like(d_real)
    zeros = K.zeros_like(d_fake_for_d)

    bce_real = K.mean(keras.losses.binary_crossentropy(ones, d_real))
    bce_fake = K.mean(keras.losses.binary_crossentropy(zeros, d_fake_for_d))
    loss_disc = bce_real + bce_fake

    # generator adv loss, wants D(cam)=1
    ones2 = K.ones_like(d_fake_for_g)
    loss_adv = K.mean(keras.losses.binary_crossentropy(ones2, d_fake_for_g))

    loss_total = loss_cls + (lambda_adv * loss_adv)

    # optimizers (old Keras uses lr=)
    opt_cls = keras.optimizers.Adam(lr=lr_cls)
    opt_disc = keras.optimizers.Adam(lr=lr_disc)

    # updates
    # classifier updates need BN moving stats, include class_model.updates
    cls_updates = opt_cls.get_updates(params=class_model.trainable_weights, loss=loss_total)
    cls_updates = cls_updates + class_model.updates

    disc_updates = opt_disc.get_updates(params=disc_model.trainable_weights, loss=loss_disc)
    disc_updates = disc_updates + disc_model.updates

    # training functions
    # train_disc uses learning_phase=0 to avoid accidentally updating classifier BN while training D
    train_disc_fn = K.function(
        inputs=[x_in, y_true, m_real, K.learning_phase()],
        outputs=[loss_disc, bce_real, bce_fake],
        updates=disc_updates
    )

    # train_cls uses learning_phase=1 for proper classifier BN training
    train_cls_fn = K.function(
        inputs=[x_in, y_true, K.learning_phase()],
        outputs=[loss_total, loss_cls, loss_adv, pred],
        updates=cls_updates
    )

    # inference prediction (learning_phase=0)
    predict_fn = K.function(
        inputs=[x_in, K.learning_phase()],
        outputs=[pred]
    )

    # also export a cam function for debugging
    cam_fn = K.function(
        inputs=[x_in, y_true, K.learning_phase()],
        outputs=[cam]
    )

    return train_disc_fn, train_cls_fn, predict_fn, cam_fn


def train_proposed(data_root,
                           backbone_name="densenet121",
                           img_size=299,
                           chan=1,
                           batch_size=4,
                           epochs=5,
                           lambda_adv=1.0,
                           warmup_epochs=1,
                           lr_cls=1e-4,
                           lr_disc=1e-4,
                           out_dir="artifacts_proposed"):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # build pairs
    train_pairs, class_to_idx = build_pairs_data_proposed(data_root, "train")
    val_pairs, _ = build_pairs_data_proposed(data_root, "validation")

    num_classes = len(class_to_idx)

    train_seq = PairedXrayMaskSequence(train_pairs, img_size=img_size, batch_size=batch_size,
                                       x_channels=chan, shuffle=True)
    val_seq = PairedXrayMaskSequence(val_pairs, img_size=img_size, batch_size=batch_size,
                                     x_channels=chan, shuffle=False)

    # build models
    class_model, backbone_model = build_backbone(backbone_name, num_classes,
                                                 input_shape=(img_size, img_size, chan))
    disc_model = build_discriminator_simple(img_size=img_size, name="disc")

    # training functions
    train_disc_fn, train_cls_fn, predict_fn, cam_fn = make_train_functions(
        class_model, backbone_model, disc_model,
        img_size=img_size, num_classes=num_classes,
        lr_cls=lr_cls, lr_disc=lr_disc, lambda_adv=lambda_adv
    )

    # logs
    log_path = os.path.join(out_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "loss_total", "loss_cls", "loss_adv", "loss_disc", "acc"])

    steps_per_epoch = len(train_seq)

    for ep in range(1, epochs + 1):
        # warmup: disable adv for first warmup_epochs by skipping disc step and using lambda 0 effect
        use_adv = (ep > warmup_epochs)

        ep_losses = []
        ep_accs = []

        for step in range(steps_per_epoch):
            x_b, y_b, m_b = train_seq[step]

            # 1) train discriminator
            if use_adv:
                ld, ldr, ldf = train_disc_fn([x_b, y_b, m_b, 0])
                loss_disc_val = float(ld)
            else:
                loss_disc_val = 0.0

            # 2) train classifier (always)
            lt, lc, la, p = train_cls_fn([x_b, y_b, 1])

            # accuracy
            y_pred = np.argmax(p, axis=1).astype(np.int64)
            acc = float(np.mean(y_pred == y_b))

            ep_losses.append([float(lt), float(lc), float(la), float(loss_disc_val)])
            ep_accs.append(acc)
    

        # quick val acc (no adv, just classification)
        val_accs = []
        for vi in range(len(val_seq)):
            xv, yv, mv = val_seq[vi]
            pv = predict_fn([xv, 0])[0]
            ypv = np.argmax(pv, axis=1).astype(np.int64)
            val_accs.append(float(np.mean(ypv == yv)))
        val_acc = float(np.mean(val_accs))

        ep_losses_np = np.array(ep_losses, dtype="float32")  # [N,4] total, cls, adv, disc

        train_total = float(ep_losses_np[:, 0].mean())
        train_cls   = float(ep_losses_np[:, 1].mean())
        train_adv   = float(ep_losses_np[:, 2].mean())
        train_disc  = float(ep_losses_np[:, 3].mean())
        train_acc   = float(np.mean(ep_accs))

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, train_total, train_cls, train_adv, train_disc, train_acc, val_acc])

        print("Epoch {} done , train acc {:.3f} , val acc {:.3f}".format(ep, float(np.mean(ep_accs)), val_acc))

        # save weights only
        class_w = os.path.join(out_dir, "{}_classifier_weights.h5".format(backbone_name))
        disc_w = os.path.join(out_dir, "{}_disc_weights.h5".format(backbone_name))
        class_model.save_weights(class_w)
        disc_model.save_weights(disc_w)

        cls_full = os.path.join(out_dir, "{}_classifier_full.h5".format(backbone_name))
        disc_full = os.path.join(out_dir, "{}_disc_full.h5".format(backbone_name))
        class_model.save(cls_full, include_optimizer=False)
        disc_model.save(disc_full, include_optimizer=False)


    print("Done, saved to:", out_dir)