# test_all.py (TF1.9 + Keras 2.1.6, weights-only)
from __future__ import print_function

import os
import csv
import numpy as np
import cv2

import keras
import tensorflow as tf
from keras import backend as K

from model_backbones import build_backbone

# =========================
# EDIT THESE
# =========================
DATA_ROOT    = r"G:\projects\paper 25\source\data_proposed"
BACKBONE     = "densenet121"
IMG_SIZE     = 299
CHAN         = 3  # recommended
WEIGHTS_PATH = r"G:\projects\paper 25\source\artifacts_proposed\densenet121_classifier_weights.h5"

OUT_DIR      = r"G:\projects\paper 25\source\test_out"
SAVE_CAM     = True
MAX_CAM_PER_CLASS = 10   # save only a few overlays per class (avoid huge disk)
# =========================


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def get_class_names(data_root):
    train_dir = os.path.join(data_root, "train")
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes = sorted(classes)
    return classes


def iter_test_images(data_root, class_names):
    """
    Yields (img_path, true_class_idx)
    """
    test_dir = os.path.join(data_root, "test")
    for ci, cname in enumerate(class_names):
        img_dir = os.path.join(test_dir, cname, "images")
        if not os.path.isdir(img_dir):
            continue
        for fn in os.listdir(img_dir):
            p = os.path.join(img_dir, fn)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMG_EXTS:
                yield p, ci


def load_xray(path, img_size, chan):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read image: {}".format(path))
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (H,W,1)
    if chan == 3:
        img = np.concatenate([img, img, img], axis=-1)  # (H,W,3)
    img = np.expand_dims(img, axis=0)  # (1,H,W,C)
    return img


def build_cam_fn(class_model, backbone_model, img_size):
    x_in = class_model.input
    pred = class_model.output
    fmap = backbone_model.output

    last = class_model.layers[-1]
    if not hasattr(last, "kernel"):
        raise ValueError("Last layer must be Dense with kernel. Got: {}".format(last.name))
    W = last.kernel  # (C,K)

    y_idx = K.placeholder(shape=(None,), dtype="int32", name="y_idx")

    def _cam(xs):
        fmap_, y_ = xs
        b = tf.shape(fmap_)[0]
        c = tf.shape(fmap_)[-1]

        Wt = tf.transpose(W)          # (K,C)
        y_ = tf.cast(y_, tf.int32)    # force int for tf.gather
        w = tf.gather(Wt, y_)         # (B,C)
        w = tf.reshape(w, [b, 1, 1, c])

        cam = tf.reduce_sum(fmap_ * w, axis=-1, keepdims=True)
        cam = tf.nn.relu(cam)

        cam_min = tf.reduce_min(cam, axis=[1, 2, 3], keepdims=True)
        cam_max = tf.reduce_max(cam, axis=[1, 2, 3], keepdims=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam = tf.image.resize_bilinear(cam, [img_size, img_size])
        return cam

    cam = keras.layers.Lambda(_cam, name="cam_map")([fmap, y_idx])
    return K.function([x_in, y_idx, K.learning_phase()], [cam, pred])


def save_cam_overlay(out_dir, img_path, cam01, tag):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    base = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    base = cv2.resize(base, (cam01.shape[1], cam01.shape[0]), interpolation=cv2.INTER_AREA)

    cam_u8 = np.clip(cam01 * 255.0, 0, 255).astype("uint8")
    cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.6, cam_color, 0.4, 0)

    out_path = os.path.join(out_dir, "{}.png".format(tag))
    cv2.imwrite(out_path, overlay)


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    class_names = get_class_names(DATA_ROOT)
    num_classes = len(class_names)

    # Build and load weights
    class_model, backbone_model = build_backbone(
        BACKBONE,
        num_classes=num_classes,
        input_shape=(IMG_SIZE, IMG_SIZE, CHAN)
    )
    class_model.load_weights(WEIGHTS_PATH)

    cam_fn = build_cam_fn(class_model, backbone_model, IMG_SIZE) if SAVE_CAM else None

    # For confusion matrix
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    # CSV
    csv_path = os.path.join(OUT_DIR, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_path", "true_label", "pred_label", "pred_prob"])

        cam_counts = [0] * num_classes

        all_imgs = list(iter_test_images(DATA_ROOT, class_names))
        print("Total test images:", len(all_imgs))

        for i, (img_path, y_true) in enumerate(all_imgs):
            x = load_xray(img_path, IMG_SIZE, CHAN)
            p = class_model.predict(x, batch_size=1)[0]
            y_pred = int(np.argmax(p))
            prob = float(p[y_pred])

            conf[y_true, y_pred] += 1

            w.writerow([img_path, class_names[y_true], class_names[y_pred], prob])

            # save limited CAM overlays
            if SAVE_CAM and cam_counts[y_true] < MAX_CAM_PER_CLASS:
                cam, _ = cam_fn([x, np.array([y_pred], dtype="int32"), 0])  # use predicted class for CAM
                cam01 = cam[0, :, :, 0]
                tag = "true_{}_pred_{}_{}".format(class_names[y_true], class_names[y_pred], i)
                save_cam_overlay(os.path.join(OUT_DIR, "cam_overlays"), img_path, cam01, tag)
                cam_counts[y_true] += 1

            if (i + 1) % 50 == 0:
                print("Processed:", i + 1)

    # metrics
    acc = float(np.trace(conf) / (np.sum(conf) + 1e-8))

    # save confusion matrix
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.csv")
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, cname in enumerate(class_names):
            w.writerow([cname] + conf[i].tolist())

    print("Saved:", csv_path)
    print("Saved:", cm_path)
    print("Accuracy:", acc)
    print("Confusion matrix:\n", conf)


if __name__ == "__main__":
    main()