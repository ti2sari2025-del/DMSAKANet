"""
Baselines — trained and evaluated with exactly the same patient-level protocol as DMSA-KANet
==========================================================================================
Identical to train_dmsa_kanet_patientlevel.py in everything except the network:
  * same image set and fold file (role_fold{k}: train / val / test, patient-level)
  * same 6x training augmentation, batch size 8, 50 epochs, seed 42
  * same model selection on the VALIDATION set (best val Dice checkpoint, ReduceLROnPlateau)
  * same threshold selection on the validation set, applied unchanged to the test set
  * same test-time augmentation (flip TTA) and the same metric code (pooled + per-image)

Two training recipes (set RECIPE):
  'common'   : DMSA-KANet composite loss (FocalTversky 0.4 + Lovasz 0.3 + Focal 0.2 + Dice 0.1),
               Adam, batch 8, 6x augmentation; learning rate as in each original notebook
               (1e-3 for the CNNs, 1e-4 for TransUNet and ViT)          -> main comparison table
  'original' : the recipe of the original baseline notebooks (binary cross-entropy, RMSprop,
               original LR and batch size, no augmentation), but with the same split, model
               selection, threshold rule and metrics                    -> supplementary table

Architectures are copied unchanged from the original notebooks: U-Net, R2 U-Net and Attention
U-Net from keras-unet-collection 0.1.13 (pip install keras-unet-collection); U-Net++, TransUNet
and ViT as custom Keras code.

Usage: set MODEL_NAME and RECIPE below (or pass them as arguments:
       python train_baseline_patientlevel.py unet common), then run the script. Completed folds are skipped,
so an interrupted run can simply be restarted.
"""

import os
import json
import random
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose,
                                     concatenate, BatchNormalization, LayerNormalization,
                                     MultiHeadAttention, Dense, UpSampling2D)
from tensorflow.keras.optimizers import Adam, RMSprop

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = 'unet'        # 'unet', 'unetpp', 'r2unet', 'attunet', 'transunet', 'vit'
RECIPE     = 'common'      # 'common' or 'original'
import sys
if len(sys.argv) >= 2 and not sys.argv[1].startswith('-'):
    MODEL_NAME = sys.argv[1]
if len(sys.argv) >= 3:
    RECIPE = sys.argv[2]

# Dataset is chosen with an environment variable (default: CDD-CESM):
#   DATASET=cbisddsm python train_baseline_patientlevel.py unet common
DATASET   = os.environ.get('DATASET', 'cddcesm')
DATASETS  = {
    'cddcesm':  ('CDD_DM_650_256/', 'fold_assignments_cddcesm_650.csv'),
    'cbisddsm': ('CBIS_DDSM_256/',  'fold_assignments_cbisddsm.csv'),
}
DATA_DIR  = DATASETS[DATASET][0]
FOLD_CSV  = os.path.join(DATA_DIR, DATASETS[DATASET][1])
OUT_DIR   = os.path.abspath(os.path.join(f'results_patientlevel_{DATASET}', f'baseline_{MODEL_NAME}_{RECIPE}'))
os.makedirs(OUT_DIR, exist_ok=True)

SIZE       = 256
N_SPLITS   = 5
EPOCHS     = 50 if DATASET == 'cddcesm' else 30       # CBIS-DDSM: 30 epochs (same for every model)
AUG_FACTOR = 6 if DATASET == 'cddcesm' else 3           # training-set augmentation multiplier

TVERSKY_ALPHA  = 0.85
TVERSKY_BETA   = 0.15
PRED_THRESHOLD = 0.20

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)


# ============================================================
# BASELINE ARCHITECTURES (copied unchanged from the original baseline notebooks)
# ============================================================
def unet_plus_plus(input_size=(SIZE, SIZE, 3), filters=64):
    inputs = layers.Input(input_size)
    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x
    x0_0 = conv_block(inputs, filters)
    x1_0 = conv_block(layers.MaxPooling2D((2, 2))(x0_0), filters * 2)
    x2_0 = conv_block(layers.MaxPooling2D((2, 2))(x1_0), filters * 4)
    x3_0 = conv_block(layers.MaxPooling2D((2, 2))(x2_0), filters * 8)
    x4_0 = conv_block(layers.MaxPooling2D((2, 2))(x3_0), filters * 16)
    x0_1 = conv_block(layers.concatenate([x0_0, layers.UpSampling2D((2, 2))(x1_0)]), filters)
    x1_1 = conv_block(layers.concatenate([x1_0, layers.UpSampling2D((2, 2))(x2_0)]), filters * 2)
    x2_1 = conv_block(layers.concatenate([x2_0, layers.UpSampling2D((2, 2))(x3_0)]), filters * 4)
    x3_1 = conv_block(layers.concatenate([x3_0, layers.UpSampling2D((2, 2))(x4_0)]), filters * 8)
    x0_2 = conv_block(layers.concatenate([x0_0, x0_1, layers.UpSampling2D((2, 2))(x1_1)]), filters)
    x1_2 = conv_block(layers.concatenate([x1_0, x1_1, layers.UpSampling2D((2, 2))(x2_1)]), filters * 2)
    x2_2 = conv_block(layers.concatenate([x2_0, x2_1, layers.UpSampling2D((2, 2))(x3_1)]), filters * 4)
    x0_3 = conv_block(layers.concatenate([x0_0, x0_1, x0_2, layers.UpSampling2D((2, 2))(x1_2)]), filters)
    x1_3 = conv_block(layers.concatenate([x1_0, x1_1, x1_2, layers.UpSampling2D((2, 2))(x2_2)]), filters * 2)
    x0_4 = conv_block(layers.concatenate([x0_0, x0_1, x0_2, x0_3, layers.UpSampling2D((2, 2))(x1_3)]), filters)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x0_4)
    return models.Model(inputs=[inputs], outputs=[outputs])


# ---- U-Net, R2 U-Net, Attention U-Net: keras-unet-collection 0.1.13 (pip install keras-unet-collection)
def kuc_unet():
    from keras_unet_collection import models as kuc
    return kuc.unet_2d((SIZE, SIZE, 3), filter_num=[64, 128, 256, 512, 1024],
                       n_labels=1, stack_num_down=2, stack_num_up=2,
                       activation='ReLU', output_activation='Sigmoid',
                       batch_norm=True, pool=True, unpool=True,
                       backbone=None, weights=None, freeze_backbone=False, freeze_batch_norm=False,
                       name='unet')


def kuc_r2unet():
    from keras_unet_collection import models as kuc
    return kuc.r2_unet_2d((SIZE, SIZE, 3), filter_num=[64, 128, 256, 512, 1024],
                          n_labels=1, stack_num_down=2, stack_num_up=2, recur_num=2,
                          activation='ReLU', output_activation='Sigmoid',
                          batch_norm=True, pool=True, unpool=True, name='r2_unet')


def kuc_attunet():
    from keras_unet_collection import models as kuc
    return kuc.att_unet_2d((SIZE, SIZE, 3), filter_num=[64, 128, 256, 512, 1024],
                           n_labels=1, stack_num_down=2, stack_num_up=2,
                           activation='ReLU', atten_activation='ReLU', attention='add',
                           output_activation='Sigmoid', batch_norm=True, pool=True, unpool=True,
                           backbone=None, weights=None, freeze_backbone=False, freeze_batch_norm=False,
                           name='att_unet')


# ---- shared transformer pieces (from the original TransUNet / ViT notebooks)
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)

    def call(self, x):
        x = self.projection(x)
        batch_size, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        return tf.reshape(x, [batch_size, h * w, self.embed_dim])

    def get_config(self):
        config = super().get_config()
        config.update({'patch_size': self.patch_size, 'embed_dim': self.embed_dim})
        return config


def transformer_encoder(x, num_heads, mlp_dim, dropout_rate=0.1):
    x1 = LayerNormalization(epsilon=1e-6)(x)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1] // num_heads)(x1, x1)
    attention_output = Dropout(dropout_rate)(attention_output)
    x2 = layers.add([x, attention_output])
    x3 = LayerNormalization(epsilon=1e-6)(x2)
    x3 = Dense(mlp_dim, activation='gelu')(x3)
    x3 = Dropout(dropout_rate)(x3)
    x3 = Dense(x.shape[-1])(x3)
    x3 = Dropout(dropout_rate)(x3)
    return layers.add([x2, x3])


# ---- TransUNet (original notebook "7. TransUnet-RMSPROP-CFD")
def vit_encoder(x, patch_size, embed_dim, num_blocks, num_heads, mlp_dim, dropout_rate=0.1):
    patch_embedding = PatchEmbedding(patch_size, embed_dim)(x)
    positions = tf.range(start=0, limit=tf.shape(patch_embedding)[1], delta=1)
    pos_embedding = layers.Embedding(input_dim=1000, output_dim=embed_dim)(positions)
    x = patch_embedding + pos_embedding
    for _ in range(num_blocks):
        x = transformer_encoder(x, num_heads, mlp_dim, dropout_rate)
    x = LayerNormalization(epsilon=1e-6)(x)
    side = tf.cast(tf.sqrt(tf.cast(tf.shape(x)[1], tf.float32)), tf.int32)
    return tf.reshape(x, [-1, side, side, embed_dim])


def transunet(input_size=(SIZE, SIZE, 3), patch_size=16, embed_dim=768, num_blocks=8,
              num_heads=12, mlp_dim=2048, dropout_rate=0.1, filters=64):
    def enc(x, f):
        x = Conv2D(f, 3, padding='same')(x); x = BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = Conv2D(f, 3, padding='same')(x); x = BatchNormalization()(x); x = layers.Activation('relu')(x)
        return x
    inputs = Input(input_size)
    skip1 = enc(inputs, filters);        x = MaxPooling2D((2, 2))(skip1)
    skip2 = enc(x, filters * 2);         x = MaxPooling2D((2, 2))(skip2)
    skip3 = enc(x, filters * 4);         x = MaxPooling2D((2, 2))(skip3)
    skip4 = enc(x, filters * 8);         x = MaxPooling2D((2, 2))(skip4)
    x = vit_encoder(x, patch_size=patch_size // 16, embed_dim=embed_dim, num_blocks=num_blocks,
                    num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate)
    for f, skip in [(filters * 8, skip4), (filters * 4, skip3), (filters * 2, skip2), (filters, skip1)]:
        x = Conv2DTranspose(f, 2, strides=(2, 2), padding='same')(x)
        x = concatenate([x, skip])
        x = enc(x, f)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    return models.Model(inputs=[inputs], outputs=[outputs], name='TransUNet')


# ---- pure ViT segmentation (original notebook "Vit-Trasnformer")
def vit_segmentation(input_size=(SIZE, SIZE, 3), patch_size=16, embed_dim=768, num_blocks=8,
                     num_heads=12, mlp_dim=2048, dropout_rate=0.1, decoder_filters=256):
    inputs = Input(input_size)
    grid_size = input_size[0] // patch_size
    num_patches = grid_size * grid_size
    x = PatchEmbedding(patch_size, embed_dim)(inputs)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    x = x + layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
    for _ in range(num_blocks):
        x = transformer_encoder(x, num_heads, mlp_dim, dropout_rate)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = layers.Reshape((grid_size, grid_size, embed_dim))(x)
    for div in (1, 2, 4, 8):
        x = Conv2D(decoder_filters // div, 3, padding='same')(x)
        x = BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(decoder_filters // 8, 3, padding='same')(x)
    x = BatchNormalization()(x); x = layers.Activation('relu')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    return models.Model(inputs=[inputs], outputs=[outputs], name='ViT_Segmentation')


# name -> (builder, learning rate of the original notebook, batch size of the original notebook)
MODEL_CFG = {
    'unet':      (kuc_unet,    1e-3, 8),
    'unetpp':    (lambda: unet_plus_plus(input_size=(SIZE, SIZE, 3), filters=64), 1e-3, 8),
    'r2unet':    (kuc_r2unet,  1e-3, 8),
    'attunet':   (kuc_attunet, 1e-3, 8),
    'transunet': (lambda: transunet(input_size=(SIZE, SIZE, 3), patch_size=16, embed_dim=768,
                                    num_blocks=8, num_heads=12, mlp_dim=2048, filters=64), 1e-4, 4),
    'vit':       (lambda: vit_segmentation(input_size=(SIZE, SIZE, 3), patch_size=16, embed_dim=768,
                                           num_blocks=8, num_heads=12, mlp_dim=2048,
                                           dropout_rate=0.1, decoder_filters=256), 1e-4, 4),
}
BUILDER, LR, ORIG_BATCH = MODEL_CFG[MODEL_NAME]
# 'common' uses batch 8 for every model (as DMSA-KANet); if a transformer runs out of GPU memory,
# set COMMON_BATCH = 4 and report it.
COMMON_BATCH = 8
BATCH_SIZE = COMMON_BATCH if RECIPE == 'common' else ORIG_BATCH


# ============================================================
# LOSSES, METRICS, AUGMENTATION, TTA (identical to train_dmsa_kanet_patientlevel.py)
# ============================================================
def tversky_loss(y_true, y_pred, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, gamma=0.75):
    s = 1e-6
    yt = K.flatten(y_true); yp = K.flatten(y_pred)
    tp = K.sum(yt * yp); fn = K.sum(yt * (1 - yp)); fp = K.sum((1 - yt) * yp)
    return K.pow(1.0 - (tp + s) / (tp + alpha*fn + beta*fp + s), gamma)


def focal_loss(y_true, y_pred, alpha=0.85, gamma=2.0):
    eps = K.epsilon(); yp = K.clip(y_pred, eps, 1 - eps)
    bce = -(y_true * K.log(yp) + (1 - y_true) * K.log(1 - yp))
    pt  = y_true * yp + (1 - y_true) * (1 - yp)
    w   = alpha * y_true + (1 - alpha) * (1 - y_true)
    return K.mean(w * K.pow(1 - pt, gamma) * bce)


def dice_loss(y_true, y_pred):
    s = 1e-6; yt = K.flatten(y_true); yp = K.flatten(y_pred)
    return 1.0 - (2 * K.sum(yt * yp) + s) / (K.sum(yt) + K.sum(yp) + s)


def lovasz_loss(y_true, y_pred):
    eps  = K.epsilon()
    yp   = K.clip(y_pred, eps, 1 - eps)
    yt   = K.cast(K.flatten(y_true), tf.float32)
    yp_f = K.flatten(yp)
    errors = 1.0 - yt * yp_f - (1.0 - yt) * (1.0 - yp_f)
    errors_sorted, perm = tf.math.top_k(errors, k=tf.shape(errors)[0])
    gt_sorted = tf.gather(yt, perm)
    gts       = tf.reduce_sum(gt_sorted)
    inter     = gts - tf.cumsum(gt_sorted)
    union     = gts + tf.cumsum(1.0 - gt_sorted)
    iou_max   = 1.0 - inter / (union + eps)
    delta = tf.concat([iou_max[:1], iou_max[1:] - iou_max[:-1]], axis=0)
    return tf.reduce_sum(tf.nn.relu(errors_sorted) * delta)


def combined_loss(y_true, y_pred):
    return (0.4 * tversky_loss(y_true, y_pred) +
            0.3 * lovasz_loss(y_true, y_pred)  +
            0.2 * focal_loss(y_true, y_pred)   +
            0.1 * dice_loss(y_true, y_pred))


def dice_coefficient(y_true, y_pred):
    yt = K.flatten(y_true); yp = K.flatten(y_pred)
    return (2 * K.sum(yt * yp) + K.epsilon()) / (K.sum(yt) + K.sum(yp) + K.epsilon())


def pixel_accuracy(y_true, y_pred, thr=PRED_THRESHOLD):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred > thr, tf.float32)), tf.float32))


def find_optimal_threshold(y_true, y_pred):
    best_iou = 0.0; best_t = PRED_THRESHOLD
    ytf = (y_true > 0.5).astype(np.uint8).flatten()
    for t in np.round(np.arange(0.01, 0.91, 0.01), 2):   # wide grid: 0.01 ... 0.90
        ypf = (y_pred.flatten() > t).astype(np.uint8)
        try:
            tn, fp, fn, tp = confusion_matrix(ytf, ypf, labels=[0, 1]).ravel()
            iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
            if iou > best_iou: best_iou = iou; best_t = t
        except: continue
    print(f"  Threshold selected on validation set: {best_t:.3f}  (val IoU={best_iou:.4f})")
    return float(best_t)


def per_image_metrics(y_true, y_pred, threshold, names):
    '''Dice and IoU computed separately for every image (empty-vs-empty counts as 1).'''
    rows = []
    for i in range(len(y_true)):
        t = y_true[i, ..., 0] > 0.5; p = y_pred[i, ..., 0] > threshold
        tp = np.logical_and(t, p).sum(); fp = np.logical_and(~t, p).sum(); fn = np.logical_and(t, ~p).sum()
        d = 1.0 if (2*tp + fp + fn) == 0 else 2*tp / (2*tp + fp + fn)
        j = 1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn)
        rows.append({'Image_name': names[i], 'dice': d, 'iou': j,
                     'lesion_pct': float(t.mean() * 100), 'pred_pct': float(p.mean() * 100)})
    return pd.DataFrame(rows)


def calculate_metrics(y_true, y_pred, threshold=None):
    if threshold is None: threshold = find_optimal_threshold(y_true, y_pred)
    yb  = (y_pred > threshold).astype(np.uint8)
    ytb = (y_true > 0.5).astype(np.uint8)
    ytf = ytb.flatten(); ypf = yb.flatten(); ypp = y_pred.flatten()
    tn, fp, fn, tp = confusion_matrix(ytf, ypf, labels=[0, 1]).ravel()
    s = 1e-7
    try:    auc = roc_auc_score(ytf, ypp)
    except: auc = 0.0
    return {
        'threshold': float(threshold),
        'dice':      (2*tp+s)/(2*tp+fp+fn+s),
        'iou':       (tp+s)/(tp+fp+fn+s),
        'precision': (tp+s)/(tp+fp+s),
        'recall':    (tp+s)/(tp+fn+s),
        'f1_score':  (2*((tp+s)/(tp+fp+s))*((tp+s)/(tp+fn+s))) /
                     (((tp+s)/(tp+fp+s)) + ((tp+s)/(tp+fn+s))),   # = 2PR/(P+R) = Dice
        'accuracy':  (tp+tn)/(tp+tn+fp+fn+s),
        'fpr':       (fp+s)/(fp+tn+s),
        'fnr':       (fn+s)/(fn+tp+s),
        'tif':       (tp+s)/(tp+fn+s),
        'roc_auc':   auc,
    }


def _rotate_pair(img, msk, angle_deg):
    """Rotate image and mask by the same angle."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    img_r = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    msk_r = cv2.warpAffine(msk, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_r, msk_r


def _cutout(img, msk, n_holes=2, max_size=32):
    """Cutout — zero out random patches of the image (mask unchanged)."""
    img_c = img.copy()
    h, w = img.shape[:2]
    for _ in range(n_holes):
        sz = np.random.randint(8, max_size + 1)
        y  = np.random.randint(0, h - sz)
        x  = np.random.randint(0, w - sz)
        img_c[y:y+sz, x:x+sz, :] = 0
    return img_c, msk


def _gamma(img, gamma):
    """Gamma correction — contrast variation."""
    return np.clip(img ** gamma, 0, 1).astype(np.float32)


def augment_mammography(X, y):
    """
    6x augmentation for mammography:
    1. orig
    2. horizontal flip (breast symmetry)
    3. flip vertical
    4. rotate ±15°
    5. gamma correction (detector contrast)
    6. cutout (spatial regularization)
    """
    n = len(X)
    Xa, ya = [X], [y]

    # 2. Horizontal flip
    Xa.append(X[:, :, ::-1, :])
    ya.append(y[:, :, ::-1, :])

    # 3. Vertical flip
    Xa.append(X[:, ::-1, :, :])
    ya.append(y[:, ::-1, :, :])

    # 4. Rotate ±15° per sample
    Xr = np.zeros_like(X); yr = np.zeros_like(y)
    for i in range(n):
        ang = np.random.uniform(-15, 15)
        img_r, msk_r = _rotate_pair(X[i], y[i, :, :, 0], ang)
        Xr[i] = img_r
        yr[i, :, :, 0] = (msk_r > 0.5).astype(np.float32)
    Xa.append(Xr); ya.append(yr)

    # 5. Gamma correction
    Xg = np.zeros_like(X)
    for i in range(n):
        g = np.random.uniform(0.7, 1.3)
        Xg[i] = _gamma(X[i], g)
    Xa.append(Xg); ya.append(y.copy())

    # 6. Cutout
    Xc = np.zeros_like(X); yc = y.copy()
    for i in range(n):
        img_c, _ = _cutout(X[i], y[i], n_holes=2, max_size=32)
        Xc[i] = img_c
    Xa.append(Xc); ya.append(yc)

    if AUG_FACTOR == 3:
        # 3x: original + horizontal flip + ONE of (flip_v, rotate, gamma, cutout) chosen per image
        pick = np.random.randint(2, 6, size=n)
        Xm = np.stack([Xa[pick[i]][i] for i in range(n)]); ym = np.stack([ya[pick[i]][i] for i in range(n)])
        Xa, ya = [Xa[0], Xa[1], Xm], [ya[0], ya[1], ym]
        desc = "3x: orig + flip_h + one random of flip_v/rotate/gamma/cutout"
    else:
        desc = "6x: orig + flip_h + flip_v + rotate + gamma + cutout"
    X_out = np.concatenate(Xa, axis=0)
    y_out = np.concatenate(ya, axis=0)
    perm  = np.random.permutation(len(X_out))
    print(f"  Augmentation: {n} -> {len(X_out)} samples ({desc})")
    return X_out[perm], y_out[perm]


def tta_predict(model, X):
    preds = []
    for fh, fv in [(False, False), (True, False), (False, True), (True, True)]:
        Xf = X.copy()
        if fh: Xf = Xf[:, :, ::-1, :]
        if fv: Xf = Xf[:, ::-1, :, :]
        p  = model.predict(Xf, verbose=0)
        p0 = p[0] if isinstance(p, list) else p
        if fh: p0 = p0[:, :, ::-1, :]
        if fv: p0 = p0[:, ::-1, :, :]
        preds.append(p0)
    return np.mean(preds, axis=0)


# ============================================================
# DATA (same loader and fold file as DMSA-KANet)
# ============================================================
folds_df = pd.read_csv(FOLD_CSV)
names = folds_df['Image_name'].astype(str).tolist()
imgs, msks = [], []
for n in names:
    im = cv2.imread(os.path.join(DATA_DIR, 'images', f'{n}.png'), cv2.IMREAD_COLOR)
    mk = cv2.imread(os.path.join(DATA_DIR, 'masks',  f'{n}.png'), cv2.IMREAD_GRAYSCALE)
    if im is None or mk is None:
        raise FileNotFoundError(f"Missing image or mask for {n}")
    if im.shape[:2] != (SIZE, SIZE):
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        mk = cv2.resize(mk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    imgs.append(im); msks.append(mk)
image_dataset = np.array(imgs, dtype=np.float32) / 255.0
mask_dataset  = (np.expand_dims(np.array(msks, dtype=np.float32), 3) / 255.0 > 0.5).astype(np.float32)
print(f"{MODEL_NAME} | recipe={RECIPE} | LR={LR} | batch={BATCH_SIZE} | images {image_dataset.shape} | out: {OUT_DIR}")


def build_model():
    model = BUILDER()
    if RECIPE == 'common':
        model.compile(optimizer=Adam(learning_rate=LR, clipnorm=1.0),
                      loss=combined_loss, metrics=[dice_coefficient])
    elif RECIPE == 'original':
        model.compile(optimizer=RMSprop(learning_rate=LR),
                      loss='binary_crossentropy', metrics=[dice_coefficient])
    else:
        raise ValueError(RECIPE)
    return model


# ============================================================
# PATIENT-LEVEL 5-FOLD CROSS-VALIDATION
# ============================================================
results = []
for fold_num in range(1, N_SPLITS + 1):
    k = fold_num - 1
    res_path = os.path.join(OUT_DIR, f'fold_{fold_num}_result.json')
    if os.path.exists(res_path):
        print(f"[SKIP] fold {fold_num} already completed")
        results.append(json.load(open(res_path)))
        continue

    print(f"\n{'='*30} {MODEL_NAME} ({RECIPE}) — FOLD {fold_num}/{N_SPLITS} {'='*30}")
    tf.keras.utils.set_random_seed(SEED + k)
    role = folds_df[f'role_fold{k}'].values
    tr = np.where(role == 'train')[0]; va = np.where(role == 'val')[0]; te = np.where(role == 'test')[0]
    pats = folds_df['Patient_ID'].values
    assert not (set(pats[tr]) & set(pats[te])) and not (set(pats[va]) & set(pats[te])) \
        and not (set(pats[tr]) & set(pats[va])), "patient overlap between roles"

    X_train, y_train = image_dataset[tr], mask_dataset[tr]
    X_val,   y_val   = image_dataset[va], mask_dataset[va]
    X_test,  y_test  = image_dataset[te], mask_dataset[te]
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")
    if RECIPE == 'common':
        X_train, y_train = augment_mammography(X_train, y_train)

    model = build_model()
    if fold_num == 1:
        print(f"Parameters: {model.count_params():,}")
    best_w = os.path.join(OUT_DIR, f'fold_{fold_num}_best.weights.h5')
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coefficient', mode='max',
                                             factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(best_w, monitor='val_dice_coefficient', mode='max',
                                           save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(OUT_DIR, f'fold_{fold_num}_history.csv')),
    ]
    start = datetime.now()
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE,
                     epochs=EPOCHS, callbacks=callbacks, shuffle=True, verbose=2)
    elapsed = datetime.now() - start
    model.load_weights(best_w)
    best_ep = int(np.argmax(hist.history['val_dice_coefficient'])) + 1

    thr = find_optimal_threshold(y_val, tta_predict(model, X_val))
    y_pred = tta_predict(model, X_test)
    m = calculate_metrics(y_test, y_pred, threshold=thr)
    test_names = [names[i] for i in te]
    pim = per_image_metrics(y_test, y_pred, thr, test_names)
    pim.insert(0, 'fold', fold_num)
    pim = pim.merge(folds_df[['Image_name', 'Patient_ID', 'Label']], on='Image_name', how='left')
    pim.to_csv(os.path.join(OUT_DIR, f'per_image_metrics_fold_{fold_num}.csv'), index=False)
    np.savez_compressed(os.path.join(OUT_DIR, f'test_probabilities_fold_{fold_num}.npz'),
                        names=np.array(test_names), prob=(y_pred[..., 0] * 255).round().astype(np.uint8))

    r = {k_: float(v) for k_, v in m.items()}
    r.update({'fold': fold_num, 'best_epoch': best_ep, 'execution_time': str(elapsed),
              'dice_per_image_mean': float(pim['dice'].mean()),
              'iou_per_image_mean': float(pim['iou'].mean()),
              'params': int(model.count_params())})
    json.dump(r, open(res_path, 'w'), indent=2)
    results.append(r)
    print(f"--- fold {fold_num} TEST (thr={thr:.3f} from val, best epoch {best_ep}) ---")
    for key in ['dice', 'iou', 'precision', 'recall', 'f1_score', 'fpr', 'roc_auc',
                'dice_per_image_mean', 'iou_per_image_mean']:
        print(f"  {key:<22} {r[key]:.4f}")
    K.clear_session()

# ============================================================
# SUMMARY
# ============================================================
df = pd.DataFrame(results).sort_values('fold')
cols = ['dice', 'iou', 'precision', 'recall', 'f1_score', 'accuracy', 'roc_auc', 'fpr', 'fnr',
        'dice_per_image_mean', 'iou_per_image_mean', 'threshold', 'best_epoch']
df.to_csv(os.path.join(OUT_DIR, 'fold_results.csv'), index=False)
summ = pd.DataFrame({'mean': df[cols].mean(), 'std': df[cols].std()})
summ.to_csv(os.path.join(OUT_DIR, 'summary.csv'))
print(f"\n{MODEL_NAME} ({RECIPE}) — mean ± SD over {len(df)} folds")
print(summ.round(4).to_string())
