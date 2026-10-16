"""
DMSA-KANet — training & evaluation with a patient-level protocol
================================================================
Architecture, loss, augmentation and two-phase schedule are unchanged from the
version used in the original submission. Changes in this version (revision 2):

  1. Data split read from a patient-level fold file (fold_assignments_*.csv):
     for each outer fold k, images have role train / val / test (column role_fold{k}).
     No patient appears in more than one role within a fold.
  2. Model selection uses the inner VALIDATION set only (best-val-Dice checkpoint,
     ReduceLROnPlateau). The TEST fold is used once, for the final evaluation.
  3. The binarization threshold is selected on the validation set and then
     applied unchanged to the test set (previously it was tuned on the test fold).
  4. F1 formula corrected: F1 = 2PR / (P + R)  (identical to pixel-level Dice).
  5. LR phase 2 = 1e-5, the validated optimum of the hyperparameter search (Trial 8).
  6. Epoch-level resume after machine shutdowns (weights + training state saved every epoch).
  7. Fixed random seeds; per-image metrics and test-set probability maps are saved
     so that every reported number can be recomputed.
  8. (revision 4) Optional learning-rate override for nested selection inside the
     training patients:  LR1=1e-4 LR2=5e-5 RUN_TAG=lr1_1e-4_lr2_5e-5 DATASET=cbisddsm ...
     Without these variables the script reproduces the runs of Tables 6 and 7.
     Each fold result also stores the configuration, the best validation Dice and its epoch.

Metrics: pixel-level confusion counts pooled over all test images of a fold
("pooled"), plus the mean of per-image Dice/IoU ("per-image"). Reported values are
mean ± SD over the five outer folds.
"""

import os
import json
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
# Dataset is chosen with an environment variable (default: CDD-CESM):
#   DATASET=cbisddsm python train_dmsa_kanet_patientlevel.py
DATASET   = os.environ.get('DATASET', 'cddcesm')
DATASETS  = {
    'cddcesm':  ('CDD_DM_650_256/', 'fold_assignments_cddcesm_650.csv'),
    'cbisddsm': ('CBIS_DDSM_256/',  'fold_assignments_cbisddsm.csv'),
}
DATA_DIR  = DATASETS[DATASET][0]                                    # images/ and masks/ inside
FOLD_CSV  = os.path.join(DATA_DIR, DATASETS[DATASET][1])
RUN_TAG   = os.environ.get('RUN_TAG', '')                           # e.g. nested/lr1_1e-4_lr2_5e-5
OUT_DIR   = os.path.abspath(os.path.join(f'results_patientlevel_{DATASET}', RUN_TAG))  # metrics, predictions
os.makedirs(OUT_DIR, exist_ok=True)

SIZE           = 256
BATCH_SIZE     = 8
N_SPLITS       = 5
EPOCHS         = 50 if DATASET == 'cddcesm' else 30   # CBIS-DDSM: 30 epochs (see README)
AUG_FACTOR     = 6 if DATASET == 'cddcesm' else 3    # training-set augmentation multiplier
LR_PHASE1      = float(os.environ.get('LR1', '1e-3'))
LR_PHASE2      = float(os.environ.get('LR2', '1e-5'))    # default: Trial 8
PHASE1_EPOCHS  = 20

KAN_GRID       = 5
KAN_ORDER      = 3
DILATION_RATES = [1, 3, 6, 12]
DROPOUT_RATE   = 0.2

TVERSKY_ALPHA  = 0.85
TVERSKY_BETA   = 0.15
DS_WEIGHTS     = [0.5, 0.2, 0.2, 0.1]

PRED_THRESHOLD = 0.20
L2_REG         = 2e-4

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)


# ============================================================
# CHECKPOINT SYSTEM (PER FOLD; allows resuming after an interruption)
# ============================================================
CHECKPOINT_DIR = os.path.join(OUT_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"[INIT] Checkpoints saved to: {CHECKPOINT_DIR}")

CKPT_PREFIX = os.path.join(CHECKPOINT_DIR, 'dmsa_kanet')


def get_fold_result_path(fold_num):
    return os.path.join(CHECKPOINT_DIR, f'fold_{fold_num}_result.json')


def get_fold_history_path(fold_num):
    return os.path.join(CHECKPOINT_DIR, f'fold_{fold_num}_history.json')


def is_fold_completed(fold_num):
    return os.path.exists(get_fold_result_path(fold_num))


def save_fold_result(fold_num, result, history):
    clean_result = {}
    for k, v in result.items():
        if isinstance(v, (np.floating, np.integer)):
            clean_result[k] = float(v)
        elif isinstance(v, (int, float, str, bool)):
            clean_result[k] = v
        else:
            clean_result[k] = str(v)
    with open(get_fold_result_path(fold_num), 'w') as f:
        json.dump(clean_result, f, indent=2)
    clean_hist = {k: [float(x) for x in v] for k, v in history.items()}
    with open(get_fold_history_path(fold_num), 'w') as f:
        json.dump(clean_hist, f, indent=2)
    print(f"  [✓] Fold {fold_num} saved to {get_fold_result_path(fold_num)}")


def load_completed_folds():
    results, histories = [], []
    for fn in range(1, N_SPLITS + 1):
        if is_fold_completed(fn):
            with open(get_fold_result_path(fn)) as f:
                results.append(json.load(f))
            with open(get_fold_history_path(fn)) as f:
                histories.append(json.load(f))
    return results, histories


# ============================================================
# MODEL COMPLEXITY UTILS
# ============================================================
def count_params(model):
    total_p     = model.count_params()
    trainable_p = sum([K.count_params(w) for w in model.trainable_weights])
    return total_p, trainable_p, total_p - trainable_p

def fmt_num(n):
    if n >= 1e9: return f"{n/1e9:.2f}G"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(n)

def estimate_flops(model):
    total = 0
    for layer in model.layers:
        try:
            cfg = layer.get_config()
            os_ = layer.output_shape
            if isinstance(os_, list): os_ = os_[0]
            if isinstance(layer, tf.keras.layers.Conv2D) and len(os_)==4:
                _,oh,ow,oc = os_; kh=cfg.get('kernel_size',(3,3))
                kh = kh[0] if isinstance(kh,(list,tuple)) else kh
                ins = layer.input_shape; ic = ins[-1] if not isinstance(ins,list) else ins[0][-1]
                total += oh*ow*oc*ic*kh*kh
            elif isinstance(layer, tf.keras.layers.Dense):
                ins = layer.input_shape; ic = ins[-1] if not isinstance(ins,list) else ins[0][-1]
                total += ic * cfg.get('units', 0)
        except: continue
    return total

def print_complexity_table(model, input_shape=(256,256,3), model_name='DMSA-KANet'):
    tp, trp, ntp = count_params(model)
    macs = estimate_flops(model)
    sz   = f"{input_shape[0]}×{input_shape[1]}×{input_shape[2]}"
    print(f"\n{'='*90}")
    print(f"  Table 6. Model Parameters — {model_name}")
    print(f"{'='*90}")
    print(f"  {'Method':<28} {'FLOPs(MACs)':>12} {'Total':>10} {'Trainable':>12} {'Non-train':>12} {'Input':>10}")
    print(f"  {'-'*88}")
    print(f"  {model_name:<28} {fmt_num(macs):>12} {fmt_num(tp):>10} {fmt_num(trp):>12} {fmt_num(ntp):>12} {sz:>10}")
    print(f"{'='*90}")
    print(f"    FLOPs: {macs:,}  Total: {tp:,}  Trainable: {trp:,}  Non-train: {ntp:,}\n")
    pd.DataFrame([{'model': model_name, 'flops': macs, 'flops_fmt': fmt_num(macs),
                   'total': tp, 'total_fmt': fmt_num(tp), 'trainable': trp,
                   'nontrainable': ntp, 'size': sz}]
                 ).to_csv(os.path.join(OUT_DIR, 'dmsa_kanet_complexity.csv'), index=False)


# ============================================================
# CUSTOM LAYERS (unchanged)
# ============================================================
@register_keras_serializable(package='AMEDDE')
class KANSplineActivation(layers.Layer):
    def __init__(self, f, gs=5, so=3, **kw):
        super().__init__(**kw)
        self.f = f; self.gs = gs; self.so = so

    def build(self, s):
        n = int(s[-1]); k = self.gs + 1; e = k + 2 * self.so
        self.g  = self.add_weight(name='g',  shape=(n, e),          initializer='zeros',         trainable=False)
        self.sw = self.add_weight(name='sw', shape=(n, self.f, self.gs + self.so), initializer='glorot_uniform')
        self.bw = self.add_weight(name='bw', shape=(n, self.f),     initializer='glorot_uniform')
        self.sc = self.add_weight(name='sc', shape=(n, self.f),     initializer='ones')
        gv = np.linspace(-1, 1, k); h = gv[1] - gv[0]
        eg = np.concatenate([gv[0] - h*np.arange(self.so, 0, -1), gv, gv[-1] + h*np.arange(1, self.so+1)])
        self.g.assign(np.tile(eg[None], (n, 1)).astype(np.float32))
        super().build(s)

    def bs(self, x):
        xe = tf.expand_dims(x, -1); g = self.g
        b  = tf.cast((xe >= g[:, :-1]) & (xe < g[:, 1:]), tf.float32)
        for k in range(1, self.so + 1):
            ld = g[:, k:-1] - g[:, :-(k+1)]; rd = g[:, (k+1):] - g[:, 1:-k]
            ld = tf.where(tf.abs(ld) < 1e-8, tf.ones_like(ld), ld)
            rd = tf.where(tf.abs(rd) < 1e-8, tf.ones_like(rd), rd)
            b  = (xe - g[:, :-(k+1)]) / ld * b[:, :, :-1] + (g[:, (k+1):] - xe) / rd * b[:, :, 1:]
        return b

    def call(self, x, training=None):
        sh = tf.shape(x)[:-1]; xf = tf.reshape(x, (-1, tf.shape(x)[-1]))
        out = tf.matmul(tf.nn.silu(xf), self.bw) + \
              tf.einsum('iok,nik->no', self.sw * tf.expand_dims(self.sc, -1), self.bs(xf))
        return tf.reshape(out, tf.concat([sh, [self.f]], 0))

    def compute_output_shape(self, s): return tuple(s[:-1]) + (self.f,)
    def get_config(self):
        c = super().get_config(); c.update({'f': self.f, 'gs': self.gs, 'so': self.so}); return c


@register_keras_serializable(package='AMEDDE')
class KANLayer(layers.Layer):
    def __init__(self, u, gs=5, so=3, dr=0.1, **kw):
        super().__init__(**kw)
        self.u = u; self.gs = gs; self.so = so; self.dr = dr

    def build(self, s):
        self.kan = KANSplineActivation(self.u, self.gs, self.so)
        self.nm  = layers.LayerNormalization()
        self.dp  = layers.Dropout(self.dr)
        self.pj  = layers.Dense(self.u, use_bias=False) if int(s[-1]) != self.u else None
        super().build(s)

    def call(self, x, training=None):
        r = x if self.pj is None else self.pj(x)
        return self.nm(self.dp(self.kan(x, training=training), training=training) + r)

    def compute_output_shape(self, s): return tuple(s[:-1]) + (self.u,)
    def get_config(self):
        c = super().get_config(); c.update({'u': self.u, 'gs': self.gs, 'so': self.so, 'dr': self.dr}); return c


@register_keras_serializable(package='AMEDDE')
class ChannelAttentionKAN(layers.Layer):
    def __init__(self, f, gs=5, so=3, dv=0.05, **kw):
        super().__init__(**kw)
        self.f = f; self.gs = gs; self.so = so; self.dv = dv

    def build(self, s):
        self.gap = layers.GlobalAveragePooling2D()
        self.kan = KANLayer(self.f, self.gs, self.so, self.dv)
        super().build(s)

    def call(self, x, training=None):
        a = self.kan(self.gap(x), training=training)
        return x * tf.reshape(tf.sigmoid(a), (-1, 1, 1, self.f))

    def compute_output_shape(self, s): return s
    def get_config(self):
        c = super().get_config(); c.update({'f': self.f, 'gs': self.gs, 'so': self.so, 'dv': self.dv}); return c


@register_keras_serializable(package='AMEDDE')
class CBAMBlock(layers.Layer):
    def __init__(self, f, r=8, **kw):
        super().__init__(**kw)
        self.f = f; self.r = r

    def build(self, s):
        rd = max(self.f // self.r, 8)
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        self.f1  = layers.Dense(rd, activation='relu', use_bias=False)
        self.f2  = layers.Dense(self.f, use_bias=False)
        self.sc  = layers.Conv2D(1, 7, padding='same', activation='sigmoid', use_bias=False)
        super().build(s)

    def call(self, x, training=None):
        ca = tf.sigmoid(self.f2(self.f1(self.gap(x))) + self.f2(self.f1(self.gmp(x))))
        xc = x * tf.reshape(ca, (-1, 1, 1, self.f))
        sp = self.sc(tf.concat([tf.reduce_mean(xc, -1, keepdims=True),
                                 tf.reduce_max(xc, -1, keepdims=True)], -1))
        return xc * sp

    def compute_output_shape(self, s): return s
    def get_config(self):
        c = super().get_config(); c.update({'f': self.f, 'r': self.r}); return c


@register_keras_serializable(package='AMEDDE')
class KANGateAttention(layers.Layer):
    def __init__(self, f, **kw):
        super().__init__(**kw)
        self.f = f

    def build(self, s):
        self.ce = layers.Conv2D(self.f, 1, use_bias=False)
        self.be = layers.BatchNormalization()
        self.cd = layers.Conv2D(self.f, 1, use_bias=False)
        self.bd = layers.BatchNormalization()
        self.rl = layers.ReLU()
        self.gap = layers.GlobalAveragePooling2D()
        self.kan = KANLayer(self.f, KAN_GRID, KAN_ORDER, dr=0.0)
        self.co  = layers.Conv2D(self.f, 1, padding='same', use_bias=False)
        self.bo  = layers.BatchNormalization()
        self.ro  = layers.ReLU()
        super().build(s)

    def call(self, inputs, training=None):
        e, d = inputs
        x  = self.be(self.ce(e), training=training)
        g  = self.bd(self.cd(d), training=training)
        xs = tf.shape(x)
        g  = tf.image.resize(g, (xs[1], xs[2]), method='bilinear')
        a  = self.kan(self.gap(self.rl(x + g)), training=training)
        a  = tf.reshape(tf.sigmoid(a), (-1, 1, 1, self.f))
        return self.ro(self.bo(self.co(e * a), training=training))

    def compute_output_shape(self, s): return s[0]
    def get_config(self):
        c = super().get_config(); c.update({'f': self.f}); return c


@register_keras_serializable(package='AMEDDE')
class ViTBottleneck(layers.Layer):
    def __init__(self, dim=1024, num_heads=8, num_blocks=4,
                 mlp_ratio=4.0, dropout=0.1, **kw):
        super().__init__(**kw)
        self.dim        = dim
        self.num_heads  = num_heads
        self.num_blocks = num_blocks
        self.mlp_ratio  = mlp_ratio
        self.dropout_r  = dropout

    def build(self, input_shape):
        C    = int(input_shape[-1])
        H    = int(input_shape[1]) if input_shape[1] is not None else 8
        W    = int(input_shape[2]) if input_shape[2] is not None else 8
        nseq = H * W + 1

        self.patch_proj = layers.Dense(self.dim, use_bias=False)
        self.cls_token = self.add_weight(
            name='cls_token', shape=(1, 1, self.dim),
            initializer='zeros', trainable=True)
        self.pos_emb = self.add_weight(
            name='pos_emb', shape=(1, nseq, self.dim),
            initializer='random_normal', trainable=True)

        mlp_dim = int(self.dim * self.mlp_ratio)
        self.norms1 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(self.num_blocks)]
        self.norms2 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(self.num_blocks)]
        self.attns  = [layers.MultiHeadAttention(num_heads=self.num_heads,
                          key_dim=self.dim // self.num_heads,
                          dropout=self.dropout_r) for _ in range(self.num_blocks)]
        self.ffn1   = [layers.Dense(mlp_dim, activation='gelu') for _ in range(self.num_blocks)]
        self.ffn2   = [layers.Dense(self.dim) for _ in range(self.num_blocks)]
        self.drops  = [layers.Dropout(self.dropout_r) for _ in range(self.num_blocks)]
        self.norm_out = layers.LayerNormalization(epsilon=1e-6)
        self.kan_out  = KANLayer(self.dim, KAN_GRID, KAN_ORDER, dr=0.05)
        super().build(input_shape)

    def call(self, x, training=None):
        B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]
        xf = tf.reshape(x, (B, H * W, tf.shape(x)[-1]))
        xf = self.patch_proj(xf)

        cls = tf.tile(self.cls_token, (B, 1, 1))
        xf  = tf.concat([cls, xf], axis=1)
        xf  = xf + self.pos_emb

        for i in range(self.num_blocks):
            xn = self.norms1[i](xf)
            xf = xf + self.drops[i](self.attns[i](xn, xn, training=training), training=training)
            xn = self.norms2[i](xf)
            xf = xf + self.drops[i](self.ffn2[i](self.drops[i](
                self.ffn1[i](xn), training=training)), training=training)

        xf = self.norm_out(xf)
        xf = xf[:, 1:, :]
        xf = self.kan_out(xf, training=training)
        return tf.reshape(xf, (B, H, W, self.dim))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.dim,)

    def get_config(self):
        c = super().get_config()
        c.update({'dim': self.dim, 'num_heads': self.num_heads,
                  'num_blocks': self.num_blocks, 'mlp_ratio': self.mlp_ratio,
                  'dropout': self.dropout_r})
        return c


@register_keras_serializable(package='AMEDDE')
class ResizeToMatch(layers.Layer):
    def __init__(self, **kw): super().__init__(**kw)
    def call(self, inputs, training=None):
        src, ref = inputs
        h = tf.shape(ref)[1]; w = tf.shape(ref)[2]
        return tf.image.resize(src, (h, w), method='bilinear')
    def compute_output_shape(self, s):
        return (s[0][0], None, None, s[0][-1])
    def get_config(self): return super().get_config()


@register_keras_serializable(package='AMEDDE')
class SpatialAttentionMap(layers.Layer):
    def __init__(self, **kw): super().__init__(**kw)

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(16, 7, padding='same', use_bias=False, activation='relu')
        self.conv2 = layers.Conv2D(1,  1, padding='same', use_bias=False, activation='sigmoid')
        super().build(input_shape)

    def call(self, x, training=None):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx  = tf.reduce_max(x,  axis=-1, keepdims=True)
        sq  = tf.concat([avg, mx], axis=-1)
        attn = self.conv2(self.conv1(sq))
        return x * attn

    def compute_output_shape(self, s): return s
    def get_config(self): return super().get_config()


# ============================================================
# BLOCK HELPERS
# ============================================================
_REG = tf.keras.regularizers.l2(L2_REG)

def _c(x, f, k=3, d=1, nm=''):
    return layers.Conv2D(f, k, padding='same', dilation_rate=d,
                         use_bias=False, kernel_regularizer=_REG, name=nm)(x)

def _bn(x, nm): return layers.BatchNormalization(name=nm)(x)
def _rl(x):     return layers.ReLU()(x)
def _proj(x, ch, nm): return _rl(_bn(_c(x, ch, 1, nm=f'{nm}p'), f'{nm}b'))


def msfe_block(x, f, nm='ms'):
    ch = f // len(DILATION_RATES)
    br = [_rl(_bn(_c(x, ch, d=r, nm=f'{nm}d{r}'), f'{nm}b{r}')) for r in DILATION_RATES]
    return _rl(_bn(_c(layers.Concatenate(name=f'{nm}cat')(br), f, 1, nm=f'{nm}fc'), f'{nm}fb'))


def res_conv_block(x, f, dr=DROPOUT_RATE, nm='rc'):
    r = _rl(_bn(_c(x, f, 1, nm=f'{nm}r'), f'{nm}rb')) if int(x.shape[-1]) != f else x
    h = _rl(_bn(_c(x, f, 3, nm=f'{nm}c1'), f'{nm}b1'))
    h = layers.Dropout(dr, name=f'{nm}dr')(h)
    h = _rl(_bn(_c(h, f, 3, nm=f'{nm}c2'), f'{nm}b2'))
    h = _bn(_c(h, f, 3, nm=f'{nm}c3'), f'{nm}b3')
    return _rl(layers.Add(name=f'{nm}add')([h, r]))


def dense_decoder_block(x, skips, f, nm='dd'):
    x = layers.UpSampling2D(2, interpolation='bilinear', name=f'{nm}up')(x)
    x = _rl(_bn(_c(x, f, 1, nm=f'{nm}p'), f'{nm}pb'))

    gated = [x]
    for i, sk in enumerate(skips):
        sk_p = _rl(_bn(_c(sk, f, 1, nm=f'{nm}sk{i}p'), f'{nm}sk{i}b'))
        sk_p = ResizeToMatch(name=f'{nm}rs{i}')([sk_p, x])
        sk_g = KANGateAttention(f, name=f'{nm}ga{i}')([sk_p, x])
        gated.append(sk_g)

    ct = layers.Concatenate(name=f'{nm}cat')(gated) if len(gated) > 1 else gated[0]
    # SpatialDropout2D — anti-overfitting di concat output
    ct = layers.SpatialDropout2D(0.15, name=f'{nm}sd')(ct)

    ch = f // 3
    br = [_rl(_bn(_c(ct, ch, d=r, nm=f'{nm}dr{r}'), f'{nm}db{r}')) for r in [1, 3, 6]]
    fs = _rl(_bn(_c(layers.Concatenate()(br), f, 1, nm=f'{nm}fc'), f'{nm}fb'))

    fs = CBAMBlock(f, name=f'{nm}cb')(fs)
    fs = SpatialAttentionMap(name=f'{nm}sam')(fs)

    res = _rl(_bn(_c(x, f, 1, nm=f'{nm}res'), f'{nm}rsb'))
    return _rl(layers.Add()([res_conv_block(fs, f, dr=DROPOUT_RATE, nm=f'{nm}rc'), res]))


def hr_path(inputs, f=16, nm='hr'):
    h = layers.SeparableConv2D(f, 3, padding='same', use_bias=False, name=f'{nm}s1')(inputs)
    h = _rl(_bn(h, f'{nm}b1'))
    h = layers.SeparableConv2D(f, 3, padding='same', use_bias=False, name=f'{nm}s2')(h)
    h = _rl(_bn(h, f'{nm}b2'))
    h = layers.SeparableConv2D(f, 3, padding='same', use_bias=False, name=f'{nm}s3')(h)
    h = _rl(_bn(h, f'{nm}b3'))
    return h


# ============================================================
# DMSA-KANet ARCHITECTURE (unchanged)
# ============================================================
def build_amedde_net_v9(input_shape=(256, 256, 3), n_labels=1):
    inputs = layers.Input(shape=input_shape, name='input')
    hr = hr_path(inputs, f=16, nm='hr')

    backbone = tf.keras.applications.EfficientNetB4(
        include_top=False, weights='imagenet', input_tensor=inputs
    )
    for layer in backbone.layers:
        layer.trainable = False

    s1 = backbone.get_layer('stem_activation').output
    s2 = backbone.get_layer('block2d_add').output
    s3 = backbone.get_layer('block3d_add').output
    s4 = backbone.get_layer('block5f_add').output
    s5 = backbone.get_layer('block7b_add').output

    p1 = _proj(s1, 64,  'p1')
    p2 = _proj(s2, 128, 'p2')
    p3 = _proj(s3, 256, 'p3')
    p4 = _proj(s4, 512, 'p4')
    p5 = _proj(s5, 512, 'p5')

    msfe_o = msfe_block(p5, 1024, nm='ms')
    vit_o  = ViTBottleneck(dim=1024, num_heads=8, num_blocks=4,
                            mlp_ratio=4.0, dropout=0.1, name='vit')(msfe_o)
    bn_out = CBAMBlock(1024, name='bncb')(
        ChannelAttentionKAN(1024, KAN_GRID, KAN_ORDER, name='bnka')(vit_o)
    )

    d4 = dense_decoder_block(bn_out, [p4],         512, nm='d4')
    d3 = dense_decoder_block(d4,     [p3, p4],     256, nm='d3')
    d2 = dense_decoder_block(d3,     [p2, p3, p4], 128, nm='d2')
    d1 = dense_decoder_block(d2,     [p1, p2, p3],  64, nm='d1')
    d0 = dense_decoder_block(d1,     [p1, hr],       32, nm='d0')

    out_main = layers.Conv2D(n_labels, 1, activation='sigmoid', name='out_main')(d0)
    od1 = layers.UpSampling2D(2, interpolation='bilinear')(
              layers.Conv2D(n_labels, 1, activation='sigmoid', name='od1')(d1))
    od2 = layers.UpSampling2D(4, interpolation='bilinear')(
              layers.Conv2D(n_labels, 1, activation='sigmoid', name='od2')(d2))
    od3 = layers.UpSampling2D(8, interpolation='bilinear')(
              layers.Conv2D(n_labels, 1, activation='sigmoid', name='od3')(d3))

    model = models.Model(inputs=inputs, outputs=[out_main, od1, od2, od3],
                         name='AMEDDE_Net_v11')
    return model, backbone


def unfreeze_backbone(backbone):
    unfrozen = 0
    for layer in backbone.layers:
        if any(b in layer.name for b in
               ['block4', 'block5', 'block6', 'block7',
                'top_conv', 'top_bn', 'top_activation']):
            layer.trainable = True
            unfrozen += 1
        else:
            layer.trainable = False
    print(f"  Backbone: {unfrozen} layers unfrozen (block4+5+6+7)")


# ============================================================
# LOSS FUNCTIONS (unchanged)
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


# ============================================================
# CUSTOM TRAINER
# ============================================================
class AMEDDETrainer(tf.keras.Model):
    def __init__(self, base_model, ds_weights=None):
        super().__init__()
        self.base_model = base_model
        self.ds_weights = ds_weights or DS_WEIGHTS

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            loss  = sum(w * combined_loss(y, p) for p, w in zip(preds, self.ds_weights))
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g
                 for g in tape.gradient(loss, self.trainable_variables)]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss,
                'dice_coefficient': dice_coefficient(y, preds[0]),
                'accuracy': pixel_accuracy(y, preds[0])}

    def test_step(self, data):
        x, y  = data
        preds = self(x, training=False)
        loss  = sum(w * combined_loss(y, p) for p, w in zip(preds, self.ds_weights))
        return {'loss': loss,
                'dice_coefficient': dice_coefficient(y, preds[0]),
                'accuracy': pixel_accuracy(y, preds[0])}


def build_and_compile(lr=LR_PHASE1):
    base, backbone = build_amedde_net_v9()
    trainer = AMEDDETrainer(base)
    trainer.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), run_eagerly=False)
    return trainer, base, backbone


# ============================================================
# METRICS
# ============================================================
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


# ============================================================
# DATA LOADING — image list and fold roles come from the fold file
# ============================================================
folds_df = pd.read_csv(FOLD_CSV)
names = folds_df['Image_name'].astype(str).tolist()
print(f"Loading {len(names)} images listed in {FOLD_CSV} ...")

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
print(f"Image shape: {image_dataset.shape}, Mask shape: {mask_dataset.shape}")
print(f"Mean lesion fraction: {mask_dataset.mean()*100:.2f}%")


# ============================================================
# AUGMENTATION — 6x, training images only (unchanged)
# ============================================================
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


# ============================================================
# TTA PREDICT
# ============================================================
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
# PATIENT-LEVEL 5-FOLD CROSS-VALIDATION (roles from the fold file)
# ============================================================

print(f"\n{'='*80}")
print("CHECKING CHECKPOINTS...")
print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
fold_results, fold_histories = load_completed_folds()
completed_folds = [r['fold'] for r in fold_results]
if completed_folds:
    print(f"  Completed folds: {completed_folds}")
    if max(completed_folds) < N_SPLITS:
        print(f"  Resuming from fold {max(completed_folds) + 1}")
    else:
        print(f"  All {N_SPLITS} folds completed — going to summary")
else:
    print(f"  No checkpoint found — starting from fold 1")
print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"DMSA-KANet — {N_SPLITS}-FOLD CROSS-VALIDATION")
print(f"Encoder  : EfficientNetB4 + HR-Path 256×256")
print(f"Bottleneck: MSFE → ViT(4block,8h,CLS,KAN) → KAN-CBAM @ 8×8")
print(f"Decoder  : Dense skip + Dual attention (CBAM + SAM) + MSFE")
print(f"Loss     : FocalTversky(0.4) + Lovász(0.3) + Focal(0.2) + Dice(0.1)")
print(f"Phase1={PHASE1_EPOCHS}ep LR={LR_PHASE1} → Phase2={EPOCHS-PHASE1_EPOCHS}ep LR={LR_PHASE2}")
print(f"Epochs={EPOCHS} | Batch={BATCH_SIZE} | Dropout={DROPOUT_RATE} | L2={L2_REG}")
print(f"Augmentation: {AUG_FACTOR}x | Dataset: {DATASET}")
print(f"Split: {FOLD_CSV} | seed: {SEED}")
print(f"{'='*80}\n")


for fold_num in range(1, N_SPLITS + 1):
    k = fold_num - 1
    if is_fold_completed(fold_num):
        print(f"\n[SKIP] Fold {fold_num} already completed — using saved results")
        continue

    print(f"\n{'='*60} FOLD {fold_num}/{N_SPLITS} {'='*60}")
    role = folds_df[f'role_fold{k}'].values
    train_idx = np.where(role == 'train')[0]
    val_idx   = np.where(role == 'val')[0]
    test_idx  = np.where(role == 'test')[0]
    pats = folds_df['Patient_ID'].values
    assert not (set(pats[train_idx]) & set(pats[test_idx])), "patient overlap train/test"
    assert not (set(pats[val_idx])   & set(pats[test_idx])), "patient overlap val/test"
    assert not (set(pats[train_idx]) & set(pats[val_idx])),  "patient overlap train/val"

    X_train, y_train = image_dataset[train_idx], mask_dataset[train_idx]
    X_val,   y_val   = image_dataset[val_idx],   mask_dataset[val_idx]
    X_test,  y_test  = image_dataset[test_idx],  mask_dataset[test_idx]
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} images "
          f"({len(set(pats[train_idx]))}/{len(set(pats[val_idx]))}/{len(set(pats[test_idx]))} patients)")

    # Same augmented training set after a restart: seed the augmentation per fold.
    np.random.seed(SEED + fold_num)
    X_train, y_train = augment_mammography(X_train, y_train)

    # ---------------- epoch-level resume (survives machine shutdowns) ----------------
    # After every epoch the model weights and the training state (phase, next epoch, learning
    # rate, plateau counter, best validation values) are saved. On restart the fold continues
    # from the last finished epoch. Note: Adam moment estimates are re-initialised on resume.
    state_path = os.path.join(CHECKPOINT_DIR, f'resume_fold_{fold_num}.json')
    wts_path   = os.path.join(CHECKPOINT_DIR, f'resume_fold_{fold_num}.weights.h5')
    hist_csv   = f'{CKPT_PREFIX}_fold_{fold_num}_history.csv'
    state = None
    if os.path.exists(state_path) and os.path.exists(wts_path):
        state = json.load(open(state_path))
        print(f"\n  [RESUME] fold {fold_num}: phase {state['phase']}, continuing at epoch "
              f"{state['next_epoch'] + 1}, lr={state['lr']:.2e}, best val_dice={state['best_dice']:.4f}")
        if os.path.exists(hist_csv):      # drop rows of an epoch whose state was not saved
            h = pd.read_csv(hist_csv)
            h.iloc[:state['next_epoch']].to_csv(hist_csv, index=False)

    start_lr = state['lr'] if (state and state['phase'] == 1) else LR_PHASE1
    trainer, base_model, backbone = build_and_compile(lr=start_lr)

    if fold_num == 1 and not completed_folds and state is None:
        base_model.summary()
        print_complexity_table(base_model, (SIZE, SIZE, 3),
                              model_name='DMSA-KANet')

    class SaveBestModel(tf.keras.callbacks.Callback):
        def __init__(self, base_model, filepath, monitor='val_dice_coefficient',
                     mode='max', verbose=1, initial_best=None):
            super().__init__()
            self.base_model = base_model; self.filepath = filepath
            self.monitor = monitor; self.verbose = verbose; self.mode = mode
            if initial_best is not None:
                self.best = initial_best
            else:
                self.best = -np.inf if mode == 'max' else np.inf

        def on_epoch_end(self, epoch, logs=None):
            cur = logs.get(self.monitor)
            if cur is None: return
            improved = (cur > self.best) if self.mode == 'max' else (cur < self.best)
            if improved:
                if self.verbose:
                    print(f"\n  [✓ Checkpoint] {self.monitor} "
                          f"{self.best:.4f}→{cur:.4f} | {self.filepath}")
                self.best = cur
                self.base_model.save(self.filepath)

    class ResumablePlateau(tf.keras.callbacks.ReduceLROnPlateau):
        def __init__(self, init_wait=0, init_best=None, **kw):
            super().__init__(**kw); self._iw = init_wait; self._ib = init_best
        def on_train_begin(self, logs=None):
            super().on_train_begin(logs)
            if self._ib is not None:
                self.best = self._ib; self.wait = self._iw
            self._ib = None

    class SaveResumeState(tf.keras.callbacks.Callback):
        def __init__(self, phase, plateau, best_dice_cb, best_loss_cb, elapsed_prev):
            super().__init__()
            self.phase = phase; self.plateau = plateau
            self.bd = best_dice_cb; self.bl = best_loss_cb; self.elapsed_prev = elapsed_prev
            self.t0 = datetime.now()
        def on_epoch_end(self, epoch, logs=None):
            base_model.save_weights(wts_path)
            st = {'phase': self.phase, 'next_epoch': epoch + 1,
                  'lr': float(np.array(self.model.optimizer.learning_rate)),
                  'plateau_wait': int(self.plateau.wait), 'plateau_best': float(self.plateau.best),
                  'best_dice': float(self.bd.best), 'best_loss': float(self.bl.best),
                  'elapsed_s': self.elapsed_prev + (datetime.now() - self.t0).total_seconds()}
            with open(state_path + '.tmp', 'w') as f:
                json.dump(st, f)
            os.replace(state_path + '.tmp', state_path)

    def make_callbacks(phase, append_csv=False, best_dice_so_far=None, best_loss_so_far=None,
                       plateau_wait=0, plateau_best=None, elapsed_prev=0.0):
        plateau = ResumablePlateau(init_wait=plateau_wait, init_best=plateau_best,
                                   monitor='val_dice_coefficient', factor=0.5, patience=5,
                                   mode='max', min_lr=1e-7, verbose=1)
        bd = SaveBestModel(base_model, f'{CKPT_PREFIX}_fold_{fold_num}_best_dice.keras',
                           'val_dice_coefficient', 'max', verbose=1, initial_best=best_dice_so_far)
        bl = SaveBestModel(base_model, f'{CKPT_PREFIX}_fold_{fold_num}_best_loss.keras',
                           'val_loss', 'min', verbose=0, initial_best=best_loss_so_far)
        return [plateau, bd, bl,
                tf.keras.callbacks.CSVLogger(hist_csv, append=append_csv),
                SaveResumeState(phase, plateau, bd, bl, elapsed_prev)]   # must stay last

    start = datetime.now()
    elapsed_prev = state['elapsed_s'] if state else 0.0

    # ── Phase 1 ──────────────────────────────────────────────
    if state is None or state['phase'] == 1:
        p1_start = state['next_epoch'] if state else 0
        if state:
            base_model.load_weights(wts_path)
        print(f"\n  [Phase 1] backbone FROZEN — LR={start_lr} — epochs {p1_start+1}-{PHASE1_EPOCHS}")
        if p1_start < PHASE1_EPOCHS:
            trainer.fit(
                X_train, y_train, validation_data=(X_val, y_val),
                batch_size=BATCH_SIZE, epochs=PHASE1_EPOCHS, initial_epoch=p1_start,
                callbacks=make_callbacks(1, append_csv=bool(state),
                                         best_dice_so_far=state['best_dice'] if state else None,
                                         best_loss_so_far=state['best_loss'] if state else None,
                                         plateau_wait=state['plateau_wait'] if state else 0,
                                         plateau_best=state['plateau_best'] if state else None,
                                         elapsed_prev=elapsed_prev),
                shuffle=True, verbose=1
            )
        state = json.load(open(state_path))
        state.update({'phase': 2, 'lr': LR_PHASE2, 'plateau_wait': 0, 'plateau_best': None})
        json.dump(state, open(state_path, 'w'))
        print(f"\n  Phase 1 done — best val_dice: {state['best_dice']:.4f}, best val_loss: {state['best_loss']:.4f}")
    else:
        base_model.load_weights(wts_path)

    # ── Phase 2 ──────────────────────────────────────────────
    p2_start = max(state['next_epoch'], PHASE1_EPOCHS)
    print(f"\n  [Phase 2] unfreeze block4-7 — LR={state['lr']} — epochs {p2_start+1}-{EPOCHS}")
    print(f"  Checkpointing continues (best val dice so far={state['best_dice']:.4f})")
    unfreeze_backbone(backbone)
    trainer.compile(optimizer=Adam(learning_rate=state['lr'], clipnorm=0.5),
                    run_eagerly=False)
    if p2_start < EPOCHS:
        trainer.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE, epochs=EPOCHS, initial_epoch=p2_start,
            callbacks=make_callbacks(2, append_csv=True,
                                     best_dice_so_far=state['best_dice'],
                                     best_loss_so_far=state['best_loss'],
                                     plateau_wait=state.get('plateau_wait', 0),
                                     plateau_best=state.get('plateau_best'),
                                     elapsed_prev=state.get('elapsed_s', 0.0)),
            shuffle=True, verbose=1
        )
    state = json.load(open(state_path))
    elapsed = pd.Timedelta(seconds=state.get('elapsed_s', (datetime.now() - start).total_seconds()))

    hdf = pd.read_csv(hist_csv)
    combined = {c: hdf[c].astype(float).tolist() for c in hdf.columns if c != 'epoch'}

    best_path = f'{CKPT_PREFIX}_fold_{fold_num}_best_dice.keras'
    best_dice = max(combined['val_dice_coefficient'])
    best_ep   = int(np.argmax(combined['val_dice_coefficient']) + 1)
    if os.path.exists(best_path):
        print(f"\n  Loading best-validation model: {best_path}")
        best_base = tf.keras.models.load_model(best_path, compile=False)
        best_dice = max(combined['val_dice_coefficient'])
        best_ep   = np.argmax(combined['val_dice_coefficient']) + 1
        print(f"  Best val_dice: {best_dice:.4f} (epoch {best_ep})")
    else:
        best_base = base_model

    # threshold chosen on the VALIDATION set, then applied unchanged to the TEST set
    thr     = find_optimal_threshold(y_val, tta_predict(best_base, X_val))
    y_pred  = tta_predict(best_base, X_test)
    metrics = calculate_metrics(y_test, y_pred, threshold=thr)

    test_names = [names[i] for i in test_idx]
    pim = per_image_metrics(y_test, y_pred, thr, test_names)
    pim.insert(0, 'fold', fold_num)
    pim = pim.merge(folds_df[['Image_name', 'Patient_ID', 'Label']], on='Image_name', how='left')
    pim.to_csv(os.path.join(OUT_DIR, f'per_image_metrics_fold_{fold_num}.csv'), index=False)
    metrics['dice_per_image_mean'] = float(pim['dice'].mean())
    metrics['iou_per_image_mean']  = float(pim['iou'].mean())
    np.savez_compressed(os.path.join(OUT_DIR, f'test_probabilities_fold_{fold_num}.npz'),
                        names=np.array(test_names),
                        prob=(y_pred[..., 0] * 255).round().astype(np.uint8))

    fold_result = {
        'fold': fold_num,
        'train_loss': float(combined['loss'][-1]),
        'val_loss':   float(combined['val_loss'][-1]),
        'execution_time': str(elapsed),
        'lr_phase1': LR_PHASE1, 'lr_phase2': LR_PHASE2, 'batch_size': BATCH_SIZE,
        'best_val_dice': float(best_dice), 'best_epoch': best_ep,
        **metrics
    }
    fold_results.append(fold_result)
    fold_histories.append(combined)

    # persist fold result (allows resuming)
    save_fold_result(fold_num, fold_result, combined)
    for p in (state_path, wts_path):
        if os.path.exists(p): os.remove(p)

    label_map = [('DICE','dice'),('IOU','iou'),('P','precision'),('R','recall'),
                 ('F1','f1_score'),('ACC','accuracy'),('ROC','roc_auc'),
                 ('FPR','fpr'),('FNR','fnr'),('TIF','tif'),
                 ('DICEimg','dice_per_image_mean'),('IOUimg','iou_per_image_mean')]
    print(f"\n--- Fold {fold_num} TEST results (thr={metrics['threshold']:.3f}, chosen on val) ---")
    for lbl, key in label_map:
        print(f"  {lbl:<6} {metrics[key]:>8.4f}")
    print(f"  Val Loss: {fold_result['val_loss']:.4f}  Time: {elapsed}")

    base_model.save(f'{CKPT_PREFIX}_fold_{fold_num}.keras')
    K.clear_session()


# ============================================================
# SUMMARY
# ============================================================
print(f"\n\n{'='*100}")
print("DMSA-KANet — CROSS-VALIDATION SUMMARY")
print(f"{'='*100}")
print(f"{'Method':<22} {'Loss':>7} {'ACC':>7} {'DICE':>7} {'P':>7} {'R':>7} "
      f"{'F1':>7} {'IOU':>7} {'ROC':>7} {'FPR':>7} {'FNR':>7} {'TIF':>7}")
print("-"*100)

results_df = pd.DataFrame(fold_results).sort_values('fold').reset_index(drop=True)
summary_cols = ['val_loss','accuracy','dice','precision','recall',
                'f1_score','iou','roc_auc','fpr','fnr','tif',
                'dice_per_image_mean','iou_per_image_mean']

for _, row in results_df.iterrows():
    print(f"  Fold {int(row['fold']):<16} " + " ".join(f"{row[c]:>7.4f}" for c in summary_cols))

print("-"*100)
means = results_df[summary_cols].mean()
stds  = results_df[summary_cols].std()
print(f"  {'Mean':<22} " + " ".join(f"{means[c]:>7.4f}" for c in summary_cols))
print(f"  {'Std':<22}  " + " ".join(f"{stds[c]:>7.4f}"  for c in summary_cols))
print(f"{'='*100}\n")

results_df.to_csv(os.path.join(OUT_DIR, 'dmsa_kanet_kfold_results.csv'), index=False)
pd.DataFrame([{'Metric': lbl, 'Mean': f"{results_df[c].mean():.4f}",
               'Std': f"{results_df[c].std():.4f}",
               'Min': f"{results_df[c].min():.4f}",
               'Max': f"{results_df[c].max():.4f}"}
              for c, lbl in {'val_loss':'LOSS','accuracy':'ACC','dice':'DICE',
                              'precision':'P','recall':'R','f1_score':'F1',
                              'iou':'IOU','roc_auc':'ROC','fpr':'FPR',
                              'fnr':'FNR','tif':'TIF'}.items()]
             ).to_csv(os.path.join(OUT_DIR, 'dmsa_kanet_kfold_summary.csv'), index=False)

# ─── Paper-style table (same format as the baselines) ───────
cv_table_rows = []
for _, row in results_df.iterrows():
    cv_table_rows.append({
        'Method': f"AMEDDE-Net Fold {int(row['fold'])}",
        'Loss':   row['val_loss'],
        'ACC':    row['accuracy'],
        'DICE':   row['dice'],
        'P':      row['precision'],
        'R':      row['recall'],
        'F1':     row['f1_score'],
        'IOU':    row['iou'],
        'ROC':    row['roc_auc'],
        'FPR':    row['fpr'],
        'FNR':    row['fnr'],
        'TIF':    row['tif'],
    })
cv_table_df = pd.DataFrame(cv_table_rows)
cols_paper = ['Loss','ACC','DICE','P','R','F1','IOU','ROC','FPR','FNR','TIF']
mean_row = dict(zip(['Method']+cols_paper, ['Mean'] + [f"{cv_table_df[c].mean():.4f}" for c in cols_paper]))
std_row  = dict(zip(['Method']+cols_paper, ['Std']  + [f"{cv_table_df[c].std():.4f}"  for c in cols_paper]))
cv_summary_df = pd.concat([cv_table_df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])],
                          ignore_index=True)
cv_table_df.to_csv(os.path.join(OUT_DIR, 'dmsa_kanet_cv_table_results.csv'), index=False)
cv_summary_df.to_csv(os.path.join(OUT_DIR, 'dmsa_kanet_cv_table_summary.csv'), index=False)
print(f"  Saved to {OUT_DIR}")

metrics_plot = ['dice','iou','precision','recall','f1_score','accuracy','roc_auc','fpr','fnr','tif']
ncols = 4; nrows = (len(metrics_plot) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()
fig.suptitle('DMSA-KANet — K-Fold Metrics', fontsize=16, fontweight='bold')
for idx, m in enumerate(metrics_plot):
    ax = axes[idx]; v = results_df[m].values
    ax.bar(range(1, len(v)+1), v, color='salmon' if m in ['fpr','fnr'] else 'steelblue', alpha=0.75)
    ax.axhline(np.mean(v), color='red', linestyle='--', lw=2, label=f'μ={np.mean(v):.4f}')
    ax.set_title(m.upper()); ax.set_xlabel('Fold'); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
for idx in range(len(metrics_plot), len(axes)): axes[idx].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'dmsa_kanet_kfold_metrics.png'), dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('DMSA-KANet — Training History', fontsize=14, fontweight='bold')
for i, h in enumerate(fold_histories):
    if 'loss' in h:
        axes[0].plot(h['loss'], alpha=0.6, label=f'F{i+1} tr')
        axes[0].plot(h['val_loss'], alpha=0.6, ls='--', label=f'F{i+1} val')
    if 'dice_coefficient' in h:
        axes[1].plot(h['dice_coefficient'], alpha=0.6, label=f'F{i+1} tr')
        axes[1].plot(h['val_dice_coefficient'], alpha=0.6, ls='--', label=f'F{i+1} val')
    if 'accuracy' in h:
        axes[2].plot(h['accuracy'], alpha=0.6, label=f'F{i+1} tr')
        axes[2].plot(h['val_accuracy'], alpha=0.6, ls='--', label=f'F{i+1} val')
for ax, t in zip(axes, ['Loss', 'Dice', 'Accuracy']):
    ax.set_title(t); ax.set_xlabel('Epoch'); ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'dmsa_kanet_training_history.png'), dpi=300, bbox_inches='tight')

best = results_df.iloc[results_df['dice'].idxmax()]
print(f"Best: Fold {int(best['fold'])} — DICE={best['dice']:.4f}, IOU={best['iou']:.4f}, ACC={best['accuracy']:.4f}")
print(f"Outputs in: {OUT_DIR}")
print(f"\n{'='*80}\nDONE")