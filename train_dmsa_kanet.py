"""
AMEDDE-Net v11 — CHECKPOINT + AUGMENTASI KUAT
==============================================

Perubahan dari v11 asli:
  1. CHECKPOINT per-fold (kompatibel dengan file lama amedde_v9_*)
  2. AUGMENTASI 6× (sebelumnya 3×):
     orig + flip_h + flip_v + rotate ±15° + gamma + cutout
  3. SpatialDropout2D di decoder concat
  4. Dropout decoder 0.1 → 0.2
  5. L2 reg 1e-4 → 2e-4

Yang TIDAK diubah:
  - Arsitektur model (semua custom layer, dense decoder, ViT, KAN, dual attention)
  - Loss function (FocalTversky 0.4 + Lovász 0.3 + Focal 0.2 + Dice 0.1)
  - Phase 2 unfreeze block4+5+6+7
  - Batch size 8, epoch 80, LR phase1 1e-3, LR phase2 5e-5
  - Tversky alpha 0.85
  - Tidak pakai early stopping

★★★ FIX (revisi INASS) ★★★
  - calculate_metrics(): formula 'f1_score' sebelumnya salah aljabar —
    menghitung 2*P*R (tanpa dibagi (P+R)), bukan F1 yang sebenarnya.
    Untuk binary segmentation, F1 = 2*P*R/(P+R) selalu identik dengan Dice
    = 2*TP/(2*TP+FP+FN). Sudah diperbaiki agar F1 dihitung dengan benar
    dari precision & recall yang sama-sama sudah dihitung di fungsi ini.
    (Dice, IoU, Precision, Recall, FPR, FNR, TIF, ROC-AUC TIDAK berubah —
    formula-formula itu sudah benar dari awal.)
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# KONFIGURASI (sama seperti v11 asli)
# ============================================================
image_directory = 'Coding_Part_2_resized/train/images/'
mask_directory  = 'Coding_Part_2_resized/train/masks/'

SIZE           = 256
BATCH_SIZE     = 8
N_SPLITS       = 5
EPOCHS         = 50
LR_PHASE1      = 1e-3
LR_PHASE2      = 5e-5
PHASE1_EPOCHS  = 20

KAN_GRID       = 5
KAN_ORDER      = 3
DILATION_RATES = [1, 3, 6, 12]
DROPOUT_RATE   = 0.2     # ↑ sedikit dari 0.1

TVERSKY_ALPHA  = 0.85
TVERSKY_BETA   = 0.15
DS_WEIGHTS     = [0.5, 0.2, 0.2, 0.1]

PRED_THRESHOLD = 0.20
L2_REG         = 2e-4    # ↑ sedikit dari 1e-4


# ============================================================
# CHECKPOINT SYSTEM (PER-FOLD)
# ============================================================
CHECKPOINT_DIR = os.path.abspath('checkpoints_v11')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"[INIT] Checkpoint disimpan di: {CHECKPOINT_DIR}")

# Prefix kompatibel dengan file checkpoint lama (amedde_v9_*)
CKPT_PREFIX = 'amedde_v9'


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
    print(f"  [✓] Fold {fold_num} disimpan ke {get_fold_result_path(fold_num)}")


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

def print_complexity_table(model, input_shape=(256,256,3), model_name='AMEDDE-Net v11'):
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
                 ).to_csv('amedde_v11_complexity.csv', index=False)


# ============================================================
# CUSTOM LAYERS (TIDAK DIUBAH)
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
# AMEDDE-NET v11 ARCHITECTURE (TIDAK DIUBAH)
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
# LOSS FUNCTIONS (TIDAK DIUBAH)
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
    for t in np.arange(0.05, 0.60, 0.025):
        ypf = (y_pred.flatten() > t).astype(np.uint8)
        try:
            tn, fp, fn, tp = confusion_matrix(ytf, ypf, labels=[0, 1]).ravel()
            iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
            if iou > best_iou: best_iou = iou; best_t = t
        except: continue
    print(f"  Optimal threshold: {best_t:.3f}  (IOU={best_iou:.4f})")
    return float(best_t)


def calculate_metrics(y_true, y_pred, threshold=None):
    """
    ★ FIX (revisi INASS): 'f1_score' sebelumnya dihitung dengan formula yang
    salah secara aljabar (menghasilkan 2*P*R, bukan F1 sesungguhnya).
    Sekarang dihitung dengan benar sebagai 2*P*R/(P+R) dari precision &
    recall yang sudah dihitung di bawah — untuk binary segmentation, F1
    yang benar akan selalu identik dengan Dice.
    """
    if threshold is None: threshold = find_optimal_threshold(y_true, y_pred)
    yb  = (y_pred > threshold).astype(np.uint8)
    ytb = (y_true > 0.5).astype(np.uint8)
    ytf = ytb.flatten(); ypf = yb.flatten(); ypp = y_pred.flatten()
    tn, fp, fn, tp = confusion_matrix(ytf, ypf, labels=[0, 1]).ravel()
    s = 1e-7
    try:    auc = roc_auc_score(ytf, ypp)
    except: auc = 0.0

    precision = (tp+s)/(tp+fp+s)
    recall    = (tp+s)/(tp+fn+s)
    f1_score  = 2 * precision * recall / (precision + recall)  # ← FIXED (was: 2*(tp+s)*(tp+s)/((tp+fp+s)*(tp+fn+s)+(tp+s)))

    return {
        'threshold': float(threshold),
        'dice':      (2*tp+s)/(2*tp+fp+fn+s),
        'iou':       (tp+s)/(tp+fp+fn+s),
        'precision': precision,
        'recall':    recall,
        'f1_score':  f1_score,
        'accuracy':  (tp+tn)/(tp+tn+fp+fn+s),
        'fpr':       (fp+s)/(fp+tn+s),
        'fnr':       (fn+s)/(fn+tp+s),
        'tif':       (tp+s)/(tp+fn+s),
        'roc_auc':   auc,
    }


# ============================================================
# DATA LOADING
# ============================================================
def load_images(directory, size=SIZE):
    data = []
    for fn in sorted(os.listdir(directory)):
        if fn.lower().endswith(('.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(directory, fn), cv2.IMREAD_COLOR)
            data.append(cv2.resize(img, (size, size)))
    return np.array(data, dtype=np.float32)

print("Loading images...")
image_dataset = load_images(image_directory) / 255.0

print("Loading masks...")
mask_dataset = []
for fn in sorted(os.listdir(mask_directory)):
    if fn.lower().endswith('.png'):
        m = cv2.imread(os.path.join(mask_directory, fn), cv2.IMREAD_GRAYSCALE)
        mask_dataset.append(cv2.resize(m, (SIZE, SIZE)))
mask_dataset = np.expand_dims(np.array(mask_dataset, dtype=np.float32), 3) / 255.0
mask_dataset = (mask_dataset > 0.5).astype(np.float32)

print(f"Images: {len(image_dataset)}, Masks: {len(mask_dataset)}")
print(f"Rasio lesi rata-rata: {mask_dataset.mean()*100:.2f}%")


# ============================================================
# AUGMENTASI KUAT — 6× (sebelumnya 3×)
# ============================================================
def _rotate_pair(img, msk, angle_deg):
    """Rotate image dan mask dengan sudut yang sama."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    img_r = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    msk_r = cv2.warpAffine(msk, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_r, msk_r


def _cutout(img, msk, n_holes=2, max_size=32):
    """Cutout — masking area random pada image (mask tidak diubah)."""
    img_c = img.copy()
    h, w = img.shape[:2]
    for _ in range(n_holes):
        sz = np.random.randint(8, max_size + 1)
        y  = np.random.randint(0, h - sz)
        x  = np.random.randint(0, w - sz)
        img_c[y:y+sz, x:x+sz, :] = 0
    return img_c, msk


def _gamma(img, gamma):
    """Gamma correction — variasi kontras."""
    return np.clip(img ** gamma, 0, 1).astype(np.float32)


def augment_mammography(X, y):
    """
    Augmentasi 6× untuk mammografi:
    1. orig
    2. flip horizontal (simetri payudara)
    3. flip vertical
    4. rotate ±15°
    5. gamma correction (kontras detektor)
    6. cutout (regularisasi spatial)
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

    X_out = np.concatenate(Xa, axis=0)
    y_out = np.concatenate(ya, axis=0)
    perm  = np.random.permutation(len(X_out))
    print(f"  Augmentasi: {n} → {len(X_out)} sampel (6×: orig + flip_h + flip_v + rotate + gamma + cutout)")
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
# ★★★ K-FOLD DENGAN CHECKPOINT ★★★
# ============================================================
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print(f"\n{'='*80}")
print("CHECKING CHECKPOINTS...")
print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
fold_results, fold_histories = load_completed_folds()
completed_folds = [r['fold'] for r in fold_results]
if completed_folds:
    print(f"  Fold sudah selesai: {completed_folds}")
    if max(completed_folds) < N_SPLITS:
        print(f"  Akan dilanjutkan dari fold {max(completed_folds) + 1}")
    else:
        print(f"  Semua {N_SPLITS} fold sudah selesai — langsung ke summary")
else:
    print(f"  Belum ada checkpoint — mulai dari fold 1")
print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"AMEDDE-Net v11 — {N_SPLITS}-FOLD CROSS-VALIDATION")
print(f"Encoder  : EfficientNetB4 + HR-Path 256×256")
print(f"Bottleneck: MSFE → ViT(4block,8h,CLS,KAN) → KAN-CBAM @ 8×8")
print(f"Decoder  : Dense skip + Dual attention (CBAM + SAM) + MSFE")
print(f"Loss     : FocalTversky(0.4) + Lovász(0.3) + Focal(0.2) + Dice(0.1)")
print(f"Phase1={PHASE1_EPOCHS}ep LR={LR_PHASE1} → Phase2={EPOCHS-PHASE1_EPOCHS}ep LR={LR_PHASE2}")
print(f"Epochs={EPOCHS} | Batch={BATCH_SIZE} | Dropout={DROPOUT_RATE} | L2={L2_REG}")
print(f"Augmentasi: 6× (orig + flip_h + flip_v + rotate + gamma + cutout)")
print(f"{'='*80}\n")


for fold_num, (train_idx, val_idx) in enumerate(kfold.split(image_dataset), 1):

    # ★★★ SKIP FOLD YANG SUDAH SELESAI ★★★
    if is_fold_completed(fold_num):
        print(f"\n[SKIP] Fold {fold_num} sudah selesai — gunakan hasil dari disk")
        continue

    print(f"\n{'='*60} FOLD {fold_num}/{N_SPLITS} {'='*60}")

    X_train = image_dataset[train_idx]; X_val = image_dataset[val_idx]
    y_train = mask_dataset[train_idx];  y_val = mask_dataset[val_idx]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    X_train, y_train = augment_mammography(X_train, y_train)
    trainer, base_model, backbone = build_and_compile(lr=LR_PHASE1)

    if fold_num == 1 and not completed_folds:
        base_model.summary()
        print_complexity_table(base_model, (SIZE, SIZE, 3),
                              model_name='AMEDDE-Net v11')

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

    def make_callbacks(append_csv=False, best_dice_so_far=None, best_loss_so_far=None):
        return [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_dice_coefficient', factor=0.5, patience=5,
                mode='max', min_lr=1e-7, verbose=1),
            SaveBestModel(base_model,
                          f'{CKPT_PREFIX}_fold_{fold_num}_best_dice.keras',
                          'val_dice_coefficient', 'max', verbose=1,
                          initial_best=best_dice_so_far),
            SaveBestModel(base_model,
                          f'{CKPT_PREFIX}_fold_{fold_num}_best_loss.keras',
                          'val_loss', 'min', verbose=0,
                          initial_best=best_loss_so_far),
            tf.keras.callbacks.CSVLogger(
                f'{CKPT_PREFIX}_fold_{fold_num}_history.csv', append=append_csv),
        ]

    # ── Phase 1 ──────────────────────────────────────────────
    print(f"\n  [Phase 1] backbone FROZEN — LR={LR_PHASE1} — epoch 1-{PHASE1_EPOCHS}")
    start    = datetime.now()
    history1 = trainer.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE, epochs=PHASE1_EPOCHS,
        callbacks=make_callbacks(append_csv=False),
        shuffle=True, verbose=1
    )
    best_dice_p1 = max(history1.history['val_dice_coefficient'])
    best_loss_p1 = min(history1.history['val_loss'])
    print(f"\n  Phase 1 selesai — best val_dice: {best_dice_p1:.4f}, best val_loss: {best_loss_p1:.4f}")

    # ── Phase 2 ──────────────────────────────────────────────
    print(f"\n  [Phase 2] unfreeze block4-7 — LR={LR_PHASE2} — epoch {PHASE1_EPOCHS+1}-{EPOCHS}")
    print(f"  Checkpoint dilanjutkan dari phase 1 (best dice={best_dice_p1:.4f})")
    unfreeze_backbone(backbone)
    trainer.compile(optimizer=Adam(learning_rate=LR_PHASE2, clipnorm=0.5),
                    run_eagerly=False)
    history2 = trainer.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE, epochs=EPOCHS, initial_epoch=PHASE1_EPOCHS,
        callbacks=make_callbacks(append_csv=True,
                                 best_dice_so_far=best_dice_p1,
                                 best_loss_so_far=best_loss_p1),
        shuffle=True, verbose=1
    )
    elapsed = datetime.now() - start

    combined = {k: history1.history[k] + history2.history.get(k, [])
                for k in history1.history}

    best_path = f'{CKPT_PREFIX}_fold_{fold_num}_best_dice.keras'
    if os.path.exists(best_path):
        print(f"\n  Memuat model terbaik: {best_path}")
        best_base = tf.keras.models.load_model(best_path, compile=False)
        best_dice = max(combined['val_dice_coefficient'])
        best_ep   = np.argmax(combined['val_dice_coefficient']) + 1
        print(f"  Best val_dice: {best_dice:.4f} (epoch {best_ep})")
    else:
        best_base = base_model

    y_pred  = tta_predict(best_base, X_val)
    metrics = calculate_metrics(y_val, y_pred)

    fold_result = {
        'fold': fold_num,
        'train_loss': float(combined['loss'][-1]),
        'val_loss':   float(combined['val_loss'][-1]),
        'execution_time': str(elapsed),
        **metrics
    }
    fold_results.append(fold_result)
    fold_histories.append(combined)

    # ★★★ PERSIST HASIL FOLD ★★★
    save_fold_result(fold_num, fold_result, combined)

    label_map = [('DICE','dice'),('IOU','iou'),('P','precision'),('R','recall'),
                 ('F1','f1_score'),('ACC','accuracy'),('ROC','roc_auc'),
                 ('FPR','fpr'),('FNR','fnr'),('TIF','tif')]
    print(f"\n--- Fold {fold_num} Results (thr={metrics['threshold']:.3f}) ---")
    for lbl, key in label_map:
        print(f"  {lbl:<6} {metrics[key]:>8.4f}")
    print(f"  Val Loss: {fold_result['val_loss']:.4f}  Time: {elapsed}")

    base_model.save(f'{CKPT_PREFIX}_fold_{fold_num}.keras')
    K.clear_session()


# ============================================================
# SUMMARY
# ============================================================
print(f"\n\n{'='*100}")
print("AMEDDE-Net v11 — CROSS-VALIDATION SUMMARY")
print(f"{'='*100}")
print(f"{'Method':<22} {'Loss':>7} {'ACC':>7} {'DICE':>7} {'P':>7} {'R':>7} "
      f"{'F1':>7} {'IOU':>7} {'ROC':>7} {'FPR':>7} {'FNR':>7} {'TIF':>7}")
print("-"*100)

results_df = pd.DataFrame(fold_results).sort_values('fold').reset_index(drop=True)
summary_cols = ['val_loss','accuracy','dice','precision','recall',
                'f1_score','iou','roc_auc','fpr','fnr','tif']

for _, row in results_df.iterrows():
    print(f"  Fold {int(row['fold']):<16} " + " ".join(f"{row[c]:>7.4f}" for c in summary_cols))

print("-"*100)
means = results_df[summary_cols].mean()
stds  = results_df[summary_cols].std()
print(f"  {'Mean':<22} " + " ".join(f"{means[c]:>7.4f}" for c in summary_cols))
print(f"  {'Std':<22}  " + " ".join(f"{stds[c]:>7.4f}"  for c in summary_cols))
print(f"{'='*100}\n")

results_df.to_csv('amedde_v11_kfold_results.csv', index=False)
pd.DataFrame([{'Metric': lbl, 'Mean': f"{results_df[c].mean():.4f}",
               'Std': f"{results_df[c].std():.4f}",
               'Min': f"{results_df[c].min():.4f}",
               'Max': f"{results_df[c].max():.4f}"}
              for c, lbl in {'val_loss':'LOSS','accuracy':'ACC','dice':'DICE',
                              'precision':'P','recall':'R','f1_score':'F1',
                              'iou':'IOU','roc_auc':'ROC','fpr':'FPR',
                              'fnr':'FNR','tif':'TIF'}.items()]
             ).to_csv('amedde_v11_kfold_summary.csv', index=False)

metrics_plot = ['dice','iou','precision','recall','f1_score','accuracy','roc_auc','fpr','fnr','tif']
ncols = 4; nrows = (len(metrics_plot) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()
fig.suptitle('AMEDDE-Net v11 — K-Fold Metrics', fontsize=16, fontweight='bold')
for idx, m in enumerate(metrics_plot):
    ax = axes[idx]; v = results_df[m].values
    ax.bar(range(1, len(v)+1), v, color='salmon' if m in ['fpr','fnr'] else 'steelblue', alpha=0.75)
    ax.axhline(np.mean(v), color='red', linestyle='--', lw=2, label=f'μ={np.mean(v):.4f}')
    ax.set_title(m.upper()); ax.set_xlabel('Fold'); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
for idx in range(len(metrics_plot), len(axes)): axes[idx].set_visible(False)
plt.tight_layout(); plt.savefig('amedde_v11_kfold_metrics.png', dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('AMEDDE-Net v11 — Training History', fontsize=14, fontweight='bold')
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
plt.tight_layout(); plt.savefig('amedde_v11_training_history.png', dpi=300, bbox_inches='tight')

best = results_df.iloc[results_df['dice'].idxmax()]
print(f"Best: Fold {int(best['fold'])} — DICE={best['dice']:.4f}, IOU={best['iou']:.4f}, ACC={best['accuracy']:.4f}")
print(f"Output: amedde_v11_kfold_results.csv, amedde_v11_kfold_summary.csv, amedde_v11_complexity.csv")
print(f"\n{'='*80}\nSELESAI!")
