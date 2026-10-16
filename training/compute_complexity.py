"""
Parameter count and FLOPs of DMSA-KANet (Table 9).

FLOPs are counted with the TensorFlow profiler (tf.compat.v1.profiler, 'float_ops') on the
traced inference graph of one forward pass at batch size 1 and 256x256x3 input. The profiler
counts one multiply-accumulate as two floating-point operations. Weights do not affect the
count, so the encoder is built without ImageNet weights here.

Run from the folder that contains train_dmsa_kanet_patientlevel.py:
    CUDA_VISIBLE_DEVICES="" python compute_complexity.py
"""
import ast
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf

SCRIPT = os.environ.get("DMSA_SCRIPT", "train_dmsa_kanet_patientlevel.py")
src = open(SCRIPT).read().replace("weights='imagenet'", "weights=None")
tree = ast.parse(src)
keep = [n for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
        or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets))]
ns = {'__name__': 'complexity'}
exec(compile(ast.Module(body=keep, type_ignores=[]), SCRIPT, 'exec'), ns)

model, backbone = ns['build_amedde_net_v9']((256, 256, 3), 1)

def count(m):
    total = m.count_params()
    trainable = sum(int(tf.size(w)) for w in m.trainable_weights)
    return total, trainable, total - trainable

def profiler_flops(m, outputs=None):
    sub = m if outputs is None else tf.keras.Model(m.inputs, outputs)
    fn = tf.function(lambda x: sub(x, training=False))
    cf = fn.get_concrete_function(tf.TensorSpec([1, 256, 256, 3], tf.float32))
    graph = cf.graph   # variables are read, not folded into constants (saves memory; same op count)
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts['output'] = 'none'
    info = tf.compat.v1.profiler.profile(graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd='scope', options=opts)
    return info.total_float_ops

# Phase 1 state (backbone frozen), as in Table 9
tp, trp, ntp = count(model)
print(f"TensorFlow {tf.__version__}  | script: {SCRIPT}")
print(f"Total parameters      : {tp:,} ({tp/1e6:.2f} M)")
print(f"Trainable (phase 1)   : {trp:,} ({trp/1e6:.2f} M)")
print(f"Non-trainable         : {ntp:,} ({ntp/1e6:.2f} M)")
ns['unfreeze_backbone'](backbone)
tp2, trp2, ntp2 = count(model)
print(f"Trainable (phase 2)   : {trp2:,} ({trp2/1e6:.2f} M)")

which = os.environ.get('OUTPUTS', 'all')
f = profiler_flops(model) if which == 'all' else profiler_flops(model, model.outputs[0])
label = 'all four outputs (deep-supervision heads included)' if which == 'all' else 'main output only'
print(f"FLOPs, {label}: {f:,} ({f/1e9:.2f} G)")
print("Convention: TF profiler float_ops, batch 1, 256x256x3, one multiply-accumulate = 2 FLOPs.")
