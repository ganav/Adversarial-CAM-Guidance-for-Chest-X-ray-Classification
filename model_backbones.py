# model_backbones.py
from __future__ import print_function

import keras
import tensorflow as tf
from keras import layers, models
from keras import backend as K

# These should exist in Keras 2.1.6, DenseNet201 might or might not exist
try:
    from keras.applications import ResNet50, DenseNet121, DenseNet201, MobileNet
except ImportError:
    from keras.applications.densenet import DenseNet121
    try:
        from keras.applications.densenet import DenseNet201
    except ImportError:
        DenseNet201 = None
    from keras.applications.mobilenet import MobileNet
    try:
        from keras.applications.resnet50 import ResNet50
    except ImportError:
        ResNet50 = None


# ---------------------------------------------------------------------------
# Helpers: MBConv-style block for EfficientNetB0-like backbone
# ---------------------------------------------------------------------------

def mbconv_block(x,
                 in_channels,
                 out_channels,
                 expansion=1,
                 kernel_size=3,
                 stride=1,
                 se_ratio=0.25,
                 block_name="mbconv"):
    shortcut = x
    bn_axis = 3  # channels_last

    # Expand
    hidden_dim = in_channels * expansion
    if expansion != 1:
        x = layers.Conv2D(hidden_dim, 1, padding="same",
                          use_bias=False, name=block_name + "_expand_conv")(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      name=block_name + "_expand_bn")(x)
        x = layers.Activation("relu", name=block_name + "_expand_relu")(x)
    else:
        hidden_dim = in_channels

    # Depthwise conv
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding="same",
                               use_bias=False, name=block_name + "_dwconv")(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  name=block_name + "_dw_bn")(x)
    x = layers.Activation("relu", name=block_name + "_dw_relu")(x)

    # Squeeze-and-Excitation
    if se_ratio is not None and 0 < se_ratio <= 1.0:
        se_channels = max(1, int(in_channels * se_ratio))
        se = layers.GlobalAveragePooling2D(name=block_name + "_se_squeeze")(x)
        se = layers.Reshape((1, 1, hidden_dim),
                            name=block_name + "_se_reshape")(se)
        se = layers.Conv2D(se_channels, 1, activation="relu",
                           name=block_name + "_se_reduce")(se)
        se = layers.Conv2D(hidden_dim, 1, activation="sigmoid",
                           name=block_name + "_se_expand")(se)
        x = layers.Multiply(name=block_name + "_se_excite")([x, se])

    # Project
    x = layers.Conv2D(out_channels, 1, padding="same",
                      use_bias=False, name=block_name + "_project_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  name=block_name + "_project_bn")(x)

    # Residual
    if stride == 1 and in_channels == out_channels:
        x = layers.Add(name=block_name + "_add")([shortcut, x])

    return x


def build_efficientnetb0_like_backbone(input_shape=(224, 224, 3)):

    img_input = keras.Input(shape=input_shape, name="effb0_like_input")
    bn_axis = 3

    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding="same",
                      use_bias=False, name="stem_conv")(img_input)
    x = layers.BatchNormalization(axis=bn_axis, name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    stages = [
        # exp, out, blocks, k, s
        (1, 16, 1, 3, 1),
        (6, 24, 2, 3, 2),
        (6, 40, 2, 5, 2),
        (6, 80, 3, 3, 2),
        (6, 112, 3, 5, 1),
        (6, 192, 4, 5, 2),
        (6, 320, 1, 3, 1),
    ]

    in_channels = 32
    block_id = 0
    for (exp, out_ch, num_blocks, k, s) in stages:
        for i in range(num_blocks):
            block_stride = s if i == 0 else 1
            x = mbconv_block(
                x,
                in_channels=in_channels,
                out_channels=out_ch,
                expansion=exp,
                kernel_size=k,
                stride=block_stride,
                se_ratio=0.25,
                block_name="b{}_{}".format(block_id, i),
            )
            in_channels = out_ch
        block_id += 1

    backbone = models.Model(img_input, x, name="efficientnetb0_like_backbone")
    return backbone


# ---------------------------------------------------------------------------
# Paper 1 (Scientific Reports 2024): FE module (GhostConv + split 0.2 + residual)
# and MF module (SPP + Ghost Dilated Conv)
# ---------------------------------------------------------------------------

def _channel_slice(x, c_start, c_end, name):
    return layers.Lambda(
        lambda t: t[:, :, :, c_start:c_end],
        name=name
    )(x)


def ghost_conv2d(x,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 ratio=2,
                 dilation_rate=1,
                 activation="relu",
                 name="ghost"):
    """
    Simplified Ghost Conv:
      - primary conv to get a smaller set of intrinsic features
      - cheap depthwise (optionally dilated) to generate more features
      - concat, then trim to out_channels
    """
    bn_axis = 3

    # primary channels
    primary_channels = int((out_channels + ratio - 1) / ratio)
    cheap_channels = out_channels - primary_channels
    if cheap_channels < 0:
        cheap_channels = 0

    x1 = layers.Conv2D(primary_channels,
                       kernel_size,
                       strides=stride,
                       padding="same",
                       use_bias=False,
                       dilation_rate=dilation_rate,
                       name=name + "_primary_conv")(x)
    x1 = layers.BatchNormalization(axis=bn_axis, name=name + "_primary_bn")(x1)
    if activation:
        x1 = layers.Activation(activation, name=name + "_primary_act")(x1)

    if cheap_channels == 0:
        # already enough
        return x1

    x2 = layers.DepthwiseConv2D(3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               dilation_rate=dilation_rate,
                               name=name + "_cheap_dw")(x1)
    x2 = layers.BatchNormalization(axis=bn_axis, name=name + "_cheap_bn")(x2)
    if activation:
        x2 = layers.Activation(activation, name=name + "_cheap_act")(x2)

    x_cat = layers.Concatenate(axis=bn_axis, name=name + "_concat")([x1, x2])

    # trim to out_channels
    x_out = layers.Lambda(
        lambda t: t[:, :, :, :out_channels],
        name=name + "_trim"
    )(x_cat)

    return x_out


def fe_module(x,
              out_channels,
              split_ratio=0.2,
              name="fe"):
    """
    FE module from the paper (conceptual):
      1x1 conv -> split channels (0.2 bypass, 0.8 ghost conv) -> concat
      -> 1x1 conv -> residual add
    """
    bn_axis = 3
    in_ch = K.int_shape(x)[-1]
    if in_ch is None:
        raise ValueError("FE module needs known input channels")

    # project input to out_channels
    proj = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                         name=name + "_proj_in")(x)
    proj = layers.BatchNormalization(axis=bn_axis, name=name + "_proj_in_bn")(proj)
    proj = layers.Activation("relu", name=name + "_proj_in_act")(proj)

    bypass_ch = int(out_channels * split_ratio)
    proc_ch = out_channels - bypass_ch
    if bypass_ch < 1:
        bypass_ch = 1
        proc_ch = out_channels - 1
    if proc_ch < 1:
        proc_ch = 1
        bypass_ch = out_channels - 1

    bypass = _channel_slice(proj, 0, bypass_ch, name=name + "_bypass")
    proc = _channel_slice(proj, bypass_ch, out_channels, name=name + "_proc_in")

    proc = ghost_conv2d(proc,
                        out_channels=proc_ch,
                        kernel_size=3,
                        stride=1,
                        ratio=2,
                        dilation_rate=1,
                        activation="relu",
                        name=name + "_ghost")

    merged = layers.Concatenate(axis=bn_axis, name=name + "_merge")([bypass, proc])

    # map back to out_channels
    merged = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                           name=name + "_proj_out")(merged)
    merged = layers.BatchNormalization(axis=bn_axis, name=name + "_proj_out_bn")(merged)

    # residual, match shortcut channels if needed
    if in_ch != out_channels:
        shortcut = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                                 name=name + "_shortcut")(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, name=name + "_shortcut_bn")(shortcut)
    else:
        shortcut = x

    out = layers.Add(name=name + "_add")([shortcut, merged])
    out = layers.Activation("relu", name=name + "_out_act")(out)
    return out


def mf_module(x,
              out_channels,
              pool_sizes=(3, 5, 7),
              dilation_rates=(1, 2, 3),
              name="mf"):
    """
    MF module (conceptual):
      SPP-style multi maxpool scales, then Ghost Dilated Conv per branch, concat, 1x1 fuse.
    The paper states multiple maxpool sizes and GDConv, but does not list exact sizes in the accessible text,
    so these defaults are editable.
    """
    bn_axis = 3

    branches = []

    # identity branch
    branches.append(x)

    for i, ps in enumerate(pool_sizes):
        p = layers.MaxPooling2D(pool_size=(ps, ps),
                                strides=(1, 1),
                                padding="same",
                                name=name + "_pool{}".format(ps))(x)

        # apply a small hierarchy of GDConv-like ops (GhostConv with dilation)
        b = p
        for j, dr in enumerate(dilation_rates):
            b = ghost_conv2d(b,
                             out_channels=out_channels,
                             kernel_size=3,
                             stride=1,
                             ratio=2,
                             dilation_rate=dr,
                             activation="relu",
                             name=name + "_gdconv{}_d{}".format(i, dr))
        branches.append(b)

    y = layers.Concatenate(axis=bn_axis, name=name + "_concat")(branches)

    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False, name=name + "_fuse")(y)
    y = layers.BatchNormalization(axis=bn_axis, name=name + "_fuse_bn")(y)
    y = layers.Activation("relu", name=name + "_fuse_act")(y)
    return y


def build_lightweight_fe_mf_backbone(input_shape=(224, 224, 3),
                                    width_mult=1.0):
    """
    Paper 1 style backbone, built from FE and MF modules.
    This is a reasonable Keras recreation using the paper's described modules.
    """
    img_input = keras.Input(shape=input_shape, name="lw_femf_input")
    bn_axis = 3

    def c(ch):
        return max(8, int(ch * width_mult))

    x = layers.Conv2D(c(16), 3, strides=2, padding="same", use_bias=False, name="lw_stem_conv")(img_input)
    x = layers.BatchNormalization(axis=bn_axis, name="lw_stem_bn")(x)
    x = layers.Activation("relu", name="lw_stem_relu")(x)

    x = fe_module(x, out_channels=c(24), split_ratio=0.2, name="lw_fe1")
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="lw_pool1")(x)

    x = fe_module(x, out_channels=c(48), split_ratio=0.2, name="lw_fe2")
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="lw_pool2")(x)

    x = fe_module(x, out_channels=c(96), split_ratio=0.2, name="lw_fe3")

    # MF at the end for multiscale context
    x = mf_module(x,
                  out_channels=c(128),
                  pool_sizes=(3, 5, 7),
                  dilation_rates=(1, 2, 3),
                  name="lw_mf")

    backbone = models.Model(img_input, x, name="lightweight_fe_mf_backbone")
    return backbone


# ---------------------------------------------------------------------------
# Paper 2 (BMC Med Imaging 2025): DenseNet201 backbone + Transformer branch + GAP branch
# ---------------------------------------------------------------------------

class LayerNorm(layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma",
                                     shape=(dim,),
                                     initializer="ones",
                                     trainable=True)
        self.beta = self.add_weight(name="beta",
                                    shape=(dim,),
                                    initializer="zeros",
                                    trainable=True)
        super(LayerNorm, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        var = K.mean(K.square(x - mean), axis=-1, keepdims=True)
        x_norm = (x - mean) / K.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


def _global_avg_pool_1d(x, name):
    return layers.Lambda(lambda t: K.mean(t, axis=1), name=name)(x)


# ---- FIXED MHA + Transformer (TF1.9 / Keras 2.1.6 safe) ----

def mha_block(x, num_heads, head_dim, name):
    """
    Self-attention that NEVER reshapes with the wrong size.
    Uses embed_dim = channels of x (not num_heads*head_dim).
    """
    import tensorflow as tf

    embed_dim = K.int_shape(x)[-1]
    if embed_dim is None:
        raise ValueError("mha_block needs known last-dim (embed_dim)")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads (got %s vs %s)" % (embed_dim, num_heads))

    head_dim = embed_dim // num_heads  # <- override, always correct

    q = layers.Dense(embed_dim, use_bias=False, name=name + "_q")(x)
    k = layers.Dense(embed_dim, use_bias=False, name=name + "_k")(x)
    v = layers.Dense(embed_dim, use_bias=False, name=name + "_v")(x)

    def _reshape_heads(t):
        b = tf.shape(t)[0]
        s = tf.shape(t)[1]
        t = tf.reshape(t, [b, s, num_heads, head_dim])
        t = tf.transpose(t, [0, 2, 1, 3])  # (B,H,S,D)
        return t

    qh = layers.Lambda(_reshape_heads, name=name + "_q_reshape")(q)
    kh = layers.Lambda(_reshape_heads, name=name + "_k_reshape")(k)
    vh = layers.Lambda(_reshape_heads, name=name + "_v_reshape")(v)

    def _attn(inputs):
        q_, k_, v_ = inputs
        scale = tf.sqrt(tf.cast(head_dim, tf.float32))
        scores = tf.matmul(q_, k_, transpose_b=True) / scale  # (B,H,S,S)
        w = tf.nn.softmax(scores)  # TF1.9 safe (last axis)
        out = tf.matmul(w, v_)  # (B,H,S,D)
        out = tf.transpose(out, [0, 2, 1, 3])  # (B,S,H,D)
        out = tf.reshape(out, [tf.shape(out)[0], tf.shape(out)[1], embed_dim])  # (B,S,C)
        return out

    attn_out = layers.Lambda(_attn, name=name + "_attn")([qh, kh, vh])

    # project back to embed_dim
    attn_out = layers.Dense(embed_dim, use_bias=False, name=name + "_proj")(attn_out)
    return attn_out


def transformer_encoder(x,
                        num_layers=2,
                        num_heads=4,
                        head_dim=32,   # ignored safely (we recompute from embed_dim)
                        mlp_ratio=2,
                        name="tx"):

    embed_dim = K.int_shape(x)[-1]
    if embed_dim is None:
        raise ValueError("Transformer encoder needs known last-dim (embed dim)")

    for i in range(num_layers):
        x1 = LayerNorm(name=name + "_ln1_{}".format(i))(x)
        a = mha_block(x1, num_heads=num_heads, head_dim=head_dim, name=name + "_mha_{}".format(i))
        x = layers.Add(name=name + "_attn_add_{}".format(i))([x, a])

        x2 = LayerNorm(name=name + "_ln2_{}".format(i))(x)
        hidden = embed_dim * mlp_ratio
        m = layers.Dense(hidden, activation="relu", name=name + "_mlp_fc1_{}".format(i))(x2)
        m = layers.Dense(embed_dim, name=name + "_mlp_fc2_{}".format(i))(m)
        x = layers.Add(name=name + "_mlp_add_{}".format(i))([x, m])

    return x


# ---- FIXED DenseNet201 + Transformer + GAP hybrid ----

def build_densenet201_vit_hybrid_models(num_classes,
                                       input_shape=(299, 299, 1),
                                       densenet_weights=None,
                                       tx_layers=2,
                                       tx_heads=4,
                                       tx_head_dim=32,
                                       mlp_out_dim=256):

    if DenseNet201 is None:
        raise ImportError("DenseNet201 is not available in this Keras version. "
                          "Option 1: upgrade Keras, option 2: swap to DenseNet121 manually.")

    img_input = keras.Input(shape=input_shape, name="dn201_vit_input")

    # DenseNet wants 3ch, so if grayscale -> repeat channels
    x_in = img_input
    if input_shape[-1] == 1:
        x_in = layers.Concatenate(axis=-1, name="dn201_gray_to_rgb")([x_in, x_in, x_in])

    dnet = DenseNet201(include_top=False,
                       weights=densenet_weights,
                       input_tensor=x_in,
                       pooling=None)

    fmap = dnet.output  # (B,H,W,C)
    c = K.int_shape(fmap)[-1]
    h = K.int_shape(fmap)[1]
    w = K.int_shape(fmap)[2]
    if c is None or h is None or w is None:
        raise ValueError("DenseNet201 output shape must be static here")

    S = h * w
    embed_dim = tx_heads * tx_head_dim  # e.g. 128 (stable small)

    # tokens: (B,S,C)
    tokens = layers.Reshape((S, c), name="dn201_tokens")(fmap)

    # project tokens down to embed_dim so attention is stable + light
    tokens = layers.Dense(embed_dim, use_bias=False, name="dn201_tok_proj")(tokens)  # (B,S,embed_dim)

    # pos embedding: (S,embed_dim) -> (1,S,embed_dim) -> tile to (B,S,embed_dim)
    positions = layers.Lambda(lambda _: tf.range(S), name="dn201_positions")(tokens)  # (S,)
    pos_emb = layers.Embedding(input_dim=S, output_dim=embed_dim, name="dn201_pos_emb")(positions)  # (S,E)
    pos_emb = layers.Lambda(lambda t: K.expand_dims(t, axis=0), name="dn201_pos_expand")(pos_emb)   # (1,S,E)
    pos_emb = layers.Lambda(
        lambda xs: tf.tile(xs[0], [tf.shape(xs[1])[0], 1, 1]),
        name="dn201_pos_tile"
    )([pos_emb, tokens])  # (B,S,E)

    tokens = layers.Add(name="dn201_add_pos")([tokens, pos_emb])  # (B,S,E)

    # transformer (now safe)
    tokens = transformer_encoder(tokens,
                                 num_layers=tx_layers,
                                 num_heads=tx_heads,
                                 head_dim=tx_head_dim,
                                 mlp_ratio=2,
                                 name="dn201_tx")

    # token -> vector (B,E) -> (B,256)
    tx_vec = layers.GlobalAveragePooling1D(name="dn201_tx_gap1d")(tokens)
    tx_vec = layers.Dense(mlp_out_dim, activation="relu", name="dn201_tx_mlp")(tx_vec)

    # fmap -> vector (B,C)
    gap_vec = layers.GlobalAveragePooling2D(name="dn201_fmap_gap2d")(fmap)

    fused = layers.Concatenate(name="dn201_fuse")([tx_vec, gap_vec])
    pred = layers.Dense(num_classes, activation="softmax", name="dn201_vit_pred")(fused)

    class_model = models.Model(img_input, pred, name="densenet201_vit_hybrid_classifier")
    backbone_model = models.Model(img_input, fmap, name="densenet201_backbone_for_cam")
    return class_model, backbone_model


# ---- FIXED ViT backbone that works with 299 too (pads instead of raising) ----

def build_vit_backbone(input_shape=(224, 224, 3),
                       patch_size=16,
                       embed_dim=192,
                       tx_layers=4,
                       tx_heads=3,
                       tx_head_dim=64):

    h = input_shape[0]
    w = input_shape[1]

    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    h2, w2 = h + pad_h, w + pad_w

    grid_h = int(h2 / patch_size)
    grid_w = int(w2 / patch_size)
    seq_len = grid_h * grid_w

    img_input = keras.Input(shape=input_shape, name="vit_input")
    x = img_input
    if pad_h or pad_w:
        x = layers.ZeroPadding2D(((0, pad_h), (0, pad_w)), name="vit_pad")(x)

    x = layers.Conv2D(embed_dim,
                      kernel_size=patch_size,
                      strides=patch_size,
                      padding="valid",
                      use_bias=False,
                      name="vit_patch_conv")(x)

    tokens = layers.Reshape((seq_len, embed_dim), name="vit_tokens")(x)

    positions = layers.Lambda(lambda _: tf.range(seq_len), name="vit_positions")(tokens)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=embed_dim, name="vit_pos_emb")(positions)  # (S,E)
    pos_emb = layers.Lambda(lambda t: K.expand_dims(t, axis=0), name="vit_pos_expand")(pos_emb)          # (1,S,E)
    pos_emb = layers.Lambda(lambda xs: tf.tile(xs[0], [tf.shape(xs[1])[0], 1, 1]),
                            name="vit_pos_tile")([pos_emb, tokens])                                     # (B,S,E)
    tokens = layers.Add(name="vit_add_pos")([tokens, pos_emb])

    tokens = transformer_encoder(tokens,
                                 num_layers=tx_layers,
                                 num_heads=tx_heads,
                                 head_dim=tx_head_dim,
                                 mlp_ratio=2,
                                 name="vit_tx")

    fmap = layers.Reshape((grid_h, grid_w, embed_dim), name="vit_fmap")(tokens)

    backbone = models.Model(img_input, fmap, name="vit_backbone")
    return backbone
    

def build_gaze_guided_vit_backbone(input_shape=(224, 224, 3),
                                  patch_size=16,
                                  embed_dim=192,
                                  tx_layers=4,
                                  tx_heads=3,
                                  tx_head_dim=64,
                                  use_external_gaze=False):
    """
    Gaze-guided ViT-like backbone hook:
      If use_external_gaze is False, learn a spatial gate from patch embeddings (sigmoid map),
      If True, expects a second input (gaze map at patch grid resolution) to gate tokens.

    Note: This is a practical hook, because the full paper architecture details are not accessible from public text.
    """
    h = input_shape[0]
    w = input_shape[1]
    if (h % patch_size) != 0 or (w % patch_size) != 0:
        raise ValueError("input_shape must be divisible by patch_size")

    grid_h = int(h / patch_size)
    grid_w = int(w / patch_size)
    seq_len = grid_h * grid_w

    img_input = keras.Input(shape=input_shape, name="gvit_input")

    # patch embedding
    x = layers.Conv2D(embed_dim,
                      kernel_size=patch_size,
                      strides=patch_size,
                      padding="valid",
                      use_bias=False,
                      name="gvit_patch_conv")(img_input)  # (B, gh, gw, embed_dim)

    if use_external_gaze:
        gaze_input = keras.Input(shape=(grid_h, grid_w, 1), name="gvit_gaze_map")
        gate = gaze_input
    else:
        # learn a gate from the patch grid
        gate = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="gvit_gate_conv")(x)

    gate_tokens = layers.Reshape((seq_len, 1), name="gvit_gate_tokens")(gate)

    tokens = layers.Reshape((seq_len, embed_dim), name="gvit_tokens")(x)
    tokens = layers.Multiply(name="gvit_apply_gate")([tokens, gate_tokens])

    # positional embedding
    pos_w = layers.Embedding(input_dim=seq_len, output_dim=embed_dim, name="gvit_pos_emb")
    positions = layers.Lambda(lambda t: K.arange(0, seq_len), name="gvit_positions")(tokens)
    pos_emb = pos_w(positions)
    pos_emb = layers.Lambda(lambda t: K.expand_dims(t, axis=0), name="gvit_pos_expand")(pos_emb)
    tokens = layers.Add(name="gvit_add_pos")([tokens, pos_emb])

    tokens = transformer_encoder(tokens,
                                 num_layers=tx_layers,
                                 num_heads=tx_heads,
                                 head_dim=tx_head_dim,
                                 mlp_ratio=2,
                                 name="gvit_tx")

    fmap = layers.Reshape((grid_h, grid_w, embed_dim), name="gvit_fmap")(tokens)

    if use_external_gaze:
        backbone = models.Model([img_input, gaze_input], fmap, name="gaze_guided_vit_backbone")
    else:
        backbone = models.Model(img_input, fmap, name="gaze_guided_vit_backbone")

    return backbone


# ---------------------------------------------------------------------------
# Existing Keras.applications backbones
# ---------------------------------------------------------------------------

def build_densenet121_backbone(input_shape=(224, 224, 3)):
    img_input = keras.Input(shape=input_shape, name="dn121_input")
    x = img_input
    if input_shape[-1] == 1:
        x = layers.Concatenate(axis=-1, name="dn121_gray_to_rgb")([x, x, x])

    base = DenseNet121(
        include_top=False,
        weights=None,
        input_tensor=x,
        pooling=None,
    )
    return models.Model(img_input, base.output, name="densenet121_backbone")


def build_mobilenet_backbone(input_shape=(224, 224, 3)):
    img_input = keras.Input(shape=input_shape, name="mb1_input")
    x = img_input
    if input_shape[-1] == 1:
        x = layers.Concatenate(axis=-1, name="mb1_gray_to_rgb")([x, x, x])

    base = MobileNet(
        include_top=False,
        weights=None,
        input_tensor=x,
        pooling=None,
    )
    return models.Model(img_input, base.output, name="mobilenet_backbone")


# ---------------------------------------------------------------------------
# Unified builder: backbone + GAP + softmax for classification (CAM-ready)
# For special multi-branch models, returns their custom classifier and a CAM backbone.
# ---------------------------------------------------------------------------

def build_backbone(
    backbone_name,
    num_classes,
    input_shape=(224, 224, 3),
):

    name = backbone_name.lower()

    # Standard CNN backbones
    if name == "densenet121":
        base = build_densenet121_backbone(input_shape)
        backbone_name_clean = "densenet121"

        Fmap = base.output
        gap = layers.GlobalAveragePooling2D(name=backbone_name_clean + "_gap")(Fmap)
        class_out = layers.Dense(num_classes, activation="softmax",
                                 name=backbone_name_clean + "_pred")(gap)

        class_model = models.Model(base.input, class_out, name=backbone_name_clean + "_classifier")
        backbone_model = models.Model(base.input, Fmap, name=backbone_name_clean + "_backbone")
        return class_model, backbone_model

    elif name == "mobilenet":
        base = build_mobilenet_backbone(input_shape)
        backbone_name_clean = "mobilenet"

        Fmap = base.output
        gap = layers.GlobalAveragePooling2D(name=backbone_name_clean + "_gap")(Fmap)
        class_out = layers.Dense(num_classes, activation="softmax",
                                 name=backbone_name_clean + "_pred")(gap)

        class_model = models.Model(base.input, class_out, name=backbone_name_clean + "_classifier")
        backbone_model = models.Model(base.input, Fmap, name=backbone_name_clean + "_backbone")
        return class_model, backbone_model

    elif name in ("efficientnetb0", "efficientnetb0_like", "effb0"):
        base = build_efficientnetb0_like_backbone(input_shape)
        backbone_name_clean = "efficientnetb0_like"

        Fmap = base.output
        gap = layers.GlobalAveragePooling2D(name=backbone_name_clean + "_gap")(Fmap)
        class_out = layers.Dense(num_classes, activation="softmax",
                                 name=backbone_name_clean + "_pred")(gap)

        class_model = models.Model(base.input, class_out, name=backbone_name_clean + "_classifier")
        backbone_model = models.Model(base.input, Fmap, name=backbone_name_clean + "_backbone")
        return class_model, backbone_model

    # Paper 1 backbone
    elif name in ("lightweight_fe_mf", "lw_femf", "yen_tsao_2024"):
        base = build_lightweight_fe_mf_backbone(input_shape=input_shape, width_mult=1.0)
        backbone_name_clean = "lightweight_fe_mf"

        Fmap = base.output
        gap = layers.GlobalAveragePooling2D(name=backbone_name_clean + "_gap")(Fmap)
        class_out = layers.Dense(num_classes, activation="softmax",
                                 name=backbone_name_clean + "_pred")(gap)

        class_model = models.Model(base.input, class_out, name=backbone_name_clean + "_classifier")
        backbone_model = models.Model(base.input, Fmap, name=backbone_name_clean + "_backbone")
        return class_model, backbone_model

    # Paper 2 special hybrid (DenseNet201 + Transformer + GAP fusion)
    elif name in ("densenet201_vit", "dn201_vit", "oltu_2025"):
        return build_densenet201_vit_hybrid_models(
            num_classes=num_classes,
            input_shape=input_shape,
            densenet_weights=None,
            tx_layers=2,
            tx_heads=4,
            tx_head_dim=32,
            mlp_out_dim=256,
        )

    # Paper 3 ViT
    elif name in ("vit", "vit_backbone"):
        base = build_vit_backbone(input_shape=input_shape)
        backbone_name_clean = "vit"

        Fmap = base.output
        gap = layers.GlobalAveragePooling2D(name=backbone_name_clean + "_gap")(Fmap)
        class_out = layers.Dense(num_classes, activation="softmax",
                                 name=backbone_name_clean + "_pred")(gap)

        class_model = models.Model(base.input, class_out, name=backbone_name_clean + "_classifier")
        backbone_model = models.Model(base.input, Fmap, name=backbone_name_clean + "_backbone")
        return class_model, backbone_model

    elif name in ("gaze_vit", "gaze_guided_vit", "chen_2026"):
        base = build_gaze_guided_vit_backbone(input_shape=input_shape, use_external_gaze=False)
        backbone_name_clean = "gaze_guided_vit"

        Fmap = base.output
        gap = layers.GlobalAveragePooling2D(name=backbone_name_clean + "_gap")(Fmap)
        class_out = layers.Dense(num_classes, activation="softmax",
                                 name=backbone_name_clean + "_pred")(gap)

        class_model = models.Model(base.input, class_out, name=backbone_name_clean + "_classifier")
        backbone_model = models.Model(base.input, Fmap, name=backbone_name_clean + "_backbone")
        return class_model, backbone_model

    else:
        raise ValueError("Unknown backbone_name: {}".format(backbone_name))


def set_trainable(model, trainable):
    for l in model.layers:
        l.trainable = trainable


def build_discriminator(input_shape=(299, 299, 1), base_filters=32, name="heatmap_discriminator"):
    """
    Discriminator: takes (H,W,1) map (mask or CAM heatmap) -> real/fake.
    No Lambda layers, so it is stable in Keras 2.1.6.
    """
    inp = keras.Input(shape=input_shape, name=name + "_in")
    x = inp

    x = layers.Conv2D(base_filters, 4, strides=2, padding="same", name=name + "_c1")(x)
    x = layers.LeakyReLU(alpha=0.2, name=name + "_l1")(x)

    x = layers.Conv2D(base_filters * 2, 4, strides=2, padding="same", name=name + "_c2")(x)
    x = layers.BatchNormalization(axis=3, name=name + "_bn2")(x)
    x = layers.LeakyReLU(alpha=0.2, name=name + "_l2")(x)

    x = layers.Conv2D(base_filters * 4, 4, strides=2, padding="same", name=name + "_c3")(x)
    x = layers.BatchNormalization(axis=3, name=name + "_bn3")(x)
    x = layers.LeakyReLU(alpha=0.2, name=name + "_l3")(x)

    x = layers.Conv2D(base_filters * 8, 4, strides=2, padding="same", name=name + "_c4")(x)
    x = layers.BatchNormalization(axis=3, name=name + "_bn4")(x)
    x = layers.LeakyReLU(alpha=0.2, name=name + "_l4")(x)

    x = layers.GlobalAveragePooling2D(name=name + "_gap")(x)
    out = layers.Dense(1, activation="sigmoid", name=name + "_out")(x)

    return models.Model(inp, out, name=name)


def build_cam_model(class_model, backbone_model, pred_layer_name, img_size=299, name="cam_model"):
    """
    Builds a model: X -> [pred, cam]
    cam is differentiable and normalized to [0,1], resized to (img_size,img_size,1).

    CAM formula here is a stable "soft-CAM":
      w_eff(B,C) = pred(B,K) @ W^T(K,C)
      cam(B,H,W) = sum_c fmap(B,H,W,c) * w_eff(B,c)
    """
    dense = class_model.get_layer(pred_layer_name)
    W = dense.kernel  # (C, num_classes)

    fmap = backbone_model.output          # (B,Hf,Wf,C)
    pred = class_model.output             # (B,num_classes)

    def _cam_fn(xs):
        fmap_, pred_ = xs
        # (B,C) = (B,K) x (K,C)
        w_eff = tf.matmul(pred_, tf.transpose(W))

        b = tf.shape(fmap_)[0]
        c = tf.shape(fmap_)[-1]
        w_eff = tf.reshape(w_eff, [b, 1, 1, c])

        cam = tf.reduce_sum(fmap_ * w_eff, axis=-1, keepdims=True)  # (B,Hf,Wf,1)
        cam = tf.nn.relu(cam)

        # normalize per-sample to [0,1]
        cam_min = tf.reduce_min(cam, axis=[1, 2, 3], keepdims=True)
        cam_max = tf.reduce_max(cam, axis=[1, 2, 3], keepdims=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam = tf.image.resize_bilinear(cam, [img_size, img_size])  # (B,img,img,1)
        return cam

    cam = layers.Lambda(_cam_fn, name=name + "_map")([fmap, pred])

    return models.Model(class_model.input, [pred, cam], name=name)


def build_proposed_models(backbone_name, num_classes, img_size=299, chan=1):
    """
    Returns:
      class_model: X -> pred
      backbone_model: X -> fmap
      cam_model: X -> [pred, cam]
      disc_model: map -> real/fake
      adv_model: X -> [pred, disc(cam)]   (compile this to train classifier with adversarial loss)
    """
    input_shape = (img_size, img_size, chan)

    class_model, backbone_model = build_backbone(backbone_name, num_classes, input_shape=input_shape)

    # pred layer names for your standard builders:
    # densenet121_pred, mobilenet_pred, efficientnetb0_like_pred, etc.
    # for vit: vit_pred, gaze_guided_vit_pred
    # for dn201_vit: dn201_vit_pred (CAM from fmap is not meaningful there, but still returns something)
    pred_layer_name = None
    for lname in ["densenet121_pred", "mobilenet_pred", "efficientnetb0_like_pred",
                  "lightweight_fe_mf_pred", "vit_pred", "gaze_guided_vit_pred", "dn201_vit_pred"]:
        try:
            class_model.get_layer(lname)
            pred_layer_name = lname
            break
        except:
            pass
    if pred_layer_name is None:
        # fallback: last layer
        pred_layer_name = class_model.layers[-1].name

    cam_model = build_cam_model(class_model, backbone_model, pred_layer_name, img_size=img_size,
                                name=backbone_name.lower() + "_cam")

    disc_model = build_discriminator(input_shape=(img_size, img_size, 1),
                                     base_filters=32,
                                     name="disc")

    # Build adversarial model (freeze D BEFORE building)
    set_trainable(disc_model, False)
    pred, cam = cam_model.output
    d_out = disc_model(cam)
    adv_model = models.Model(cam_model.input, [pred, d_out], name="adv_" + backbone_name.lower())
    # NOTE: when you want to train D, you will set_trainable(disc_model, True) and compile D.
    # When you want to train classifier with adv_model, set_trainable(disc_model, False) and compile adv_model.

    return class_model, backbone_model, cam_model, disc_model, adv_model