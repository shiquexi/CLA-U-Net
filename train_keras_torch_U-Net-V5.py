import sys

import os
import numpy as np
import json
import shutil

os.environ["KERAS_BACKEND"] = "torch"

import keras

import logging
logging.basicConfig(level=logging.INFO)

from typing import Dict, List, Callable, Tuple, Any
from keras import ops, layers
from keras.regularizers import l2
import datetime
import torch

from keras.layers import Layer, Conv2D, DepthwiseConv2D, GlobalMaxPooling2D
# from tornet.data.loader import get_dataloader
# from tornet.data.preprocess import get_shape
# from tornet.data.constants import ALL_VARIABLES

# from tornet.models.keras.losses import mae_loss

# from tornet.models.keras.cnn_baseline import build_model

# from tornet.metrics.keras import metrics as tfm

# from tornet.utils.general import make_exp_dir, make_callback_dirs

EXP_DIR=os.environ.get('EXP_DIR','.')
# DATA_ROOT=os.environ['TORNET_ROOT']
DATA_ROOT='/seu_share/home/230249662/Tornado/data-yuanshi'
# TORNET_ROOT
logging.info('TORNET_ROOT='+DATA_ROOT)


def get_dataloader(
    dataloader: str,
    data_root: str,
    years: list[int],
    data_type: str,
    batch_size: int,
    weights: Dict[str, float] = None,
    **kwargs,
):
    """Creates a dataloader for keras, torch or tensorflow

    Inputs:
    dataloader - str describing the dataloader to use. Valid options are: keras,
    tensorflow, tensorflow-tfds, torch and torch-tfds
    data_root - where data is located
    years - years to load btwn 2013 and 2022
    data_type - either train or test
    batch_size - size of batch
    weights - weights for different categories of sample

    weights is optional, if provided must be a dict of the form
      weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
    where wN,w0,w1,w2,wW are numeric weights assigned to random,
    ef0, ef1, ef2+ and warnings samples, respectively.

    Returns:
    tf.data.Dataset, torch.utils.data.DataLoader or tornet.data.keras.loader.KerasDataLoader
    """

    # Argument validation
    valid_dataloaders = ["tensorflow", "tensorflow-tfds", "torch", "torch-tfds", "keras"]

    # Convert string to lower case
    dataloader = dataloader.lower()
    assert dataloader in valid_dataloaders, f"dataloader must be in {valid_dataloaders}!"

    from_tfds = False
    if "tfds" in dataloader:
        from_tfds = True

    if "tensorflow" in dataloader:
        import tensorflow as tf
        from tornet.data.tf.loader import make_tf_loader
        ds = make_tf_loader(data_root,data_type,years,batch_size,weights,from_tfds=from_tfds,**kwargs)

        data_opts = tf.data.Options()
        data_opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds=ds.with_options(data_opts)

    elif "torch" in dataloader:
        from tornet.data.torch.loader import make_torch_loader
        ds = make_torch_loader(data_root,data_type,years,batch_size,weights,from_tfds=from_tfds,**kwargs)
    else:
        from tornet.data.keras.loader import KerasDataLoader
        ds = KerasDataLoader(data_root=data_root,
                             data_type=data_type,
                             years=years,
                             batch_size=batch_size,
                             weights=weights,**kwargs)

    return ds


# List all potential input variables
ALL_VARIABLES=['DBZ',
               'VEL',
               'KDP',
               'RHOHV',
               'ZDR',
               'WIDTH']

# Provides a typical min-max range for each variable (but not exact)
# Used for normalizing in a NN
CHANNEL_MIN_MAX = {
    'DBZ': [-20.,60.],
    'VEL': [-60.,60.],
    'KDP': [-2.,5.],
    'RHOHV': [0.2, 1.04],
    'ZDR': [-1.,8.],
    'WIDTH':[0.,9.]
}


def mae_loss( class_labels, logits, sample_weights=None ):
    """
    class_labels represents tensor of known binary classes (1,0)
    logits are output of final classification layer that has not yet been run through a sigmoid (from_logits)
    """
    y_true,y_pred=_prep(class_labels, logits)
    if sample_weights is not None:
        denom=ops.sum(sample_weights)
        return ops.sum( sample_weights*ops.absolute(y_true-y_pred) ) / denom
    else:
        return ops.mean( ops.absolute(y_true-y_pred) )

def _prep(class_labels, logits):
    y_true = ops.cast(class_labels, dtype=logits.dtype)
    y_pred = ops.sigmoid(logits) # p=1 means class label=1
    return y_true,y_pred


def count_training_classes(data_root, train_years):
    """Count binary targets using the same catalog split as the TorNet loader."""
    import pandas as pd

    catalog = pd.read_csv(
        os.path.join(data_root, "catalog.csv"),
        usecols=["type", "start_time", "ef_number"],
        parse_dates=["start_time"],
    )
    training = catalog["type"].eq("train") & catalog["start_time"].dt.year.isin(train_years)
    ef_number = catalog.loc[training, "ef_number"]
    if ef_number.isna().any():
        raise ValueError("Missing EF ratings in the training catalog. Set ldam_class_counts explicitly.")
    positives = int(ef_number.ge(0).sum())
    return [len(ef_number) - positives, positives]


@keras.saving.register_keras_serializable(package="cla_unet")
class BinaryLDAMLoss(keras.losses.Loss):
    """Apply a class-frequency margin to the true-class logit."""

    def __init__(self, class_counts, max_margin=0.5, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        counts = np.asarray(class_counts, dtype=float)
        if (counts.shape != (2,) or not np.isfinite(counts).all()
                or np.any(counts <= 0) or np.any(counts != np.floor(counts))):
            raise ValueError("ldam_class_counts must be [n_non_tornado, n_tornado], both positive integers.")
        if not np.isfinite(max_margin) or not np.isfinite(scale) or max_margin < 0 or scale <= 0:
            raise ValueError("ldam_max_margin must be nonnegative and ldam_scale must be positive.")
        self.class_counts = [int(count) for count in counts]
        self.max_margin = float(max_margin)
        self.scale = float(scale)
        self.negative_margin, self.positive_margin = (
            self.max_margin * (counts.min() / counts) ** 0.25
        ).tolist()

    def call(self, y_true, y_pred):
        labels = ops.cast(ops.reshape(y_true, (-1,)), y_pred.dtype)
        logits = ops.reshape(y_pred, (-1,))
        # For [0, z], reducing the class-0 score is equivalent to increasing z.
        logits = ops.where(
            labels > 0.5,
            logits - self.positive_margin,
            logits + self.negative_margin,
        )
        return ops.binary_crossentropy(labels, self.scale * logits, from_logits=True)

    def get_config(self):
        return {
            **super().get_config(),
            "class_counts": self.class_counts,
            "max_margin": self.max_margin,
            "scale": self.scale,
        }


@keras.saving.register_keras_serializable(package="cla_unet")
class LMFTrainingModel(keras.Model):
    """Train a single-logit detector with a weighted Focal + LDAM objective."""

    def __init__(self, network, lmf_config, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.lmf_config = dict(lmf_config)
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=lmf_config["focal_alpha"],
            gamma=lmf_config["focal_gamma"],
            from_logits=True,
            reduction="none",
        )
        self.ldam_loss = BinaryLDAMLoss(
            class_counts=lmf_config["ldam_class_counts"],
            max_margin=lmf_config["ldam_max_margin"],
            scale=lmf_config["ldam_scale"],
            reduction="none",
        )
        focal_weight = lmf_config["lmf_initial_focal_weight"]
        trainable = lmf_config["lmf_trainable_weights"]
        if not 0 < focal_weight < 1:
            raise ValueError("lmf_initial_focal_weight must be strictly between 0 and 1.")
        initial_logit = float(np.log(focal_weight / (1.0 - focal_weight)))
        regularizer = keras.regularizers.L2(lmf_config["lmf_mix_l2"]) if trainable else None
        self.mix_logit = self.add_weight(
            name="lmf_mix_logit",
            shape=(),
            initializer=keras.initializers.Constant(initial_logit),
            regularizer=regularizer,
            trainable=trainable,
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

    def mix_weights(self):
        return ops.sigmoid(self.mix_logit), ops.sigmoid(-self.mix_logit)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        labels = ops.reshape(y, (-1, 1))
        logits = ops.reshape(y_pred, (-1, 1))
        focal_weight, ldam_weight = self.mix_weights()
        loss = (
            focal_weight * self.focal_loss(labels, logits)
            + ldam_weight * self.ldam_loss(labels, logits)
        )
        if sample_weight is not None:
            loss = loss * ops.cast(ops.reshape(sample_weight, (-1,)), loss.dtype)
        return ops.mean(loss) + sum(ops.cast(term, loss.dtype) for term in self.losses)

    def get_config(self):
        return {
            **super().get_config(),
            "network": keras.saving.serialize_keras_object(self.network),
            "lmf_config": self.lmf_config,
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        network = keras.saving.deserialize_keras_object(config.pop("network"))
        return cls(network=network, **config)


def attention_gate(skip, decoder, coordinates, filters, name):
    """Apply coordinate-aware additive attention to an encoder skip feature."""
    skip_projection = Conv2D(filters, 1, use_bias=False, name=f"{name}_skip")(skip)
    decoder_projection = Conv2D(filters, 1, use_bias=False, name=f"{name}_decoder")(decoder)
    coordinate_projection = Conv2D(filters, 1, name=f"{name}_coordinates")(coordinates)
    attention = layers.Add(name=f"{name}_sum")(
        [skip_projection, decoder_projection, coordinate_projection]
    )
    attention = layers.ReLU(name=f"{name}_relu")(attention)
    alpha = Conv2D(1, 1, activation="sigmoid", name=f"{name}_alpha")(attention)
    return layers.Multiply(name=f"{name}_gated_skip")([skip, alpha])


class DWConv(Layer):
    """Depthwise Convolution"""
    def __init__(self, dim, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.dwconv = DepthwiseConv2D(
            kernel_size=kernel_size,
            padding='same',
            use_bias=True,
            depthwise_initializer='glorot_uniform'
        )

    def call(self, x):
        return self.dwconv(x)

class LSKBlock(Layer):
    """Large Selective Kernel Block"""
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        
        # Spatial branch components
        self.conv0 = DepthwiseConv2D(5, padding='same', depth_multiplier=1)
        self.conv_spatial = DepthwiseConv2D(
            7, strides=1, padding='same', dilation_rate=3, depth_multiplier=1)
        
        # Channel processing
        self.conv1 = Conv2D(dim//2, 1, use_bias=False)
        self.conv2 = Conv2D(dim//2, 1, use_bias=False)
        self.conv_squeeze = Conv2D(2, 7, padding='same', activation='sigmoid')
        self.conv_fuse = Conv2D(dim, 1, use_bias=False)

    def build(self, input_shape):
        self.conv0.build(input_shape)
        self.conv_spatial.build(input_shape)
        self.conv1.build(input_shape)
        self.conv2.build(input_shape)
        self.conv_squeeze.build((*input_shape[:-1], 2))
        self.conv_fuse.build((*input_shape[:-1], self.dim // 2))
        super().build(input_shape)

    def call(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        
        # Split and process channels
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        # Spatial attention fusion
        attn = ops.concatenate([attn1, attn2], axis=-1)
        avg_attn = ops.mean(attn, axis=-1, keepdims=True)
        max_attn = ops.max(attn, axis=-1, keepdims=True)
        agg = ops.concatenate([avg_attn, max_attn], axis=-1)
        
        # Adaptive weighting
        sig = self.conv_squeeze(agg)
        attn = attn1 * sig[:,:,:, 0:1] + attn2 * sig[:,:,:, 1:2]
        return self.conv_fuse(attn) * x

@keras.saving.register_keras_serializable(package="cla_unet")
class LSKAttention(Layer):
    """LSK Attention Module"""
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.proj1 = Conv2D(dim, 1, activation='gelu')
        self.spatial_gate = LSKBlock(dim)
        self.proj2 = Conv2D(dim, 1)
        self.gamma = self.add_weight(
            shape=(dim,), initializer='zeros', trainable=True)

    def build(self, input_shape):
        self.proj1.build(input_shape)
        projected_shape = self.proj1.compute_output_shape(input_shape)
        self.spatial_gate.build(projected_shape)
        self.proj2.build(projected_shape)
        super().build(input_shape)

    def call(self, x):
        shortcut = x
        x = self.proj1(x)
        x = self.spatial_gate(x)
        x = self.proj2(x)
        return shortcut + self.gamma * x

    def get_config(self):
        return {**super().get_config(), "dim": self.dim}


def build_model(shape:Tuple[int]=(120,240,2),
                  c_shape:Tuple[int]=(120,240,2),
                  input_variables:List[str]=ALL_VARIABLES,
                  start_filters:int=64,
                  l2_reg:float=0.001,
                  background_flag:float=-3.0,
                  include_range_folded:bool=True,
                  head='maxpool',
                  use_attention_gate=True):
    # 输入层（保持原VGG的输入处理）
    inputs = {}
    for v in input_variables:
        inputs[v] = keras.Input(shape, name=v)
    
    normalized_inputs = keras.layers.Concatenate(axis=-1, name='Concatenate1')(
        [normalize(inputs[v], v) for v in input_variables]
    )
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)

    if include_range_folded:
        range_folded = keras.Input(shape[:2]+(shape[2],), name='range_folded_mask')
        inputs['range_folded_mask'] = range_folded
        normalized_inputs = keras.layers.Concatenate(axis=-1, name='Concatenate2')(
            [normalized_inputs, range_folded])
    
    cin = keras.Input(c_shape, name='coordinates')
    inputs['coordinates'] = cin

    x, c = normalized_inputs, cin

    # 编码器路径
    encoder_x = []
    encoder_c = []
    for i in range(4):
        x,c = conv_block(x,c, filters=start_filters*(2**i), ksize=3, l2_reg=l2_reg)
        encoder_x.append(x)
        encoder_c.append(c)
        x = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
        c = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(c)

    encoder_c.append(c)

    # 瓶颈层
    x,c = conv_block(x,c, filters=start_filters*16, ksize=3, l2_reg=l2_reg)

    # 解码器路径
    for i in reversed(range(4)):
        x = keras.layers.concatenate([x, encoder_c[i+1]], axis=-1)
        x = keras.layers.Conv2DTranspose(
            filters=start_filters*(2**i), kernel_size=3, strides=2, padding='valid')(x)        
        
        # 动态调整尺寸保证跳跃连接
        target_x_shape = encoder_x[i].shape[1:3]
        x = keras.layers.Resizing(*target_x_shape)(x)

        skip = encoder_x[i]
        if use_attention_gate:
            skip = attention_gate(
                skip, x, encoder_c[i], filters=start_filters * (2 ** i), name=f"ag_{i + 1}"
            )
        x = keras.layers.concatenate([x, skip])
        x = keras.layers.Conv2D(filters=64,
                          kernel_size=3,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          padding='same',
                          activation='relu')(x)
        x = keras.layers.Conv2D(filters=64,
                          kernel_size=3,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          padding='same',
                          activation='relu')(x)    

    if head=='mlp':
        # MLP head
        x = keras.layers.Flatten()(x) 
        x = keras.layers.Dense(units = 4096, activation ='relu')(x) 
        x = keras.layers.Dense(units = 2024, activation ='relu')(x) 
        output = keras.layers.Dense(1)(x)
    elif head=='maxpool':
        # Per gridcell
                                              
        # x = keras.layers.Conv2D(filters=512, kernel_size=1,
        #                   kernel_regularizer=keras.regularizers.l2(l2_reg),
        #                   activation='relu')(x)
        x = keras.layers.Conv2D(filters=2, kernel_size=3,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          activation='relu')(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=1,name='heatmap')(x)
        # Max in scene
        output = keras.layers.GlobalMaxPooling2D()(x)

    return keras.Model(inputs=inputs,outputs=output)

# 核心U-Net架构
def conv_block(x,c, filters, ksize=3, l2_reg=1e-6):
    x,c = CoordConv2D(filters=filters,
                        kernel_size=ksize,
                        kernel_regularizer=keras.regularizers.l2(l2_reg),
                        padding='same',
                        activation='relu')([x,c])
    x,c = CoordConv2D(filters=filters,
                        kernel_size=ksize,
                        kernel_regularizer=keras.regularizers.l2(l2_reg),
                        padding='same',
                        activation='relu')([x,c])
    x = LSKAttention(filters)(x)            
    x = keras.layers.Dropout(rate=0.1)(x)
    return x,c

def vgg_block(x,c, filters=64, ksize=3, n_convs=2, l2_reg=1e-6, drop_rate=0.0):

    for _ in range(n_convs):
        x,c = CoordConv2D(filters=filters,
                          kernel_size=ksize,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          padding='same',
                          activation='relu')([x,c])
    x = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    c = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(c)
    if drop_rate>0:
        x = keras.layers.Dropout(rate=drop_rate)(x)
    return x,c


def normalize(x,
              name:str):
    """
    Channel-wise normalization using known CHANNEL_MIN_MAX
    """
    min_max = np.array(CHANNEL_MIN_MAX[name]) # [2,]
    n_sweeps=x.shape[-1]
    
    # choose mean,var to get approximate [-1,1] scaling
    var=((min_max[1]-min_max[0])/2)**2 # scalar
    var=np.array(n_sweeps*[var,])    # [n_sweeps,]
    
    offset=(min_max[0]+min_max[1])/2    # scalar
    offset=np.array(n_sweeps*[offset,]) # [n_sweeps,]

    return keras.layers.Normalization(mean=offset,
                                         variance=var,
                                         name='Normalize_%s' % name)(x)


@keras.saving.register_keras_serializable(package="cla_unet")
class CoordConv2D(keras.layers.Layer):
    """
    Adopted from the CoodConv2d layers as described in 

    Liu, Rosanne, et al. "An intriguing failing of convolutional neural networks and 
    the coordconv solution." Advances in neural information processing systems 31 (2018).
    
    """
    def __init__(self,filters,
                      kernel_size,
                      kernel_regularizer,
                      activation,
                      padding='same',
                      strides=1,
                      conv2d_kwargs = {},
                      **kwargs):

        super(CoordConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.conv2d_kwargs = conv2d_kwargs
        self.strd = strides  # assume equal strides

        self.conv = keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            kernel_regularizer=self.kernel_regularizer,
            activation=self.activation,
            padding=self.padding,
            strides=self.strides,
            **conv2d_kwargs
        )

    def build(self, input_shape):
        x_shape, coord_shape = input_shape
        concat_shape = list(x_shape)
        concat_shape[-1] += coord_shape[-1]
        self.conv.build(concat_shape)

    def call(self,inputs):
        """
        inputs is a tuple 
           [N, L, W, C] data tensor,
           [N, L, W, nd] tensor of coordiantes
        """
        x, coords = inputs

        # Stack x with coordinates
        x = ops.concatenate( (x,coords), axis=-1)

        # Run convolution
        conv=self.conv(x)

        # The returned coordinates should have same shape as conv
        # prep the coordiantes by slicing them to the same shape
        # as conv
        if self.padding=='same' and self.strd>1:
            coords = coords[:,::self.strd,::self.strd]
        elif self.padding=='valid':
            # If valid padding,  need to start slightly off the corner
            i0 = self.kernel_size//2
            if i0>0:
                coords = coords[:,i0:-i0:self.strd,i0:-i0:self.strd]
            else:
                coords = coords[:,::self.strd,::self.strd]

        return conv,coords

    def get_config(self):
        """Get model configuration, used for saving model."""
        config = super().get_config()
        config.update(
            {   "filters": self.filters,
                "kernel_size": self.kernel_size,
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
                "activation":self.activation,
                "padding": self.padding,
                "strides": self.strides,
                "conv2d_kwargs": self.conv2d_kwargs
            }
        )
        return config


@keras.saving.register_keras_serializable()
class FillNaNs(keras.layers.Layer):
    """Fill NaNs with fill_val"""
    def __init__(self, fill_val, **kwargs):
        super(FillNaNs, self).__init__(**kwargs)
        self.fill_val = fill_val

    def __call__(self, x):
        return ops.where(ops.isnan(x), self.fill_val, x)

    def get_config(self):
        """Get model configuration, used for saving model."""
        config = super().get_config()
        config.update({"fill_val": self.fill_val})
        return config


class FromLogitsMixin:
  def __init__(self, from_logits=True, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.from_logits = from_logits

  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.from_logits:
      y_pred = ops.sigmoid(y_pred)
    return super().update_state(y_true, y_pred, sample_weight)

  def get_config(self):
    return {**super().get_config(), "from_logits": self.from_logits}

class AUC(FromLogitsMixin, keras.metrics.AUC):
  ...

@keras.saving.register_keras_serializable(package="cla_unet")
class BinaryAccuracy(FromLogitsMixin, keras.metrics.BinaryAccuracy):
  ...

@keras.saving.register_keras_serializable(package="cla_unet")
class TruePositives(FromLogitsMixin, keras.metrics.TruePositives):
  ...

@keras.saving.register_keras_serializable(package="cla_unet")
class FalsePositives(FromLogitsMixin, keras.metrics.FalsePositives):
  ...

@keras.saving.register_keras_serializable(package="cla_unet")
class TrueNegatives(FromLogitsMixin, keras.metrics.TrueNegatives):
  ...

@keras.saving.register_keras_serializable(package="cla_unet")
class FalseNegatives(FromLogitsMixin, keras.metrics.FalseNegatives):
  ...

@keras.saving.register_keras_serializable(package="cla_unet")
class Precision(FromLogitsMixin, keras.metrics.Precision):
  ...

@keras.saving.register_keras_serializable(package="cla_unet")
class Recall(FromLogitsMixin, keras.metrics.Recall):
  ...

@keras.saving.register_keras_serializable(package="cla_unet")
class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1", from_logits=False, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.precision = Precision(from_logits)
        self.recall = Recall(from_logits)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return (2 * p * r) / (p + r + keras.config.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        return {**super().get_config(), "from_logits": self.from_logits}


def make_exp_dir(exp_dir='../experiments',prefix='',symlink_name='latest',
                 task_type=None, task_id=0):
    """
    Creates a dated directory for an experiement, and also creates a symlink 
    """
    linked_dir=exp_dir+'/%s' % symlink_name
    dated_dir=prefix+'%s-%s-%s' % (datetime.datetime.now().strftime('%y%m%d%H%M%S'),
                                   os.getenv('SLURM_JOB_ID'),
                                   os.getenv('SLURM_ARRAY_TASK_ID'))
    try:
        dated_dir = os.path.join(os.getenv('SLURM_ARRAY_JOB_ID'),dated_dir)
    except:
        pass
    os.makedirs(os.path.join(exp_dir,dated_dir))
    if os.path.islink(linked_dir):
        os.unlink(linked_dir)
    os.symlink(dated_dir,linked_dir)
    return os.path.join(exp_dir,dated_dir)


def make_callback_dirs(logdir):
    tensorboard_dir = os.path.join(logdir, 'tboard')
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    
    checkpoints_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    return tensorboard_dir, checkpoints_dir


LMF_DEFAULTS = {
    "focal_alpha": 0.5,  # Class-balancing factor for tornado labels inside Focal loss.
    "focal_gamma": 2.0,
    "ldam_class_counts": None,  # Set [n_non_tornado, n_tornado] for custom training data.
    "ldam_max_margin": 0.5,
    "ldam_scale": 1.0,
    "lmf_initial_focal_weight": 0.5,  # The initial LDAM weight is its complement.
    "lmf_trainable_weights": True,
    "lmf_mix_l2": 0.01,  # Stabilize the learned balance between the two terms.
}

DEFAULT_CONFIG={
    'epochs':100,
    'input_variables':ALL_VARIABLES,
    'train_years':list(range(2013,2021)),
    'val_years':list(range(2021,2023)),
    'batch_size':128,
    'model':'vgg',
    'start_filters':64,
    'learning_rate':1e-4,
    'decay_steps':1386,
    'decay_rate':0.958,
    'l2_reg':1e-5,
    'wN':1.0,
    'w0':1.0,
    'w1':1.0,
    'w2':2.0,
    'wW':0.5,
    'label_smooth':0,
    'loss':'lmf',
    'use_attention_gate':True,
    'head':'maxpool',
    'exp_name':'tornet_baseline',
    'exp_dir':EXP_DIR,
    'dataloader':"torch",
    'dataloader_kwargs': {},
    **LMF_DEFAULTS,
}

def main(config):
    config = dict(config)
    # Gather all hyperparams
    epochs=config.get('epochs')
    batch_size=config.get('batch_size')
    start_filters=config.get('start_filters')
    learning_rate=config.get('learning_rate')
    decay_steps=config.get('decay_steps')
    decay_rate=config.get('decay_rate')
    l2_reg=config.get('l2_reg')
    wN=config.get('wN')
    w0=config.get('w0')
    w1=config.get('w1')
    w2=config.get('w2')
    wW=config.get('wW')
    head=config.get('head')
    label_smooth=config.get('label_smooth')
    loss_fn = config.get('loss').lower()
    input_variables=config.get('input_variables')
    exp_name=config.get('exp_name')
    exp_dir=config.get('exp_dir')
    train_years=config.get('train_years')
    val_years=config.get('val_years')
    dataloader=config.get('dataloader')
    dataloader_kwargs = config.get('dataloader_kwargs')

    logging.info(f"Using {keras.config.backend()} backend")
    logging.info(f'Using {dataloader} dataloader')
    logging.info('Running with config:')
    logging.info(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Using {device} device_1")

    weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
    
    # Create data laoders
    dataloader_kwargs = {'select_keys':input_variables+['range_folded_mask','coordinates']}
    # dataloader_kwargs = dict(config.get("dataloader_kwargs") or {})
    # dataloader_kwargs["select_keys"] = input_variables + [
    #     "range_folded_mask",
    #     "coordinates",
    # ]
    # if dataloader.lower() in ("torch", "torch-tfds"):
    #     dataloader_kwargs.setdefault("workers", 3)

    ds_train = get_dataloader(dataloader, DATA_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs)
    ds_val = get_dataloader(dataloader, DATA_ROOT, val_years, "train", batch_size, weights, **dataloader_kwargs)    
    
    in_shapes = (120,240,2)
    c_shapes = (120,240,2)
    nn = build_model(shape=in_shapes,
                     c_shape=c_shapes,
                     start_filters=start_filters,
                     l2_reg=l2_reg,
                     input_variables=input_variables,
                     head=head,
                     use_attention_gate=config.get('use_attention_gate', True))
    
    # model setup
    lr=keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate, staircase=False, name="exp_decay")
    loss_config = {key: config.get(key, default) for key, default in LMF_DEFAULTS.items()}
    if loss_fn in {"lmf", "ldam"}:
        counts = loss_config["ldam_class_counts"]
        if counts is None:
            if "tfds" in dataloader.lower():
                raise ValueError("Set ldam_class_counts for the TFDS training split.")
            counts = count_training_classes(DATA_ROOT, train_years)
        loss_config["ldam_class_counts"] = counts
        config.update(loss_config)
        logging.info("LDAM training class counts [non-tornado, tornado]: %s", counts)

    from_logits=True
    if loss_fn in {"cce", "bce"}:
        loss = keras.losses.BinaryCrossentropy( from_logits=from_logits, 
                                                    label_smoothing=label_smooth )
    elif loss_fn == 'hinge':
        loss = keras.losses.Hinge() # automatically converts labels to -1,1
    elif loss_fn == 'mae':
        loss = lambda yt,yp: mae_loss(yt,yp)
    elif loss_fn == 'focal':
        loss = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=loss_config["focal_alpha"],
            gamma=loss_config["focal_gamma"],
            from_logits=from_logits,
        )
    elif loss_fn == 'ldam':
        loss = BinaryLDAMLoss(
            class_counts=loss_config["ldam_class_counts"],
            max_margin=loss_config["ldam_max_margin"],
            scale=loss_config["ldam_scale"],
        )
    elif loss_fn == 'lmf':
        nn = LMFTrainingModel(nn, loss_config, name="cla_unet_lmf")
        loss = None  # LMFTrainingModel.compute_loss includes the learnable weights.
    else:
        raise RuntimeError('unknown loss %s' % loss_fn)


    opt  = keras.optimizers.Adam(learning_rate=lr)

    # Compute various metrics while training
    metrics = [keras.metrics.AUC(from_logits=from_logits,name='AUC',num_thresholds=2000),
                keras.metrics.AUC(from_logits=from_logits,curve='PR',name='AUCPR',num_thresholds=2000), 
                BinaryAccuracy(from_logits,name='BinaryAccuracy'), 
                TruePositives(from_logits,name='TruePositives'),
                FalsePositives(from_logits,name='FalsePositives'), 
                TrueNegatives(from_logits,name='TrueNegatives'),
                FalseNegatives(from_logits,name='FalseNegatives'), 
                Precision(from_logits,name='Precision'), 
                Recall(from_logits,name='Recall'),
                F1Score(from_logits=from_logits,name='F1')]
    
    nn.compile(loss=loss,
                metrics=metrics,
                optimizer=opt,
                weighted_metrics=[])
    
    ## Setup experiment directory and model callbacks
    expdir = make_exp_dir(exp_dir=exp_dir,prefix=exp_name)
    logging.info('expdir='+expdir)

    # Copy the properties that were used
    with open(os.path.join(expdir,'data.json'),'w') as f:
        json.dump(
            {'data_root':DATA_ROOT,
             'train_data':list(train_years), 
             'val_data':list(val_years)},f)
    with open(os.path.join(expdir,'params.json'),'w') as f:
        json.dump({'config':config},f)
    # Copy the training script
    shutil.copy(__file__, os.path.join(expdir,'train.py')) 
    
    # Callbacks
    tboard_dir, checkpoints_dir=make_callback_dirs(expdir)
    checkpoint_name=os.path.join(checkpoints_dir, 'tornadoDetector'+'_{epoch:03d}.keras' )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint_name,monitor='val_loss',save_best_only=False),
        keras.callbacks.CSVLogger(os.path.join(expdir,'history.csv')),
        keras.callbacks.TerminateOnNaN(),
    ]

    # TensorBoard callback requires tensorflow backend
    if keras.config.backend() == "tensorflow":
        callbacks.append(keras.callbacks.TensorBoard(log_dir=tboard_dir,write_graph=False))#,profile_batch=(5,15)),

    ## FIT
    history=nn.fit(ds_train,
                   epochs=epochs,
                    validation_data=ds_val,
                    callbacks=callbacks,
                    verbose=1) 

    if loss_fn == 'lmf':
        focal_weight, ldam_weight = nn.mix_weights()
        logging.info(
            "Final LMF weights: Focal=%.6f, LDAM=%.6f",
            ops.convert_to_numpy(focal_weight), ops.convert_to_numpy(ldam_weight),
        )
    
    # At the end,  report the best score observed over all epochs
    if len(history.history['val_AUC'])>0:
        best_auc = np.max(history.history['val_AUC'])
        best_aucpr = np.max(history.history['val_AUCPR'])
    else:
        best_auc,best_aucpr=0.5,0.0
    
    return {'AUC':best_auc,'AUCPR':best_aucpr}


if __name__=='__main__':
    config=DEFAULT_CONFIG
    # Load param file if given
    if len(sys.argv)>1:
        config.update(json.load(open(sys.argv[1],'r')))
    main(config)
