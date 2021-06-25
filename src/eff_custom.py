from typing import Tuple

import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Concatenate, Add, UpSampling2D, BatchNormalization
from config import INPUT_SHAPE_IMAGE, ENCODER_WEIGHTS

# r = base_model.get_layer('block5b_expand_bn')


class EffCustom(keras):
    def __init__(self, out_channel: int, channel_agg: str, image_shape: Tuple[int, int, int] = INPUT_SHAPE_IMAGE,
                 encoder_weights: str = ENCODER_WEIGHTS) -> None:

        self.image_shape = image_shape
        self.encoder_weights = encoder_weights
        self.out_channel = out_channel
        self.channel_agg = channel_agg

    def build_base_model(self) -> tf.keras.models.Model:

        base_model = efn.EfficientNetB0(input_shape=self.image_shape, weights=self.encoder_weights, include_top=False)
        return base_model

    def BasicConv2d(self, x, out_channel_basic: int, kernel_size: Tuple = (0, 0), stride: Tuple = (1, 1),
                    padding: Tuple = (0, 0), dilation: Tuple = (0, 0)):

        conv = Conv2D(filters=out_channel_basic, kernel_size=kernel_size, strides=stride,
                      padding=padding, dilation_rate=dilation, use_bias=False)(x)
        bn = BatchNormalization(conv)
        relu = keras.layers.Activation('relu')(bn)

        return relu

    def rfb(self, x):

        relu = keras.layers.Activation('relu')
        branch0 = keras.Sequential(
            self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(1, 1))(x)
        )
        branch1 = keras.Sequential(
            I
            self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(1, 1)),
            self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(1, 3), padding=(0, 1)),
            self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(3, 1), padding=(1, 0)),
            self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(3, 3), padding=(3, 3),
                             dilation=(3, 3))(
        )(x)
        branch2 = keras.Sequential(
            self.BasicConv2d(x, out_channel_basic=self.out_channel, kernel_size=(1, 1)),
            self.BasicConv2d(x, out_channel_basic=self.out_channel, kernel_size=(1, 5), padding=(0, 2)),
            self.BasicConv2d(x, out_channel_basic=self.out_channel, kernel_size=(5, 1), padding=(2, 0)),
            self.BasicConv2d(x, out_channel_basic=self.out_channel, kernel_size=(3, 3), padding=(5, 5))
        )(x)
        branch3 = keras.Sequential(
            Conv2D(filters=self.out_channel, kernel_size=1),
            Conv2D(filters=self.out_channel, kernel_size=(1, 7), padding=(0, 3)),
            Conv2D(filters=self.out_channel, kernel_size=(3, 1), padding=(3, 0)),
            Conv2D(filters=self.out_channel, kernel_size=3, padding=7, dilation_rate=7)
        )(x)

        conv_res = Conv2D(filters=self.out_channel, kernel_size=1)
        conv_cat = Conv2D()
        conc = Concatenate(axis=1)([branch0, branch1, branch2, branch3])

        x = relu(conc)

    def aggregation(self):
        # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
        # used after MSF
        relu = keras.layers.Activation('relu')

        upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')
        conv_upsample_1 = Conv2D(filters=self.channel_agg, kernel_size=3, padding=1)(upsample)
        conv_upsample_2 = Conv2D(filters=self.channel_agg, kernel_size=3, padding=1)(conv_upsample_1)
        conv_upsample_3 = Conv2D(filters=self.channel_agg, kernel_size=3, padding=1)(conv_upsample_2)
        conv_upsample_4 = Conv2D(filters=self.channel_agg, kernel_size=3, padding=1)(conv_upsample_3)
        conv_upsample_5 = Conv2D(filters=2*self.channel_agg, kernel_size=3, padding=1)

        conv_concat2 = Conv2D(filters=2*self.channel_agg, kernel_size=3, padding=1)
        conv_concat3 = Conv2D(filters=2*self.channel_agg, kernel_size=3, padding=1)

        conv4 = Conv2D(filters=3*self.channel_agg, kernel_size=3, padding=1)
        conv5 = Conv2D(filters=3*self.channel_agg, kernel_size=1, padding=1)


