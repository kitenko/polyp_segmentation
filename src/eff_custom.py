from typing import Tuple

import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Concatenate, Add, UpSampling2D, BatchNormalization, multiply
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

    @staticmethod
    def BasicConv2d(x, out_channel_basic: int, kernel_size: Tuple = (0, 0), stride: Tuple = (1, 1),
                    padding: Tuple = (0, 0), dilation: Tuple = (0, 0)):

        conv = Conv2D(filters=out_channel_basic, kernel_size=kernel_size, strides=stride,
                      padding=padding, dilation_rate=dilation, use_bias=False)(x)
        bn = BatchNormalization(conv)
        relu = keras.layers.Activation('relu')(bn)

        return relu

    def rfb(self, x):

        relu = keras.layers.Activation('relu')
        branch0 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(1, 1))(x)

        branch1 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(1, 1))(x)
        branch1 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(1, 3), padding=(0, 1))(branch1)
        branch1 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(3, 1), padding=(1, 0))(branch1)
        branch1 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(3, 3), padding=(3, 3),
                                   dilation=(3, 3))(branch1)

        branch2 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(1, 1))(x)
        branch2 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(1, 5), padding=(0, 2))(branch2)
        branch2 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(5, 1), padding=(2, 0))(branch2)
        branch2 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(3, 3), padding=(5, 5),
                                   dilation=(5, 5))(branch2)

        branch3 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=1)(x)
        branch3 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(1, 7), padding=(0, 3))(branch3)
        branch3 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=(3, 1), padding=(3, 0))(branch3)
        branch3 = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=3, padding=7,
                                   dilation=(7, 7))(branch3)

        conv_res = self.BasicConv2d(out_channel_basic=self.out_channel, kernel_size=1)
        concat = Concatenate(axis=1)([branch0, branch1, branch2, branch3])
        concat_conv_res = Add()([concat, conv_res])
        out_rfb = relu(concat_conv_res)

        return out_rfb

    def aggregation(self, x1, x2, x3, channel):
        # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
        # used after MSF

        x1_1 = x1
        upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1)
        conv_upsample_1 = self.BasicConv2d(out_channel_basic=channel, kernel_size=3, padding=1)(upsample)
        x2_1 = multiply([conv_upsample_1, x2])

        upsample_2_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(upsample)
        conv_upsample_2 = self.BasicConv2d(out_channel_basic=channel, kernel_size=3, padding=1)(upsample_2_1)
        upsample_2_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x2)
        conv_upsample_3 = self.BasicConv2d(out_channel_basic=channel, kernel_size=3, padding=1)(upsample_2_2)
        conv_upsample_2_upsample_3 = multiply([conv_upsample_2, conv_upsample_3])
        x3_1 = multiply([conv_upsample_2_upsample_3, x3])

        upsample_3_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1_1)
        conv_upsample_4 = self.BasicConv2d(out_channel_basic=channel, kernel_size=3, padding=1)(upsample_3_1)
        x2_2 = Concatenate([x2_1, conv_upsample_4], axis=1)
        x2_2 = self.BasicConv2d(out_channel_basic=2*channel, kernel_size=3, padding=1)(x2_2)

        upsample_4_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x2_2)
        conv_upsample_5 = self.BasicConv2d(out_channel_basic=2*channel, kernel_size=3, padding=1)(upsample_4_1)
        x3_2 = Concatenate([x3_1, conv_upsample_5], axis=1)
        x3_2 = self.BasicConv2d(out_channel_basic=3*channel, kernel_size=3, padding=1)(x3_2)

        x = self.BasicConv2d(out_channel_basic=3*channel, kernel_size=3, padding=1)(x3_2)
        x = Conv2D(filters=1, kernel_size=1)(x)

        return x

    def build(self, x):



