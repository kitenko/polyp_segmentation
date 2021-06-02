from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, Input, Activation,
                                     Concatenate, ZeroPadding2D)

from config import INPUT_SHAPE_IMAGE


class UnetResnet18:
    def __init__(self, input_shape: Tuple[int, int, int] = INPUT_SHAPE_IMAGE, n_class_mask: int = 2,
                 final_activation: str = 'softmax', kernel_initializer_conv2d: str = 'HeUniform',
                 epsilon_batch: float = 0.00002, initial_filters: int = 64,
                 activation: str = 'swish', depth: int = 4) -> None:
        """
        Build UNet model with ResBlock.

        :param input_shape: Input image size.
        :param n_class_mask: How many classes in the output layer. Defaults to 2.
        :param initial_filters: Number of filters to start with in first convolution.
        :param depth: How deep to go in UNet i.e. how many down and up sampling you want to do in the model.
                           Filter root and image size should be multiple of 2^depth.
        :param activation: activation to use in each convolution. Defaults to 'relu'.
        :param final_activation: activation for output layer. Defaults to 'softmax'.
        """
        self.input_shape = input_shape
        self.n_class_mask = n_class_mask
        self.initial_filters = initial_filters
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation
        self.epsilon_batch = epsilon_batch
        self.kernel_initializer_conv2D = kernel_initializer_conv2d

    def build(self) -> tf.keras.models.Model:
        """
        This function creates a model from depending on the input parameters.
        :return: model
        """
        # list for long connections
        long_connection_store = []

        # first block
        inputs = Input(self.input_shape)
        bt_nr_first_block = BatchNormalization(epsilon=self.epsilon_batch, scale=False)(inputs)
        zp_first_block = ZeroPadding2D(padding=(3, 3))(bt_nr_first_block)
        conv_first_block = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation='linear',
                                  kernel_initializer=self.kernel_initializer_conv2D, use_bias=False)(zp_first_block)
        bt_nr_2_first_block = BatchNormalization(epsilon=self.epsilon_batch)(conv_first_block)
        act_first_block = Activation(self.activation)(bt_nr_2_first_block)

        long_connection_store.append(act_first_block)

        # second block
        zp_second_block = ZeroPadding2D((1, 1))(act_first_block)
        max_pool_second_block = MaxPooling2D(data_format='channels_last', pool_size=(3, 3), strides=(2, 2),
                                             trainable=True)(zp_second_block)
        bt_nr_second_block = BatchNormalization(epsilon=self.epsilon_batch)(max_pool_second_block)
        act_second_block = Activation(self.activation)(bt_nr_second_block)

        # Down sampling
        for i in range(self.depth):
            # number of convolution layer filters_conv2D
            number_filters = 2 ** i * self.initial_filters

            # First Conv2D Block
            # Skip connection
            if i == 0:
                skip_con_down_samp = Conv2D(number_filters, kernel_size=1, padding='valid', use_bias=False, strides=1,
                                            name='skip_con_down_samp{}_1'.format(i), kernel_initializer='HeUniform',
                                            data_format='channels_last', activation='linear')(act_second_block)
            else:
                skip_con_down_samp = Conv2D(number_filters, kernel_size=1, padding='valid', use_bias=False, strides=2,
                                            name='skip_con_down_samp{}_1'.format(i), kernel_initializer='HeUniform',
                                            data_format='channels_last', activation='linear')(act4_down_samp)

            if i == 0:
                zp_down_samp = ZeroPadding2D((1, 1), name='ZeroPadding2D{}_1'.format(i))(act_second_block)
                conv_down_samp = Conv2D(number_filters, kernel_size=3, padding='valid', name='Conv2D{}_1'.format(i),
                                        activation='linear', data_format='channels_last', strides=1,
                                        kernel_initializer='HeUniform', use_bias=False)(zp_down_samp)
            else:
                zp_down_samp = ZeroPadding2D((1, 1), name='ZeroPadding2D{}_1'.format(i))(act4_down_samp)
                conv_down_samp = Conv2D(number_filters, kernel_size=3, padding='valid', name='Conv2D{}_1'.format(i),
                                        activation='linear', data_format='channels_last', strides=2,
                                        kernel_initializer='HeUniform', use_bias=False)(zp_down_samp)

            bt_nr_down_samp = BatchNormalization(name='BatchNormalization{}_1'.format(i),
                                                 epsilon=self.epsilon_batch)(conv_down_samp)
            act_down_samp = Activation(self.activation, name="Activation{}_1".format(i))(bt_nr_down_samp)

            zp2_down_samp = ZeroPadding2D((1, 1), name='ZeroPadding2D{}_2'.format(i))(act_down_samp)
            conv2_down_samp = Conv2D(number_filters, kernel_size=3, padding='valid', name="Conv2D{}_2".format(i),
                                     data_format='channels_last', strides=1, kernel_initializer='HeUniform',
                                     use_bias=False, activation='linear')(zp2_down_samp)

            resconnection_down_samp = Add(name="Add{}_1".format(i))([skip_con_down_samp, conv2_down_samp])

            # Second Conv2D Block
            bt_nr_2_down_samp = BatchNormalization(name="BN{}_2".format(i),
                                                   epsilon=self.epsilon_batch)(resconnection_down_samp)
            act2_down_samp = Activation(self.activation, name="Act{}_2".format(i))(bt_nr_2_down_samp)
            zp3_down_samp = ZeroPadding2D((1, 1), name='ZeroPadding2D{}_3'.format(i))(act2_down_samp)
            conv3_down_samp = Conv2D(number_filters, kernel_size=3, padding='valid', name="Conv2D{}_3".format(i),
                                     activation='linear', data_format='channels_last', strides=1,
                                     kernel_initializer='HeUniform', use_bias=False)(zp3_down_samp)
            bt_nr_3_down_samp = BatchNormalization(name="BN{}_3".format(i),
                                                   epsilon=self.epsilon_batch)(conv3_down_samp)
            act3_down_samp = Activation(self.activation, name="Act{}_3".format(i))(bt_nr_3_down_samp)
            zp4_down_samp = ZeroPadding2D((1, 1), name='ZeroPadding2D{}_4'.format(i))(act3_down_samp)
            conv3_down_samp = Conv2D(number_filters, kernel_size=3, padding='valid', name="Conv2D{}_4".format(i),
                                     activation='linear', data_format='channels_last', strides=1,
                                     kernel_initializer='HeUniform', use_bias=False)(zp4_down_samp)

            resconnection_2_down_samp = Add(name="Add{}_2".format(i))([resconnection_down_samp, conv3_down_samp])

            bt_nr_4_down_samp = BatchNormalization(name="BN{}_4".format(i),
                                                   epsilon=self.epsilon_batch)(resconnection_2_down_samp)
            act4_down_samp = Activation(self.activation, name="Act{}_4".format(i))(bt_nr_4_down_samp)
            if i + 1 < self.depth:
                long_connection_store.append(act4_down_samp)

        # Up sampling
        # long connection from down sampling path.
        long_connection_1 = long_connection_store[-1]

        up_samp = UpSampling2D(name="UpSampling2D{}_1".format('first'), data_format='channels_last',
                               interpolation='nearest', size=(2, 2), trainable=True)(act4_down_samp)
        # Concatenation
        concatenation = Concatenate(axis=3, name="upConcatenate{}_1".format('first'))([up_samp, long_connection_1])

        for i in range(self.depth - 2, -1, -1):
            # long connection from down sampling path.
            long_connection = long_connection_store[i]
            number_filters = (2 ** i) * self.initial_filters
            #  Convolutions
            if i == self.depth - 2:
                conv_up_samp = Conv2D(number_filters, kernel_size=3, padding='same', name="upConv2D{}_1".format(i),
                                      activation='linear', data_format='channels_last', strides=1, use_bias=False,
                                      trainable=True, kernel_initializer='HeUniform')(concatenation)
            else:
                conv_up_samp = Conv2D(number_filters, kernel_size=3, padding='same', name="upConv2D{}_1".format(i),
                                      activation='linear', data_format='channels_last', strides=1, use_bias=False,
                                      trainable=True, kernel_initializer='HeUniform')(concatenation_2)

            bt_nr_1_up_samp = BatchNormalization(name="upBN{}_1".format(i))(conv_up_samp)
            act1_up_samp = Activation(self.activation, name="upAct{}_1".format(i))(bt_nr_1_up_samp)
            conv2_up_samp = Conv2D(number_filters, 3, padding='same', name="upConv2D{}_2".format(i),
                                   activation='linear', data_format='channels_last', strides=1, use_bias=False,
                                   trainable=True, kernel_initializer='HeUniform')(act1_up_samp)
            bt_nr_2_up_samp = BatchNormalization(name="upBN{}_2".format(i))(conv2_up_samp)
            act2_up_samp = Activation(self.activation, name="upAct{}_2".format(i))(bt_nr_2_up_samp)
            up_samp2 = UpSampling2D(name="UpSampling{}_2".format(i), data_format='channels_last',
                                    interpolation='nearest', size=(2, 2), trainable=True)(act2_up_samp)

            concatenation_2 = Concatenate(axis=-1, name="upConcatenate{}_2".format(i))([up_samp2, long_connection])

        # Third block
        for i in range(5):
            if i == 0:
                conv_third_block = Conv2D(32, kernel_size=3, padding='same', name="Conv2D{}_out".format(i),
                                          activation='linear', data_format='channels_last', strides=1,
                                          kernel_initializer='HeUniform', use_bias=False)(concatenation_2)
            elif i == 1 or i == 2:
                conv_third_block = Conv2D(32, kernel_size=3, padding='same', name="Conv2D{}_out".format(i),
                                          activation='linear', data_format='channels_last', strides=1,
                                          kernel_initializer='HeUniform', use_bias=False)(act_third_block)
            else:
                conv_third_block = Conv2D(16, kernel_size=3, padding='same', name="Conv2D{}_out".format(i),
                                          activation='linear', data_format='channels_last', strides=1,
                                          kernel_initializer='HeUniform', use_bias=False)(up_samp_third_block)

            bt_nr_third_block = BatchNormalization(name="BN{}_out".format(i))(conv_third_block)
            act_third_block = Activation(self.activation, name="Act{}_out".format(i))(bt_nr_third_block)

            if i == 2:
                up_samp_third_block = UpSampling2D(name="UpSampling2D{}_3".format(i), data_format='channels_last',
                                                   interpolation='nearest', size=(2, 2),
                                                   trainable=True)(act_third_block)

        # last layer
        conv_last = Conv2D(2, kernel_size=3, padding='same', name='Conv2D_last', activation='linear',
                           data_format='channels_last', strides=1, kernel_initializer='GlorotUniform',
                           use_bias=True)(act_third_block)
        softmax_out = Activation('softmax', name='segmentation')(conv_last)

        return Model(inputs, outputs=softmax_out, name='Res-UNet')


if __name__ == '__main__':
    x = UnetResnet18().build()
    x.save('custom.h5')
