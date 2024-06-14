from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, Lambda,Multiply,Concatenate
from keras.engine import Model
from keras.optimizers import Adam

from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss,dice_coefficient,get_label_dice_coefficient_function,\
    tree_triplit_loss_function,conditional_loss,NSUP_loss,cosine_annealing


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def DHAS_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
               n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
               train_mode="conditional"):
    """
       This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
       https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

       This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
       Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


       :param input_shape:
       :param n_base_filters:
       :param depth:
       :param dropout_rate:
       :param n_segmentation_levels:
       :param n_labels:
       :param optimizer:
       :param initial_learning_rate:
       :param loss_function:
       :param activation_name:
       :return:
       """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2 ** level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers=None
    triple_layer=None
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            ###  generate the conditional probability
            segmentation_layers = Conv3D(level_filters[level_number], (1, 1, 1))(current_layer)
            segmentation_layers = Activation('relu')(segmentation_layers)
            segmentation_layers = Conv3D(n_labels, (1, 1, 1))(segmentation_layers)

            triple_layer = Conv3D(32, (1, 1, 1))(current_layer)
            triple_layer = Activation('relu')(triple_layer)
            triple_layer = Conv3D(32, (1, 1, 1))(triple_layer)

    segmentation_output = Activation("sigmoid")(segmentation_layers)
    triple_output = triple_layer

    ###  generate the unconditional probability
    if train_mode == "unconditional":
        # 提取
        m = Lambda(lambda x: x[:, :, :, :, 0])(segmentation_output)
        n = Lambda(lambda x: x[:, :, :, :, 1])(segmentation_output)
        p = Lambda(lambda x: x[:, :, :, :, 2])(segmentation_output)

        # 计算 unconditional prob
        mn = Multiply()([m, n])
        mnp = Multiply()([mn, p])

        # 将 m、mn 和 mnp 在最后一个轴叠加起来
        segmentation_output = Concatenate(axis=-1)([m, mn, mnp])

    model = Model(inputs=inputs, outputs=[segmentation_output, triple_output])

    if train_mode == "conditional":
        model.compile(optimizer=optimizer(lr=initial_learning_rate),
                      loss=[conditional_loss, tree_triplit_loss_function],
                      loss_weights=[1, 0.5],
                      metrics=None)

    else:
        model.compile(optimizer=optimizer(lr=initial_learning_rate),
                      loss=[NSUP_loss, tree_triplit_loss_function],
                      loss_weights=[1, 0.2],
                      metrics=None)

    return model


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2



