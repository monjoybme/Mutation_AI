import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout

def custom_resnet50_block(x, filters, kernel_size, strides=1, use_shortcut=False):
    shortcut = x

    x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    if use_shortcut:
        shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_custom_resnet50(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual Blocks
    x = custom_resnet50_block(x, filters=64, kernel_size=3, use_shortcut=True)
    x = custom_resnet50_block(x, filters=64, kernel_size=3)

    x = custom_resnet50_block(x, filters=128, kernel_size=3, strides=2, use_shortcut=True)
    x = custom_resnet50_block(x, filters=128, kernel_size=3)

    x = custom_resnet50_block(x, filters=256, kernel_size=3, strides=2, use_shortcut=True)
    x = custom_resnet50_block(x, filters=256, kernel_size=3)

    x = custom_resnet50_block(x, filters=512, kernel_size=3, strides=2, use_shortcut=True)
    x = custom_resnet50_block(x, filters=512, kernel_size=3)

    # Global Average Pooling and Dense Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model
