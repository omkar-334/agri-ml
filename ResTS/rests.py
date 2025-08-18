import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Reshape,
    SeparableConv2D,
    UpSampling2D,
    ZeroPadding2D,
    concatenate,
)
from keras.models import Model
from sklearn.metrics import classification_report

# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Reshape,
    SeparableConv2D,
    UpSampling2D,
    ZeroPadding2D,
    concatenate,
)
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_generators(train_path, val_path, batch_size=16):
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        preprocessing_function=preprocess_input,
        horizontal_flip=False,
        fill_mode="nearest",
    )
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        shuffle=True,
        class_mode="categorical",
        batch_size=batch_size,
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(224, 224),
        class_mode="categorical",
        shuffle=True,
        batch_size=batch_size,
    )
    return train_generator, val_generator


# For plantvillage dataset, the number of classes is 38 and the input shape is (224, 224, 3)
# For sugarcane dataset, the number of classes is 5 and the input shape is (224, 224, 3)


def rename_model(model, new_name):
    with tf.name_scope(new_name):
        return clone_model(model)


def resize_like(tensor, ref):
    return tf.image.resize(
        tensor, size=(tf.shape(ref)[1], tf.shape(ref)[2]), method="bilinear"
    )


def sep_block(
    x, filters, skip_conn, block_name, add_input=None, upsample=False, suffix=""
):
    if upsample:
        x = UpSampling2D(
            size=(2, 2),
            interpolation="bilinear",
            name=f"{block_name}_upsample_{suffix}",
        )(x)

    x = Activation("relu", name=f"{block_name}_act1_{suffix}")(x)
    x = SeparableConv2D(
        filters, (3, 3), padding="same", name=f"{block_name}_sepconv1_{suffix}"
    )(x)
    x = BatchNormalization(name=f"{block_name}_bn1_{suffix}")(x)

    skip_conn = Lambda(
        lambda s: tf.image.resize(
            s[0], size=(tf.shape(s[1])[1], tf.shape(s[1])[2]), method="bilinear"
        ),
        output_shape=lambda shapes: (shapes[1][0], None, None, shapes[0][-1]),
        name=f"{block_name}_resize_skip_{suffix}",
    )([skip_conn, x])

    x = concatenate([skip_conn, x], axis=-1, name=f"{block_name}_concat_{suffix}")

    x = Activation("relu", name=f"{block_name}_act2_{suffix}")(x)
    x = SeparableConv2D(
        filters, (3, 3), padding="same", name=f"{block_name}_sepconv2_{suffix}"
    )(x)
    x = BatchNormalization(name=f"{block_name}_bn2_{suffix}")(x)

    if add_input is not None:
        add_input = Lambda(
            lambda s: tf.image.resize(
                s[0], size=(tf.shape(s[1])[1], tf.shape(s[1])[2]), method="bilinear"
            ),
            output_shape=lambda shapes: (shapes[1][0], None, None, shapes[0][-1]),
            name=f"{block_name}_resize_add_{suffix}",
        )([add_input, x])

        if add_input.shape[-1] != filters:
            add_input = Conv2D(
                filters, (1, 1), padding="same", name=f"{block_name}_proj_add_{suffix}"
            )(add_input)

        x = Add(name=f"{block_name}_add_{suffix}")([add_input, x])

    return x


def load_model(num_classes=38, input_shape=(224, 224, 3)):
    base_model1 = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        name="xception_encoder",
    )
    x1_flat = Flatten(name="flatten_encoder1")(base_model1.output)
    dense1 = Dense(256, activation="relu", name="dense_encoder1")(x1_flat)
    out1 = Dense(num_classes, activation="softmax", name="output_encoder1")(dense1)

    convs = {
        f"conv{i + 1}": base_model1.get_layer(name).output
        for i, name in enumerate([
            "block14_sepconv2_act",
            "block13_sepconv2_bn",
            "block12_sepconv3_bn",
            "block11_sepconv3_bn",
            "block10_sepconv3_bn",
            "block9_sepconv3_bn",
            "block8_sepconv3_bn",
            "block7_sepconv3_bn",
            "block6_sepconv3_bn",
            "block5_sepconv3_bn",
            "block4_sepconv2_bn",
            "block3_sepconv2_bn",
            "block2_sepconv2_bn",
            "block1_conv2_act",
        ])
    }

    x = Dense(256, activation="relu", name="bottleneck_dense")(out1)
    x = Add(name="merge_dense")([dense1, x])
    x = Dense(7 * 7 * 2048, name="reshape_dense")(x)
    x = Reshape((7, 7, 2048), name="reshape_layer")(x)

    # Decoder blocks
    c14 = sep_block(x, 1536, convs["conv1"], "block14", suffix="1")
    add1 = sep_block(
        c14, 728, convs["conv2"], "block13", c14, upsample=True, suffix="2"
    )
    add2 = sep_block(add1, 728, convs["conv3"], "block12", add1, suffix="3")
    add3 = sep_block(add2, 728, convs["conv4"], "block11", add2, suffix="4")
    add4 = sep_block(add3, 728, convs["conv5"], "block10", add3, suffix="5")
    add5 = sep_block(add4, 728, convs["conv6"], "block9", add4, suffix="6")
    add6 = sep_block(add5, 728, convs["conv7"], "block8", add5, suffix="7")
    add7 = sep_block(add6, 728, convs["conv8"], "block7", add6, suffix="8")
    add8 = sep_block(add7, 728, convs["conv9"], "block6", add7, suffix="9")
    add9 = sep_block(add8, 728, convs["conv10"], "block5", add8, suffix="10")
    add10 = sep_block(
        add9, 728, convs["conv11"], "block4", add9, upsample=True, suffix="11"
    )

    # Final decoder blocks with transpose conv
    x = Conv2DTranspose(1, (3, 3), strides=(2, 2), name="upconv1")(add10)
    x = sep_block(x, 256, convs["conv12"], "block3", suffix="12")
    x = Conv2DTranspose(1, (3, 3), strides=(2, 2), name="upconv2")(x)
    x = sep_block(x, 128, convs["conv13"], "block2", suffix="13")

    x = Conv2D(64, (3, 3), padding="same", name="block1_conv1")(x)
    x = BatchNormalization(name="block1_bn1")(x)
    x = Activation("relu", name="block1_relu1")(x)
    x = concatenate([convs["conv14"], x], axis=3, name="block1_concat")
    x = ZeroPadding2D(name="block1_zeropad1")(x)
    x = Conv2D(32, (3, 3), padding="same", name="block1_conv2")(x)
    x = BatchNormalization(name="block1_bn2")(x)
    x = Activation("relu", name="block1_relu2")(x)
    x = UpSampling2D(name="block1_upsample")(x)
    x = ZeroPadding2D(name="block1_zeropad2")(x)
    x = Conv2D(2, 3, activation="relu", padding="same", name="block1_conv3")(x)
    mask = Conv2D(3, 1, activation="sigmoid", name="mask_output")(x)

    base_model2 = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        name="xception_decoder",
    )
    x2 = base_model2(mask)
    x2_flat = Flatten(name="flatten_encoder2")(x2)
    x2_dense = Dense(256, activation="relu", name="dense_encoder2")(x2_flat)
    out2 = Dense(num_classes, activation="softmax", name="output_encoder2")(x2_dense)

    return Model(
        inputs=base_model1.input, outputs=[out1, out2], name="DualXceptionAutoencoder"
    )


# train_generator, valid_generator = load_generators(train_path, val_path, batch_size)
def train_model(
    model,
    train_generator,
    valid_generator,
    batch_size=16,
    lr=0.0001,
    epochs=35,
    alpha=0.4,
):
    train_steps = len(train_generator) // batch_size
    val_steps = len(valid_generator) // batch_size

    def train():
        while True:
            x, y = next(train_generator)
            yield x, {"out1": y, "out2": y}

    def valid():
        while True:
            x, y = next(valid_generator)
            yield x, {"out1": y, "out2": y}

    losses = {"out1": "categorical_crossentropy", "out2": "categorical_crossentropy"}
    weights = {"out1": alpha, "out2": (1.0 - alpha)}
    metrics = {"out1": "accuracy", "out2": "accuracy"}

    # model.compile(optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9), loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
    model.compile(
        optimizer=optimizers.SGD(learning_rate=lr, momentum=0.9),
        loss=losses,
        loss_weights=weights,
        metrics=metrics,
    )

    history = model.fit(
        train(),
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid(),
        validation_steps=val_steps,
    )

    df = pd.DataFrame(history.history)
    df.to_csv("ResTS15epochs.csv")
    validate_model(model, valid_generator, batch_size)
    return model, history


def validate_model(model, valid_generator, batch_size=16):
    val_steps = len(valid_generator) // batch_size

    def valid():
        while True:
            x, y = next(valid_generator)
            yield x, {"out1": y, "out2": y}

    y_true, y_pred = [], []
    valid_generator.reset()

    for _ in range(val_steps):
        x_batch, y_batch = next(valid())
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch["out2"], axis=1))
        y_pred.extend(np.argmax(preds[1], axis=1))

    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("classification_report.csv", float_format="%.4f")

    return model
