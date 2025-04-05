import pandas as pd
from keras import optimizers
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, SeparableConv2D, UpSampling2D, ZeroPadding2D, concatenate
from keras.models import Model

# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_generators(train_path, val_path, batch_size=16):
    train_datagen = ImageDataGenerator(
        rotation_range=40, width_shift_range=0.1, height_shift_range=0.1, preprocessing_function=preprocess_input, horizontal_flip=False, fill_mode="nearest"
    )
    train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), shuffle=True, class_mode="categorical", batch_size=batch_size)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator = val_datagen.flow_from_directory(val_path, target_size=(224, 224), class_mode="categorical", shuffle=True, batch_size=batch_size)
    return train_generator, val_generator


# For plantvillage dataset, the number of classes is 38 and the input shape is (224, 224, 3)
# For sugarcane dataset, the number of classes is 5 and the input shape is (224, 224, 3)
def load_model(num_classes=38, input_shape=(224, 224, 3)):
    # Encoder Start
    base_model1 = Xception(include_top=False, weights="imagenet", input_shape=input_shape)
    x1_0 = base_model1.output
    x1_0 = Flatten(name="Flatten1")(x1_0)
    dense1 = Dense(256, name="fc1", activation="relu")(x1_0)
    x = classif_out_encoder1 = Dense(num_classes, name="out1", activation="softmax")(dense1)  # Latent Representation / Bottleneck

    # Get Xception's tensors for skip connection.
    conv14 = base_model1.get_layer("block14_sepconv2_act").output
    conv13 = base_model1.get_layer("block13_sepconv2_bn").output
    conv12 = base_model1.get_layer("block12_sepconv3_bn").output
    conv11 = base_model1.get_layer("block11_sepconv3_bn").output
    conv10 = base_model1.get_layer("block10_sepconv3_bn").output
    conv9 = base_model1.get_layer("block9_sepconv3_bn").output
    conv8 = base_model1.get_layer("block8_sepconv3_bn").output
    conv7 = base_model1.get_layer("block7_sepconv3_bn").output
    conv6 = base_model1.get_layer("block6_sepconv3_bn").output
    conv5 = base_model1.get_layer("block5_sepconv3_bn").output
    conv4 = base_model1.get_layer("block4_sepconv2_bn").output
    conv3 = base_model1.get_layer("block3_sepconv2_bn").output
    conv2 = base_model1.get_layer("block2_sepconv2_bn").output
    conv1 = base_model1.get_layer("block1_conv2_act").output

    # Decoder Start
    dense2 = Dense(256, activation="relu")(x)

    x = Add(name="first_merge")([dense1, dense2])
    x = Dense(7 * 7 * 2048)(x)
    reshape1 = Reshape((7, 7, 2048))(x)

    # BLOCK 1
    x = SeparableConv2D(2048, (3, 3), padding="same", name="block14_start")(reshape1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = concatenate([conv14, x], axis=3)
    x = SeparableConv2D(1536, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = c14 = Activation("relu")(x)

    # BLOCK 2
    x = UpSampling2D((2, 2))(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(1024, (3, 3), padding="same", name="block13_start")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv13, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    c1314 = Conv2D(728, (1, 1))(UpSampling2D()(c14))
    x = add1 = Add()([c1314, x])

    # BLOCK 3
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same", name="blockmiddle_start")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv12, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = add2 = Add()([add1, x])
    # BLOCK 4
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv11, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = add3 = Add()([add2, x])
    # BLOCK 5
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv10, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = add4 = Add()([add3, x])
    # BLOCK 6
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv9, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = add5 = Add()([add4, x])
    # BLOCK 7
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv8, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = add6 = Add()([add5, x])
    # BLOCK 8
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv7, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = add7 = Add()([add6, x])
    # BLOCK 9
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv6, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = add8 = Add()([add7, x])
    # BLOCK 10
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv5, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same", name="blockmiddle_end")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = add9 = Add()([add8, x])

    # BLOCK 11
    x = UpSampling2D((2, 2))(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same", name="block4_start")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv4, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(728, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    c45 = Conv2D(728, (1, 1))(UpSampling2D()(add9))
    x = add10 = Add()([c45, x])

    # BLOCK 12
    x = Conv2DTranspose(1, (3, 3), strides=(2, 2))(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(256, (3, 3), padding="valid", name="block3_start")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv3, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    c34 = Conv2D(256, (3, 3), padding="valid")(Conv2DTranspose(1, (3, 3), strides=(2, 2))(add10))
    x = add11 = Add()([c34, x])

    # BLOCK 13
    x = Conv2DTranspose(1, (3, 3), strides=(2, 2))(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(128, (3, 3), padding="valid", name="block2_start")(x)
    x = BatchNormalization()(x)
    x = concatenate([conv2, x], axis=3)
    x = Activation("relu")(x)
    x = SeparableConv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    c23 = Conv2D(128, (3, 3), padding="valid")(Conv2DTranspose(1, (3, 3), strides=(2, 2))(add11))
    x = Add()([c23, x])

    # BLOCK 14
    x = Conv2D(64, (3, 3), padding="same", name="block1_start")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = concatenate([conv1, x], axis=3)
    x = ZeroPadding2D()(x)
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling2D()(x)
    x = ZeroPadding2D()(x)

    x = Conv2D(
        2,
        3,
        activation="relu",
        padding="same",
    )(x)
    mask = x = Conv2D(3, 1, activation="sigmoid", name="Mask")(x)

    base_model2 = Xception(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    x2_0 = base_model2(mask)
    x2_0 = Flatten(name="Flatten2")(x2_0)
    x2_1 = Dense(256, name="fc2", activation="relu")(x2_0)
    classif_out_encoder2 = Dense(num_classes, name="out2", activation="softmax")(x2_1)
    model = Model(base_model1.input, [classif_out_encoder1, classif_out_encoder2])
    return model


def train_model(model, train_path, val_path, batch_size=16, lr=1e-4, epochs=35):
    train_generator, valid_generator = load_generators(train_path, val_path, batch_size)
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
    alpha = 0.4
    lossWeights = {"out1": alpha, "out2": (1.0 - alpha)}
    # model.compile(optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9), loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
    model.compile(optimizer=optimizers.SGD(learning_rate=lr, momentum=0.9), loss=losses, loss_weights=lossWeights, metrics={"out1": "accuracy", "out2": "accuracy"})

    # model.summary()

    history = model.fit(train(), steps_per_epoch=train_steps, epochs=epochs, validation_data=valid(), validation_steps=val_steps)

    df = pd.DataFrame(history.history)
    df.to_csv("ResTS15epochs.csv")
    try:
        model.save("ResTS.h5")
    except:
        print("Check if the model has been saved!")
    return model, history
