from __future__ import annotations
from typing import Tuple, Optional, Literal
import tensorflow as tf
# pyright: reportMissingImports=false
from tensorflow.keras import layers, Model


BackboneName = Literal["efficientnetb0", "resnet50", "mobilenetv2"]


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.4,
    dense_units: int = 64,
    name: str = "baseline_cnn_flatten",
) -> Model:
    """
    Baseline CNN uczona od zera.
    - 3 bloki konwolucyjne: 32 -> 64 -> 128
    - MaxPooling po każdym bloku
    - Flatten + Dense(dense_units) + Dropout(dropout_rate)
    - Wyjście: Dense(1) + sigmoid
    - Normalizacja: Rescaling(1/255) w modelu (loader bez normalizacji)
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # Normalizacja pod baseline (loader zwraca wartości 0..255 w float32)
    x = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    # Blok 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)  # 224 -> 112

    # Blok 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)  # 112 -> 56

    # Blok 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)  # 56 -> 28

    # Klasyfikator
    x = layers.Flatten(name="flatten")(x)  # 28*28*128 = 100352
    x = layers.Dense(dense_units, activation="relu", name=f"dense_{dense_units}")(x)
    x = layers.Dropout(dropout_rate, name=f"dropout_{dropout_rate}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_transfer_model(
    backbone: BackboneName = "efficientnetb0",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    trainable_backbone: bool = False,
    dropout_rate: float = 0.3,
    name: Optional[str] = None,
) -> Model:
    """
    Model Transfer Learning:
    - Wybierany backbone z tf.keras.applications (pretrained ImageNet)
    - Wlasciwy preprocess_input zależny od backbone (w modelu)
    - GlobalAveragePooling2D
    - Dropout
    - Dense(1) + sigmoid

    trainable_backbone=False -> czyli klasyczny start: zamrozony backbone.
    """
    inputs = layers.Input(shape=input_shape, name="input")

    if backbone == "efficientnetb0":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=None, input_shape=input_shape
        )
        default_name = "tl_efficientnetb0"
    elif backbone == "resnet50":
        preprocess = tf.keras.applications.resnet50.preprocess_input
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=None, input_shape=input_shape
        )
        default_name = "tl_resnet50"
    elif backbone == "mobilenetv2":
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_tensor=None, input_shape=input_shape
        )
        default_name = "tl_mobilenetv2"
    else:
        raise ValueError(f"Nieznany backbone: {backbone}")

    base.trainable = trainable_backbone

    # Preprocess_input jako warstwa (w modelu bo loader jest wspolny)
    x = layers.Lambda(preprocess, name="preprocess_input")(inputs)

    # Backbone
    x = base(x, training=trainable_backbone)

    # Head klasyfikacyjny
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name=f"dropout_{dropout_rate}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model_name = name or default_name
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model
