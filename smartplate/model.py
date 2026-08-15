"""Pre-trained multi-task EfficientNetV2B0 model construction."""

from __future__ import annotations

from typing import Any

from .config import ProjectConfig


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - exercised only without ML extras
        raise RuntimeError(
            "TensorFlow is required for model training. Install requirements.txt first."
        ) from exc
    return tf


def compile_model(
    model: Any,
    num_ingredients: int,
    learning_rate: float,
    config: ProjectConfig,
) -> None:
    tf = _tensorflow()
    keras = tf.keras
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "ingredients": keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True,
                alpha=0.75,
                gamma=2.0,
                name="ingredient_focal_loss",
            ),
            "nutrition": keras.losses.Huber(name="nutrition_huber_loss"),
        },
        loss_weights={
            "ingredients": config.recognition_loss_weight,
            "nutrition": config.nutrition_loss_weight,
        },
        metrics={
            "ingredients": [
                keras.metrics.AUC(
                    name="auc", multi_label=True, num_labels=num_ingredients
                ),
                keras.metrics.Precision(name="precision", thresholds=0.5),
                keras.metrics.Recall(name="recall", thresholds=0.5),
            ],
            "nutrition": [keras.metrics.MeanAbsoluteError(name="scaled_mae")],
        },
    )


def build_multitask_model(
    num_ingredients: int,
    config: ProjectConfig | None = None,
    weights: str | None = "imagenet",
) -> tuple[Any, Any]:
    """Build the project model and return ``(model, pretrained_backbone)``."""

    if num_ingredients < 1:
        raise ValueError("num_ingredients must be positive")
    config = config or ProjectConfig()
    tf = _tensorflow()
    keras = tf.keras
    layers = keras.layers

    inputs = keras.Input(
        shape=(config.image_size, config.image_size, 3),
        dtype="float32",
        name="image",
    )
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )
    augmented = augmentation(inputs)

    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_shape=(config.image_size, config.image_size, 3),
        include_preprocessing=True,
    )
    backbone.trainable = False
    features = backbone(augmented, training=False)
    features = layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = layers.Dropout(config.dropout_rate, name="backbone_dropout")(features)
    features = layers.Dense(256, activation="swish", name="shared_dense")(features)
    features = layers.Dropout(config.dropout_rate, name="shared_dropout")(features)

    ingredient_output = layers.Dense(
        num_ingredients, activation="sigmoid", name="ingredients"
    )(features)
    nutrition_output = layers.Dense(2, activation="softplus", name="nutrition")(
        features
    )
    model = keras.Model(
        inputs=inputs,
        outputs={"ingredients": ingredient_output, "nutrition": nutrition_output},
        name="smartplate_efficientnetv2b0",
    )
    compile_model(model, num_ingredients, config.head_learning_rate, config)
    return model, backbone


def unfreeze_for_fine_tuning(
    model: Any,
    backbone: Any,
    num_ingredients: int,
    config: ProjectConfig | None = None,
) -> None:
    """Unfreeze the final backbone fraction while keeping BatchNorm stable."""

    config = config or ProjectConfig()
    tf = _tensorflow()
    start = int(len(backbone.layers) * (1.0 - config.fine_tune_fraction))
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= start and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    compile_model(model, num_ingredients, config.fine_tune_learning_rate, config)

