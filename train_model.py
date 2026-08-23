import os
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 2
EPOCHS = 15

DATASET_PATH = "dataset"


datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.5,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)


train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)


validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)


model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)


os.makedirs("model", exist_ok=True)

model.save(
    "model/deepfake_model.h5"
)


print("Model training completed!")
print("Model saved to model/deepfake_model.h5")