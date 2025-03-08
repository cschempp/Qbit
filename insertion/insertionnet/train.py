import os
import datetime
import h5py
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Concatenate, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def build_insertion_net():
    """
    Build the model for insertion tasks.
    The model takes two inputs:
        1. An image input of shape (256, 256, 3).
        2. A force/torque input of shape (6,).
    """
    
    inputs_images = Input(shape=(256, 256, 3))
    inputs_force_torque = Input(shape=(6,))

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu")(inputs_images)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
    x = Conv2D(filters=125, kernel_size=(1, 1), strides=(1, 1), activation="relu")(x)
    x = Flatten()(x)

    x = Concatenate()([x, inputs_force_torque])
    x = Dense(units=32, activation="relu")(x)
    action = Dense(units=5)(x)

    return Model([inputs_images, inputs_force_torque], action)


def load_dataset(dataset_path):
    with h5py.File(dataset_path, "r") as f:
        augmented_force_torque = f["augmented_force_torque"][()]
        augmented_images = f["augmented_images"][()]
        policy_actions = f["policy_actions"][()]
    return augmented_images, augmented_force_torque, policy_actions


def main():
    InsertionNet = build_insertion_net()
    InsertionNet.summary()

    dataset_path = os.path.join("data", "augmented_dataset.h5")
    augmented_images, augmented_force_torque, policy_actions = load_dataset(dataset_path)

    InsertionNet.compile(optimizer=Adam(learning_rate=0.25e-4), loss="mse")

    cb_checkpoint = ModelCheckpoint(
        "checkpoints/insertionnet_I",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    )

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = InsertionNet.fit(
        x=[augmented_images, augmented_force_torque],
        y=policy_actions,
        batch_size=64,
        epochs=200,
        verbose="auto",
        callbacks=[cb_checkpoint, tensorboard_callback],
        validation_split=0.2,
        shuffle=True,
    )


if __name__ == "__main__":
    main()
