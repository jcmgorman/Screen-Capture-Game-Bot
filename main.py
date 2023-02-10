import os
import pyautogui
import keyboard

from mss import mss
from datetime import datetime
from enum import Enum
from PIL import Image

import numpy as np
# ML Stuff
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


img_height = 270
img_width = 480


class Inputs(Enum):
    """
    Issue with RoR:
    Some frames are the exact same for different inputs
    May be causing issues with classification
    Possible new games:
        Lethal League
        Binding of Issac
        Chrome Dino Game
        Bit Trip Runner
        Super Meat Boy
    """
    # Bit Trip Controls
    # Spring (w), Kick (k), Slide (s), Block (d), Jump (j)

    Left = 'a'
    Right = 'd'
    Up = 'w'
    Down = 's'
    Jump = 'j'
    Ability1 = 'k'
    Ability2 = 'd'
    Ability3 = 'r'
    Ability4 = '4'
    Use = 'l'
    Enter = 'q'
    Swap = 'r'


# needs more combat data
# make it so if p is press then record and p to stop recording
def capture_data():

    print("Capturing Screen and Game Inputs")
    paused = True
    while True:
        pressedKeys = ''
        # Left
        if keyboard.is_pressed(Inputs.Left.value):
            pressedKeys += Inputs.Left.value
        # Right
        if keyboard.is_pressed(Inputs.Right.value):
            pressedKeys += Inputs.Right.value
        # Up
        if keyboard.is_pressed(Inputs.Up.value):
            pressedKeys += Inputs.Up.value
        # Down
        if keyboard.is_pressed(Inputs.Down.value):
            pressedKeys += Inputs.Down.value
        # Jump
        if keyboard.is_pressed(Inputs.Jump.value):
            pressedKeys += Inputs.Jump.value
        # Ability 1
        if keyboard.is_pressed(Inputs.Ability1.value):
            pressedKeys += Inputs.Ability1.value
        # Ability 2
        if keyboard.is_pressed(Inputs.Ability2.value):
            pressedKeys += Inputs.Ability2.value
        # Ability 3
        if keyboard.is_pressed(Inputs.Ability3.value):
            pressedKeys += Inputs.Ability3.value
        # Ability 4
        if keyboard.is_pressed(Inputs.Ability4.value):
            pressedKeys += Inputs.Ability4.value
        # Use Item
        if keyboard.is_pressed(Inputs.Use.value):
            pressedKeys += Inputs.Use.value
        # Enter
        if keyboard.is_pressed(Inputs.Enter.value):
            pressedKeys += Inputs.Enter.value
        # Swap Item
        if keyboard.is_pressed(Inputs.Swap.value):
            pressedKeys += Inputs.Swap.value
        # Pause Recording
        if keyboard.is_pressed('0'):
            paused = not paused
            if not paused:
                print("Recording Resumed...")
            else:
                print("Recording Paused...")

        # Add logic to keep track of previous button to make sure
        # not capturing too much of the same action (walking, etc.)

        if not pressedKeys == '' and not paused:
            print("Recording...")

            timeStr = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            fileName = timeStr + "-" + pressedKeys + ".png"
            if not os.path.exists("./Capture Data/" + pressedKeys):
                os.mkdir("./Capture Data/" + pressedKeys)

            fileName = "./Capture Data/" + pressedKeys + "/" + fileName

            # The simplest use, save a screen shot of the 1st monitor
            with mss() as sct:
                sct.shot(mon=1, output=fileName)

        # Capture rate is 24 frames per second
        # 1 second / 24 frames
        time.sleep(0.04)


def train_model():
    batch_size = 64
    epochs = 25
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "./Capture Data",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "./Capture Data",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    num_classes = len(train_ds.class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 5, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 5, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 5, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Add logic to save and load class names as part of model
    np.save('model_labels', class_names)
    model.save('RORModel')
    print("MODEL SAVED!")


def main():
    batch_size = 64
    img_width = 480
    img_height = 270

    # This should be own sep script
    # capture_data()

    # train_model()

    model = load_model('RORModel')
    labels = np.load('model_labels.npy')
    print(labels)
    print(model)

    time.sleep(5)

    keyboard.press('esc')

    keys = []
    while True:
        with mss() as sct:
            sct.shot(mon=1, output="./buffer.png")
        keyboard.press('esc')
        img = tf.keras.utils.load_img("./buffer.png", target_size=(270, 480))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = model.predict(img_array)

        for key in keys:
            keyboard.release(key)

        score = tf.nn.softmax(predictions[0])
        keys = labels[np.argmax(score)]
        print(keys)
        keyboard.release('esc')
        for key in keys:
            keyboard.press(key)
        #time.sleep(.03)

    return -1


if __name__ == "__main__":
    main()
