import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.model_selection import train_test_split

IMG_SIZE = 96

def build_model(input_shape=(96, 96, 1), alpha=0.25, freeze_backbone=True):
    inputs = keras.Input(shape=input_shape)

    # grayscale → RGB
    x = layers.Concatenate()([inputs, inputs, inputs])
    #x = tf.image.grayscale_to_rgb(inputs)

    base_model = keras.applications.MobileNet(
        input_shape=(96, 96, 3),
        #input_shape=(96, 96, 1),
        alpha=alpha,
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False
    x = base_model(x, training=False)
    print("Feature map:", x.shape)

    # 🔥 BOTH POOLINGS (paper style)
    embedding = layers.GlobalAveragePooling2D()(x)
    print("Embedding shape:", embedding.shape)
    print("Feature vector size:", x.shape[-1])
    #hidden = layers.Dense(256, activation='relu')(embedding)
    #score = layers.Dense(1, activation='sigmoid')(hidden)
    #model = keras.Model(inputs, outputs= [score, embedding])
    model = keras.Model(inputs, outputs= embedding)
    return model, base_model


def load_data_int8(base_path):
    """Load data and apply EXACT Arduino preprocessing"""
    X_person, X_nonperson, y_person, y_nonperson = [], [], [], []

    for label_name, label_id in [("Nonperson", 0), ("Person", 1)]:  # ← Changed to "non_person"
        folder = os.path.join(base_path, label_name)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]

        if(label_id == 0):
            for file in files:
                path = os.path.join(folder, file)

                # Load as grayscale
                img = tf.keras.utils.load_img(
                    path,
                    color_mode="grayscale",
                    target_size=(IMG_SIZE, IMG_SIZE)
                )

                img = tf.keras.utils.img_to_array(img).astype(np.float32)

                # EXACT Arduino preprocessing
                img = img - 128.0      # [0, 255] -> [-128, 127]
                img = img / 128.0      # [-128, 127] -> [-1.0, 1.0]

                X_nonperson.append(img)
                y_nonperson.append(label_id)

        elif(label_id == 1):
            for file in files:
                path = os.path.join(folder, file)

                # Load as grayscale
                img = tf.keras.utils.load_img(
                    path,
                    color_mode="grayscale",
                    target_size=(IMG_SIZE, IMG_SIZE)
                )

                img = tf.keras.utils.img_to_array(img).astype(np.float32)

                # EXACT Arduino preprocessing
                img = img - 128.0      # [0, 255] -> [-128, 127]
                img = img / 128.0      # [-128, 127] -> [-1.0, 1.0]

                X_person.append(img)
                y_person.append(label_id)


    return np.array(X_person, dtype=np.float32), np.array(X_nonperson, dtype=np.float32), np.array(y_person, dtype=np.int32), np.array(y_nonperson, dtype=np.int32)

def get_persondetect_data(path):

    print(path)
    # Build model
    model, base_model = build_model(input_shape=(96, 96, 1), alpha=0.25, freeze_backbone=True)

    # Load ALL data (not just 100!)
    X_person, X_nonperson, y_person, y_nonperson = load_data_int8(path)

    print(f"Data shape: {X_person.shape}")
    print(f"Labels shape: {y_person.shape}")
    print(f"X min: {X_person.min()}, X max: {X_person.max()}")  # Should be ~[-1, 1]
    #print(f"Class distribution: {np.bincount(y_person)}")

    # Split data
    X_person_train, X_person_test, y_person_train, y_person_test = train_test_split(
        X_person, y_person, test_size=0.2, random_state=42
    )

    X_nonperson_train, X_nonperson_test, y_nonperson_train, y_nonperson_test = train_test_split(
        X_nonperson, y_nonperson, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_person_train.shape}, Test set: {X_person_test.shape}")

    X_person_train_feat = model.predict(X_person_train, verbose=0)
    X_person_test_feat  = model.predict(X_person_test, verbose=0)
    X_nonperson_train_feat = model.predict(X_nonperson_train, verbose=0)
    X_nonperson_test_feat  = model.predict(X_nonperson_test, verbose=0)

    num_classes = 2
    y_person_train_oh = np.eye(num_classes)[y_person_train]
    y_person_test_oh  = np.eye(num_classes)[y_person_test]
    y_nonperson_train_oh = np.eye(num_classes)[y_nonperson_train]
    y_nonperson_test_oh  = np.eye(num_classes)[y_nonperson_test]

    idx = np.random.permutation(len(X_person_train_feat))
    X_person_train_feat = X_person_train_feat[idx]
    y_person_train_oh = y_person_train_oh[idx]

    # Person test
    idx = np.random.permutation(len(X_person_test_feat))
    X_person_test_feat = X_person_test_feat[idx]
    y_person_test_oh = y_person_test_oh[idx]

    # Non-person train
    idx = np.random.permutation(len(X_nonperson_train_feat))
    X_nonperson_train_feat = X_nonperson_train_feat[idx]
    y_nonperson_train_oh = y_nonperson_train_oh[idx]

    # Non-person test
    idx = np.random.permutation(len(X_nonperson_test_feat))
    X_nonperson_test_feat = X_nonperson_test_feat[idx]
    y_nonperson_test_oh = y_nonperson_test_oh[idx]


    return np.array(X_person_train_feat, dtype=np.float32), np.array(X_nonperson_train_feat, dtype=np.float32),np.array(X_person_test_feat, dtype=np.float32), np.array(X_nonperson_test_feat, dtype=np.float32), np.array(y_person_train_oh, dtype=np.int32), np.array(y_nonperson_train_oh, dtype=np.int32), np.array(y_person_test_oh, dtype=np.int32), np.array(y_nonperson_test_oh, dtype=np.int32)

