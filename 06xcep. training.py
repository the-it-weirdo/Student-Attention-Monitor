'''
Code to train the Xception model
Author: Team 1
'''
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

relative_directory_to_img = "./"
dataframe_path = "./Split_20_Train70.csv"
BATCH_SIZE = 16
EPOCHS = 5
IMG_SHAPE = (299, 299, 3)


def create_model():
    '''
    Function to build the xception model
    '''
    # base model as Xception
    # removed top layer
    base_model = tf.keras.applications.Xception(include_top=False,
                                                weights='imagenet',
                                                pooling="avg",
                                                input_shape=IMG_SHAPE)

    base_model.trainable = False  # ensuring base model is not trainable

    # top layer
    top = tf.keras.models.Sequential()
    top.add(tf.keras.layers.Dense(256, activation="relu",
            input_shape=base_model.output_shape[1:]))
    top.add(tf.keras.layers.Dropout(0.5))
    top.add(tf.keras.layers.Dense(128, activation="relu"))
    top.add(tf.keras.layers.Dropout(0.5))
    top.add(tf.keras.layers.Dense(64, activation="relu"))
    top.add(tf.keras.layers.Dropout(0.5))
    top.add(tf.keras.layers.Dense(16, activation="softmax"))

    model = tf.keras.Sequential(layers=[base_model, top])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

    tf.keras.utils.plot_model(base_model, "Base_Model.png")
    tf.keras.utils.plot_model(top, "Top_Layer.png")
    tf.keras.utils.plot_model(model, "Final_Model.png")

    return model


callbacks = [tf.keras.callbacks.ModelCheckpoint(
    filepath="Checkpoints",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2,
                                     patience=2, verbose=1),]


model = create_model()
model.summary()

# Reading input
df = pd.read_csv(dataframe_path)

df = df[["Filepath", "label"]]

# Using an ImageDataGenerator as data size is large
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.3,
    preprocessing_function=tf.keras.applications.xception.preprocess_input)  # Change preprocessing function to your appropirate model

# flowing from the Processed dataset
train_generator = generator.flow_from_dataframe(
    dataframe=df, directory="./", x_col="Filepath", y_col="label", class_mode="categorical", batch_size=BATCH_SIZE, shuffle=True, subset="training")

validation_generator = generator.flow_from_dataframe(
    dataframe=df, directory="./", x_col="Filepath", y_col="label", class_mode="categorical", batch_size=BATCH_SIZE, shuffle=True, subset="validation")


history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.samples // BATCH_SIZE,
                              epochs=EPOCHS,
                              callbacks=callbacks,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.samples // BATCH_SIZE)


with open("history.pickle", "wb") as outfile:
    pickle.dump(history, outfile)


model.save("trained_xception.h5")
