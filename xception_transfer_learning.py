import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from pprint import pprint
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score
import pandas as pd
import pickle

BATCH_SIZE = 32
IMG_SIZE = (299, 299)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def compute_F1_score(model):
    y_pred = []
    y_true = []
    data_ids = os.listdir("./dataset/test")
    # loss, accuracy = model.evaluate(test_dataset)
    # print('Test accuracy :', accuracy)

    # image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    # predictions = model.predict_on_batch(image_batch).flatten()

    # predictions = tf.nn.sigmoid(predictions)
    # predictions = tf.where(predictions < 0.5, 0, 1)

    # print('Predictions:\n', predictions.numpy())
    # print('Labels:\n', label_batch)
    for file in tqdm(data_ids):
        y_true.append(file.split(".")[0])
        img = tf.keras.preprocessing.image.load_img(
            f"./dataset/test/{file}", target_size=IMG_SIZE
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        prob = predictions[0]
        pprint(prob)
        print(
            "This image is %.2f percent fake and %.2f percent real."
            % (100 * (1 - prob), 100 * prob)
        )
        pred = "real" if prob > 0.5 else "fake"
        y_pred.append(pred)
    print(f"F1-score: {f1_score(y_true, y_pred, average='macro')}")

def store_model(model, name):
    with open(name+'.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        print("MODEL STORED ON DISK")

def load_model(name):
    with open(name+'.pickle', 'rb') as f:
        model = pickle.load(f)
        print("MODEL LOADED FROM DISK")
        return model

train_dataset = tf.keras.utils.image_dataset_from_directory("./DFGC-2021",
                                                            seed=1337,
                                                            validation_split=0.2,
                                                            subset="training",
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory("./DFGC-2021",
                                                            seed=1337,
                                                            validation_split=0.2,
                                                            subset="validation",
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

test_dataset = tf.keras.utils.image_dataset_from_directory("./dataset/modified_test",
                                                            seed=1337,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

class_names = train_dataset.class_names

# plt.figure(figsize=(33, 33))
# for images, labels in train_dataset.take(1):
#     for i in range(32):
#         ax = plt.subplot(4, 8, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomFlip('vertical'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1,0.1)
])

preprocess_input = tf.keras.applications.xception.preprocess_input
# Create the base model from the pre-trained model Xception V1
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.xception.Xception(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()

initial_epochs = 40
loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

# # store_model(model, 'model')
# # load_model('model')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()
plt.savefig("before_fine.jpg")

##
# y_pred = []
# y_true = []
# data_ids = os.listdir("./dataset/test")
# for file in tqdm(data_ids):
#   y_true.append(file.split(".")[0])
#   img = tf.keras.preprocessing.image.load_img(
#       f"./dataset/test/{file}", target_size=IMG_SIZE
#   )
#   img_array = tf.keras.preprocessing.image.img_to_array(img)
#   img_array = tf.expand_dims(img_array, 0)  # Create batch axis

#   predictions = model.predict(img_array)
#   pprint(f"./dataset/test/{file}")
#   prob = predictions[0]
#   pprint(prob)
#   print(
#      "This image is %.2f percent fake and %.2f percent real."
#      % (100 * (1 - prob), 100 * prob)
#   )
#   pred = "real" if prob > 0.5 else "fake"
#   y_pred.append(pred)

# display and save the output predictions of test set
# df = pd.DataFrame(data={"identifiant": data_ids, "classe predite": y_pred})
# df.to_csv('output_pred.csv') 
# print(f"F1-score: {f1_score(y_true, y_pred, average='macro')}")

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))
fine_tune_at = 120

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()
print(len(model.trainable_variables))

fine_tune_epochs = 120
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()
plt.savefig("after_fine_tune.jpg")

store_model(model, 'fine_tuned_model')

def test_for_accuracy(modelname = 'fine_tuned_model'):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    model = load_model(modelname)

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

def test_for_F1(modelname = 'fine_tuned_model'):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    model = load_model(modelname)
    compute_F1_score(model)

test_for_accuracy()
test_for_F1()