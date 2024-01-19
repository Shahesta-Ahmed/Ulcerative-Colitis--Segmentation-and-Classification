import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import json

import os

data_dir = '/Users/shahestaahmed/Documents/ulcerative_colitis/newoverlay'
class_names = ['M0', 'M1', 'M2', 'M3']
class_counts = {}

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    class_counts[class_name] = len(os.listdir(class_dir))

print(class_counts)

total_samples = sum(class_counts.values())
class_weights = {i: total_samples / class_counts[class_name] for i, class_name in enumerate(class_names)}

print(class_weights)

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Create a simple model on top
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')  
])

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# Data loaders with split
data_dir = '/Users/shahestaahmed/Documents/ulcerative_colitis/newoverlay'
train_dataset = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    batch_size=32,
    image_size=(224, 224))

val_dataset = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    batch_size=32,
    image_size=(224, 224))



class_names = train_dataset.class_names
print("Class names: ", class_names)
# Preprocess input
train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y))

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=25,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,  # You can set a higher epoch since early stopping is in place
    class_weight=class_weights,  # Using class weights
    callbacks=[early_stopping])  # Add early stopping here


model.save('/Users/shahestaahmed/Documents/overlay_resnet')

history_file_path = '/Users/shahestaahmed/Documents/overlay_resnet_history.json'
with open(history_file_path, 'w') as f:
    json.dump(history.history, f)

