from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

data_path = 'data/tomato'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'val')

train_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
)

test_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True
)

train_iter = train_gen.flow_from_directory(
    train_path,
    target_size=(256, 256),
    color_mode="rgb",
)

test_iter = test_gen.flow_from_directory(
    test_path,
    target_size=(256, 256),
    color_mode="rgb",
)
