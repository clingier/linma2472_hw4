from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

data_path = 'data/tomato'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'val')
BATCH_SIZE = 32

train_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
)

test_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True
)

training_set = train_gen.flow_from_directory(
    train_path,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=BATCH_SIZE
)

test_set = test_gen.flow_from_directory(
    test_path,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=BATCH_SIZE
)
