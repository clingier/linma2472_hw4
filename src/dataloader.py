from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

data_path = 'data/tomato'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'val')
BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

train_gen = ImageDataGenerator()

test_gen = ImageDataGenerator()

training_set = train_gen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=BATCH_SIZE
)

test_set = test_gen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=TEST_BATCH_SIZE,
)
