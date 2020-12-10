import tensorflow as tf
from dataloader import training_set, test_set, BATCH_SIZE
from model import model

with tf.device('cpu:0'):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(
        training_set,
        epochs=10,
        steps_per_epoch=10000//BATCH_SIZE,
        validation_data=test_set,
        validation_steps=984//BATCH_SIZE
    )

    model.save_weights('models/first_try.h5')
