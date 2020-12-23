import tensorflow as tf
from dataloader import training_set, test_set, BATCH_SIZE, TEST_BATCH_SIZE
from model import simple_model
from time import time

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    f'models/simple_model_{time()}.h5',
    monitor='val_accuracy',
    verbose=0,
    save_best_only=True,    
    save_weights_only=False,
    mode='auto',
    save_freq='epoch',
    )

simple_model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


history = simple_model.fit_generator(
            training_set,
            epochs=50,
            steps_per_epoch=10_000//BATCH_SIZE,
            validation_data=test_set,
            validation_steps=984//TEST_BATCH_SIZE,
            callbacks=[model_checkpoint]
        )
