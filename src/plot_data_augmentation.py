from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

p = Path(r'''.\data\tomato\train\Tomato___healthy\84cbb98b-5c2f-4367-8d78-52be491e66bd___GH_HL Leaf 336.jpg''')

x = mpimg.imread(p)
# print(x.shape)
# print(x)
# plt.imshow(x)
# plt.show()

dg = ImageDataGenerator(
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

x = x[np.newaxis, :]
transformed_data = dg.flow(x, y=None)

fig = plt.figure(figsize=(10, 10))
axes = []

for i, x in enumerate(transformed_data):
    x = x[0, :].astype(np.uint8)
    axes.append(fig.add_subplot(2, 2, i + 1))
    axes[i].imshow(x, aspect='auto')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    if i == 3:
        break
fig.savefig('plots/data_augmentation.pdf')