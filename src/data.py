"""Task 2.1: tf.data pipeline."""
import pandas as pd, tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE
AUG = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)])

def read_split(name):
    df = pd.read_csv(f"splits/{name}.txt", header=None, names=["path", "label"])
    return df.path.tolist(), df.label.astype("float32").tolist()

def make_ds(name, img=224, train=False, augment=True, bs=32, seed=42):
    paths, labels = read_split(name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(paths), seed=seed)
    def load(p, y):
        x = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        x = tf.image.resize(x, (img, img))   # plain resize: aspect ratio is distorted
        x.set_shape((img, img, 3))
        return x, y
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(bs)
    if train and augment:                    # augmentation ONLY on training data
        ds = ds.map(lambda x, y: (AUG(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
