"""Task 2.2 / 2.3: models."""
import tensorflow as tf
L = tf.keras.layers
METRICS = ["accuracy", tf.keras.metrics.AUC(name="auc")]

def compile_model(m, lr):
    m.compile(tf.keras.optimizers.Adam(lr), "binary_crossentropy", metrics=METRICS)
    return m

def build_cnn(img=224, dropout=0.3, lr=1e-3):
    m = tf.keras.Sequential([L.Input((img, img, 3)), L.Rescaling(1 / 255.)])
    for f in (32, 64, 128):
        m.add(L.Conv2D(f, 3, padding="same", activation="relu"))
        m.add(L.BatchNormalization()); m.add(L.MaxPooling2D())
    m.add(L.GlobalAveragePooling2D()); m.add(L.Dropout(dropout))
    m.add(L.Dense(64, activation="relu")); m.add(L.Dense(1, activation="sigmoid"))
    return compile_model(m, lr)

BACKBONES = {
    "mobilenet": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "efficientnet": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
    "resnet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
}

def build_tl(backbone="mobilenet", img=224, dropout=0.3, lr=1e-3):
    cls, prep = BACKBONES[backbone]
    base = cls(include_top=False, weights="imagenet", input_shape=(img, img, 3))
    base.trainable = False
    inp = tf.keras.Input((img, img, 3))
    x = prep(inp)                       # the preprocessing this backbone was pretrained with
    x = base(x, training=False)         # keep BatchNorm in inference mode
    x = L.GlobalAveragePooling2D()(x); x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation="sigmoid")(x)
    return compile_model(tf.keras.Model(inp, out), lr), base

def unfreeze_last(model, base, n=20, lr=1e-5):
    base.trainable = True
    for l in base.layers[:-n]:
        l.trainable = False
    for l in base.layers:               # keep BatchNorm frozen
        if isinstance(l, L.BatchNormalization):
            l.trainable = False
    return compile_model(model, lr)

def trainable_names(model, base=None):
    layers = base.layers if base is not None else model.layers
    return [l.name for l in layers if l.trainable and l.weights]
