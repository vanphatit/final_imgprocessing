import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import os

# === CONFIG ===
DATA_DIR = "asl-detect/dataset_retrain"
MODEL_OUT = "asl-detect/asl-detector-retrained-v2.h5"
CLASS_NAMES_FILE = "asl-detect/class_names.txt"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# === Tạo class_names.txt ===
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
with open(CLASS_NAMES_FILE, "w") as f:
    for name in class_names:
        f.write(name + "\n")
print(f"📁 Đã ghi {len(class_names)} lớp vào {CLASS_NAMES_FILE}")

# === Data augmentation mạnh tay ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.6, 1.4),
    shear_range=0.2,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# === Tạo model MobileNetV2 ===
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = True  # fine-tune tất cả

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile với LR thấp
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    ]
)

# Save model
model.save(MODEL_OUT)
print(f"✅ Mô hình đã lưu: {MODEL_OUT}")