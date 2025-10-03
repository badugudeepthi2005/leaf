import json
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset info
with open('images_manifest.json') as f:
    data = json.load(f)

# Get image paths and classes
image_paths = [item['full'] for item in data]
classes = sorted(list(set(item['class'] for item in data)))
class_to_index = {cls: i for i, cls in enumerate(classes)}

# Convert labels to indices
labels = [class_to_index[item['class']] for item in data]

# Image dimensions we will use
IMG_SIZE = (224, 224)

# Load and preprocess all images (simplest way)
print("Loading and preprocessing images, please wait...")
images = []
for path in image_paths:
    img = load_img(path, target_size=IMG_SIZE)  # Resize images to same size
    arr = img_to_array(img) / 255.0  # Normalize pixel values to [0,1]
    images.append(arr)
X = np.array(images)
y = to_categorical(labels, num_classes=len(classes))

print(f"Loaded {len(X)} images.")

# Split to train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train the model for 10 epochs (you can increase after)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model to use later in Flask
model.save('model.h5')
print("Saved model as model.h5")

# Save class mapping for reuse
with open('class_mapping.json', 'w') as f:
    json.dump(class_to_index, f, indent=2)
print("Saved class mapping to class_mapping.json")
