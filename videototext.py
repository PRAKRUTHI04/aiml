import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Step 2: Load clips and prepare dataset
def load_clips(data_dir, clip_length=16, frame_size=(64, 64)):
    clips = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Get the list of class folders (e.g., class001, class002)
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):  # Skip if it's not a folder
            continue
            
        for video_folder in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_folder)
            if not os.path.isdir(video_path):  # Ensure it's a directory
                continue

            nested_dirs = [d for d in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, d))]
            if nested_dirs:
                video_path = os.path.join(video_path, nested_dirs[0])

            frames = sorted(os.listdir(video_path))  # Ensure frames are ordered
            video_frames = []

            for frame_file in frames:
                frame_path = os.path.join(video_path, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:  # Skip invalid frames
                    continue
                frame = cv2.resize(frame, frame_size)  # Resize to the target size
                video_frames.append(frame)

            # Break the video into clips of clip_length
            for i in range(0, len(video_frames) - clip_length + 1, clip_length):
                clip = video_frames[i:i + clip_length]
                clips.append(np.array(clip))
                labels.append(label)
    
    return np.array(clips), np.array(labels)

# Example usage
data_dir = "video_to_frames"  # Root directory containing classXXX folders
X_data, y_data = load_clips(data_dir)

print("Number of clips:", len(X_data))
print("Clip shape:", X_data[0].shape if len(X_data) > 0 else "No clips loaded")
print("Labels shape:", y_data.shape)

# Step 3: Model Definition
class_names = sorted(os.listdir(data_dir))  # Get class names
model = Sequential([
    Conv3D(32, (3, 3, 3), activation='relu', input_shape=(16, 64, 64, 3)),
    MaxPooling3D((2, 2, 2)),
    Conv3D(64, (3, 3, 3), activation='relu'),
    MaxPooling3D((2, 2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 4: Data Preprocessing and Training
x, y = load_clips(data_dir)
x = x.astype(np.float32)

# Split the data into training, validation, and test sets
x_train, x_remaining, y_train, y_remaining = train_test_split(x, y, test_size=0.2, random_state=100)
x_val, x_test, y_val, y_test = train_test_split(x_remaining, y_remaining, test_size=0.5, random_state=100)

# Normalize the data
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
num_classes = len(class_names)  # Number of classes in your dataset
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,  # Adjust based on your needs
    batch_size=8,  # Adjust based on available GPU memory
    verbose=1
)

# Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model
model.save("sign_language_model.h5")

# Optionally, load the saved model
loaded_model = tf.keras.models.load_model("sign_language_model.h5")
