# Image-classification-CatsVsDogs
This project involves developing and evaluating two Convolutional Neural Network (CNN) models for classifying images of cats and dogs using TensorFlow and Keras.

#### Repository Structure
- **data/**: Directory containing the dataset of cat and dog images.
- **models/**: Contains the definitions and training scripts for the two CNN models.
- **notebooks/**: Jupyter notebooks used for data exploration and model training.
- **scripts/**: Python scripts for preprocessing data, training models, and evaluating performance.
- **results/**: Stores training logs, model checkpoints, and evaluation metrics.

#### Key Components
1. **Data Preparation:**
   - **Loading and Preprocessing**: Images are loaded and split into training, validation, and test sets.
   - **Augmentation**: Data augmentation techniques are applied to enhance the training dataset.

2. **Model Architecture:**
   - **Custom CNN Model**: A sequential model with convolutional, pooling, and dense layers.
   - **MobileNetV2 Transfer Learning Model**: A pre-trained MobileNetV2 model with additional dense layers for binary classification.

3. **Training and Evaluation:**
   - Models are trained on the augmented dataset.
   - Performance is evaluated using accuracy and loss metrics on the test set.
   - Training and validation metrics are visualized.

4. **Results:**
   - **Custom CNN Model**:
     - Training Accuracy: 85.58%
     - Validation Accuracy: 81.52%
     - Test Accuracy: 82.56%
   - **MobileNetV2 Model**:
     - Training Accuracy: 84.76%
     - Validation Accuracy: 80.80%
     - Test Accuracy: 81.12%

#### Sample Code
```python
# Define and compile custom CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('data/train', target_size=(150, 150), batch_size=32, class_mode='binary')
val_generator = val_test_datagen.flow_from_directory('data/validation', target_size=(150, 150), batch_size=32, class_mode='binary')

# Train model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Evaluate model
test_generator = val_test_datagen.flow_from_directory('data/test', target_size=(150, 150), batch_size=32, class_mode='binary')
test_loss, test_accuracy = model.evaluate(test_generator)

# Plot metrics
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
```

#### Conclusion
The project demonstrates the effectiveness of CNNs for binary image classification, comparing a custom CNN with a transfer learning approach using MobileNetV2. Both models achieve good accuracy, with the custom CNN slightly outperforming the MobileNetV2 model on the test set.

