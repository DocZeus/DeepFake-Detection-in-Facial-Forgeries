# Required libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed, LSTM, Bidirectional
from tensorflow.keras.applications import InceptionResNetV1, InceptionResNetV2
import cv2
import numpy as np
import os
from mtcnn import MTCNN

#Video frame extraction
def extract_frames(video_path, sample_rate=30):
    """Extract frames from video at specified sample rate"""
    frames = []
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        if frame_count % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frame_count += 1
    
    video.release()
    return frames

#Face detection and extraction using MTCNN
def extract_faces(frames):
    """Extract faces from frames using MTCNN"""
    detector = MTCNN()
    faces = []
    
    for frame in frames:
        # Detect faces
        detection_result = detector.detect_faces(frame)
        
        if detection_result:
            # Get the largest face in the frame
            detection = max(detection_result, key=lambda x: x['box'][2] * x['box'][3])
            x, y, width, height = detection['box']
            
            # Add margin (20 pixels as mentioned in the paper)
            margin = 20
            x_extended = max(0, x - margin)
            y_extended = max(0, y - margin)
            width_extended = min(frame.shape[1] - x_extended, width + 2*margin)
            height_extended = min(frame.shape[0] - y_extended, height + 2*margin)
            
            # Extract face with margin
            face = frame[y_extended:y_extended+height_extended, 
                         x_extended:x_extended+width_extended]
            
            # Resize to 300x300 as mentioned in the paper
            face = cv2.resize(face, (300, 300))
            faces.append(face)
        else:
            # If no face is detected, use a placeholder or skip
            # Here we use a black image as placeholder
            faces.append(np.zeros((300, 300, 3), dtype=np.uint8))
    
    return np.array(faces)

#Dataset preparation
def prepare_dataset(dataset_path, is_real=True):
    """Prepare dataset by extracting faces from videos"""
    X = []
    y = []
    
    video_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                  if f.endswith(('.mp4', '.avi'))]
    
    for video_path in video_paths:
        frames = extract_frames(video_path)
        faces = extract_faces(frames)
        
        if len(faces) > 0:
            X.append(faces)
            y.append(1 if is_real else 0)  # 1 for real, 0 for fake
    
    return X, y

#Custom Inception-ResNet model
def create_custom_inception_model():
    """Create a custom model combining InceptionResNetV1 and InceptionResNetV2"""
    
    # InceptionResNetV1 branch
    base_model_v1 = InceptionResNetV1(include_top=False, weights='imagenet', 
                                      input_shape=(300, 300, 3), pooling='avg')
    
    # Add dense layers to InceptionResNetV1 as mentioned in the paper
    x1 = base_model_v1.output
    x1 = Dense(128, activation='relu')(x1)
    x1 = Dense(68, activation='relu')(x1)
    
    # InceptionResNetV2 branch
    base_model_v2 = InceptionResNetV2(include_top=False, weights='imagenet', 
                                      input_shape=(300, 300, 3), pooling='avg')
    
    # Add dense layers to InceptionResNetV2
    x2 = base_model_v2.output
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dense(68, activation='relu')(x2)
    
    # Create models
    model_v1 = Model(inputs=base_model_v1.input, outputs=x1)
    model_v2 = Model(inputs=base_model_v2.input, outputs=x2)
    
    return model_v1, model_v2

#Feature extraction function
def extract_features(faces, model_v1, model_v2):
    """Extract features using both InceptionResNet models"""
    # Preprocess input for InceptionResNet models
    faces = faces.astype('float32')
    faces = tf.keras.applications.inception_resnet_v2.preprocess_input(faces)
    
    # Extract features from both models
    features_v1 = model_v1.predict(faces)
    features_v2 = model_v2.predict(faces)
    
    # Concatenate features
    combined_features = np.concatenate([features_v1, features_v2], axis=1)
    
    return combined_features

#Complete HODFF-DD Model
def create_hodff_dd_model(sequence_length=20, feature_dim=136):
    """Create the complete HODFF-DD model with BiLSTM"""
    
    # Input shape: (sequence_length, feature_dim)
    input_layer = Input(shape=(sequence_length, feature_dim))
    
    # BiLSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    
    # Dense layers
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

#Spotted Hyena Optimizer implementation
class SpottedHyenaOptimizer(tf.keras.optimizers.Optimizer):
    """Simplified implementation of Spotted Hyena Optimizer"""
    
    def __init__(self, learning_rate=0.00001, beta_1=0.9, beta_2=0.999, name="SpottedHyenaOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._learning_rate = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        
        m_t = m.assign(self._beta_1 * m + (1. - self._beta_1) * grad)
        v_t = v.assign(self._beta_2 * v + (1. - self._beta_2) * tf.square(grad))
        
        # Simplified spotted hyena behavior
        var_update = var.assign_sub(lr_t * m_t / (tf.sqrt(v_t) + 1e-7))
        
        return var_update
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError("Sparse gradients not supported yet")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._learning_rate,
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
        })
        return config
    
#Data Generator
def sequence_generator(X, y, batch_size=10, sequence_length=20, feature_dim=136):
    """Generate batches of sequences for training"""
    model_v1, model_v2 = create_custom_inception_model()
    
    while True:
        # Shuffle indices
        indices = np.random.permutation(len(X))
        
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_X = []
            batch_y = []
            
            for idx in batch_indices:
                faces = X[idx]
                
                # If we have enough frames
                if len(faces) >= sequence_length:
                    # Randomly select a starting point
                    start_frame = np.random.randint(0, len(faces) - sequence_length + 1)
                    selected_faces = faces[start_frame:start_frame + sequence_length]
                    
                    # Extract features
                    features = extract_features(selected_faces, model_v1, model_v2)
                    
                    batch_X.append(features)
                    batch_y.append(y[idx])
            
            if batch_X:
                yield np.array(batch_X), np.array(batch_y)

#Training function
def train_model(train_X, train_y, val_X, val_y, epochs=50, batch_size=10):
    """Train the HODFF-DD model"""
    
    # Create model
    model = create_hodff_dd_model()
    
    # Compile with spotted hyena optimizer
    optimizer = SpottedHyenaOptimizer(learning_rate=1e-5, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create generators
    train_gen = sequence_generator(train_X, train_y, batch_size)
    val_gen = sequence_generator(val_X, val_y, batch_size)
    
    # Calculate steps per epoch
    train_steps = len(train_X) // batch_size
    val_steps = len(val_X) // batch_size
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=[early_stopping]
    )
    
    return model, history

#Frame-level prediction
def predict_frames(model, video_path):
    """Predict on individual frames of a video"""
    # Extract frames
    frames = extract_frames(video_path)
    faces = extract_faces(frames)
    
    # Create feature extractor models
    model_v1, model_v2 = create_custom_inception_model()
    
    # Process in sequences
    sequence_length = 20
    predictions = []
    
    for i in range(0, len(faces) - sequence_length + 1):
        sequence = faces[i:i+sequence_length]
        features = extract_features(sequence, model_v1, model_v2)
        pred = model.predict(np.expand_dims(features, axis=0))[0][0]
        predictions.append(pred)
    
    return predictions

#Video-level classification with majority voting
def classify_video(model, video_path, threshold=0.5):
    """Classify a video as real or fake using majority voting"""
    frame_predictions = predict_frames(model, video_path)
    
    # Apply threshold to get binary predictions
    binary_preds = [1 if pred >= threshold else 0 for pred in frame_predictions]
    
    # Count real and fake frames
    real_count = sum(binary_preds)
    fake_count = len(binary_preds) - real_count
    
    # Apply majority voting rule
    if real_count >= fake_count:
        return "Real", real_count / len(binary_preds)
    else:
        return "Fake", fake_count / len(binary_preds)
    
#Evaluation metrics
def evaluate_model(model, test_X, test_y):
    """Evaluate model performance"""
    model_v1, model_v2 = create_custom_inception_model()
    
    y_pred = []
    
    for i, faces in enumerate(test_X):
        if len(faces) >= 20:  # Minimum sequence length
            # Extract features
            features = extract_features(faces[:20], model_v1, model_v2)
            pred = model.predict(np.expand_dims(features, axis=0))[0][0]
            y_pred.append(1 if pred >= 0.5 else 0)
    
    # Calculate metrics
    y_true = test_y[:len(y_pred)]
    
    # Accuracy
    accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
    
    # True Positive Rate (TPR)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    actual_positives = sum(1 for a in y_true if a == 1)
    tpr = tp / actual_positives if actual_positives > 0 else 0
    
    # True Negative Rate (TNR)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    actual_negatives = sum(1 for a in y_true if a == 0)
    tnr = tn / actual_negatives if actual_negatives > 0 else 0
    
    return {
        'accuracy': accuracy,
        'tpr': tpr,
        'tnr': tnr
    }

#Complete implementation workflow
def main():
    """Main function to execute the complete workflow"""
    
    # 1. Prepare datasets
    print("Preparing real videos dataset...")
    real_dataset_path = "path/to/real/videos"
    fake_dataset_path = "path/to/fake/videos"
    
    # Check if preprocessed data exists to avoid reprocessing
    if os.path.exists('preprocessed_data.npz'):
        print("Loading preprocessed data...")
        data = np.load('preprocessed_data.npz', allow_pickle=True)
        X = data['X']
        y = data['y']
    else:
        print("Extracting features from real videos...")
        real_X, real_y = prepare_dataset(real_dataset_path, is_real=True)
        
        print("Extracting features from fake videos...")
        fake_X, fake_y = prepare_dataset(fake_dataset_path, is_real=False)
        
        # Combine datasets
        X = real_X + fake_X
        y = real_y + fake_y
        
        # Save preprocessed data
        np.savez('preprocessed_data.npz', X=X, y=y)
    
    # 2. Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {len(X_train)} videos")
    print(f"Validation set: {len(X_val)} videos")
    print(f"Testing set: {len(X_test)} videos")
    
    # 3. Train the model
    print("Training the HODFF-DD model...")
    hodff_dd_model, history, model_v1, model_v2 = train_model(
        X_train, y_train, X_val, y_val, epochs=50, batch_size=10
    )
    
    # 4. Save the trained model
    print("Saving the trained model...")
    hodff_dd_model.save('hodff_dd_model.h5')
    model_v1.save('inception_resnet_v1_model.h5')
    model_v2.save('inception_resnet_v2_model.h5')
    
    # 5. Evaluate the model on test set
    print("Evaluating the model on test set...")
    metrics = evaluate_model(hodff_dd_model, model_v1, model_v2, X_test, y_test)
    
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"True Positive Rate (TPR): {metrics['tpr']:.4f}")
    print(f"True Negative Rate (TNR): {metrics['tnr']:.4f}")
    
    # 6. Test on individual videos
    if len(sys.argv) > 1:
        test_video_path = sys.argv[1]
        print(f"Testing on video: {test_video_path}")
        result, confidence = classify_video(hodff_dd_model, model_v1, model_v2, test_video_path)
        print(f"Video classification: {result} with confidence {confidence:.4f}")

# 13. Cross-Set Evaluation
def perform_cross_set_evaluation(dataset_paths, subset_names):
    """Perform cross-set evaluation as described in the paper"""
    results = {}
    
    for i, test_subset in enumerate(subset_names):
        print(f"Cross-set evaluation: Testing on {test_subset}")
        
        # Prepare training set (all subsets except the test one)
        train_X = []
        train_y = []
        
        for j, train_subset in enumerate(subset_names):
            if train_subset != test_subset:
                # Load data from this subset
                subset_path = dataset_paths[j]
                X, y = prepare_dataset(subset_path, is_real=(train_subset == 'Pristine'))
                train_X.extend(X)
                train_y.extend(y)
        
        # Prepare test set
        test_path = dataset_paths[i]
        test_X, test_y = prepare_dataset(test_path, is_real=(test_subset == 'Pristine'))
        
        # Split training data for validation
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
        
        # Train model
        hodff_dd_model, _, model_v1, model_v2 = train_model(
            train_X, train_y, val_X, val_y, epochs=30, batch_size=10
        )
        
        # Evaluate on test set
        metrics = evaluate_model(hodff_dd_model, model_v1, model_v2, test_X, test_y)
        results[test_subset] = metrics['accuracy']
        
        print(f"Accuracy on {test_subset}: {metrics['accuracy']:.4f}")
    
    return results

# 14. Close-Set Evaluation
def perform_close_set_evaluation(dataset_paths, subset_names):
    """Perform close-set evaluation as described in the paper"""
    # Combine all datasets for training
    all_X = []
    all_y = []
    
    for i, subset in enumerate(subset_names):
        subset_path = dataset_paths[i]
        X, y = prepare_dataset(subset_path, is_real=(subset == 'Pristine'))
        all_X.extend(X)
        all_y.extend(y)
    
    # Split for training and validation
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.3, random_state=42)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    
    # Train model on combined dataset
    hodff_dd_model, _, model_v1, model_v2 = train_model(
        train_X, train_y, val_X, val_y, epochs=50, batch_size=10
    )
    
    # Evaluate on each subset separately
    results = {}
    
    for i, subset in enumerate(subset_names):
        subset_path = dataset_paths[i]
        subset_X, subset_y = prepare_dataset(subset_path, is_real=(subset == 'Pristine'))
        
        # Use only test portion
        _, subset_test_X, _, subset_test_y = train_test_split(subset_X, subset_y, test_size=0.3, random_state=42)
        
        metrics = evaluate_model(hodff_dd_model, model_v1, model_v2, subset_test_X, subset_test_y)
        results[subset] = metrics['accuracy']
        
        print(f"Close-set accuracy on {subset}: {metrics['accuracy']:.4f}")
    
    return results

# 15. Handling Class Imbalance for FakeAVCeleb Dataset
def handle_class_imbalance(real_X, real_y, fake_X, fake_y, augmentation_factor=5):
    """Handle class imbalance using data augmentation as mentioned for FakeAVCeleb dataset"""
    print(f"Before augmentation - Real videos: {len(real_X)}, Fake videos: {len(fake_X)}")
    
    augmented_real_X = []
    augmented_real_y = []
    
    for i, faces in enumerate(real_X):
        augmented_real_X.append(faces)  # Original faces
        augmented_real_y.append(real_y[i])
        
        if len(faces) >= 20:  # Minimum sequence length
            # Apply augmentations
            for _ in range(augmentation_factor):
                # Random horizontal flip
                if np.random.rand() > 0.5:
                    flipped_faces = [cv2.flip(face, 1) for face in faces]
                    augmented_real_X.append(np.array(flipped_faces))
                    augmented_real_y.append(real_y[i])
                
                # Random brightness adjustment
                brightness_factor = np.random.uniform(0.8, 1.2)
                bright_faces = [np.clip(face * brightness_factor, 0, 255).astype(np.uint8) for face in faces]
                augmented_real_X.append(np.array(bright_faces))
                augmented_real_y.append(real_y[i])
                
                # Random rotation (small angles)
                angle = np.random.uniform(-15, 15)
                rotated_faces = []
                for face in faces:
                    h, w = face.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    rotated = cv2.warpAffine(face, M, (w, h))
                    rotated_faces.append(rotated)
                augmented_real_X.append(np.array(rotated_faces))
                augmented_real_y.append(real_y[i])
    
    print(f"After augmentation - Real videos: {len(augmented_real_X)}, Fake videos: {len(fake_X)}")
    
    # Combine augmented real data with fake data
    X = augmented_real_X + fake_X
    y = augmented_real_y + fake_y
    
    return X, y

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()