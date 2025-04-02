from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define activities
ACTIVITIES = ['Walking', 'Running', 'Sitting', 'Standing', 'Laying']

# Sample training data for each activity
SAMPLE_TRAINING_DATA = {
    'Walking': [
        [0.69, 10.8, -2.03],
        [0.75, 9.12, -1.8],
        [0.72, 9.5, -1.9],
        [0.68, 9.8, -2.0],
        [0.71, 9.3, -1.85]
    ],
    'Running': [
        [2.15, 15.2, -3.45],
        [2.85, 15.0, -3.0],
        [2.5, 14.8, -3.2],
        [2.7, 15.1, -3.1],
        [2.3, 14.9, -3.3]
    ],
    'Sitting': [
        [0.12, 1.5, 9.65],
        [0.12, 1.4, 9.65],
        [0.11, 1.45, 9.66],
        [0.13, 1.42, 9.64],
        [0.12, 1.43, 9.65]
    ],
    'Standing': [
        [0.15, 0.20, 9.81],
        [0.14, 0.20, 9.81],
        [0.15, 0.19, 9.82],
        [0.14, 0.21, 9.80],
        [0.15, 0.20, 9.81]
    ],
    'Laying': [
        [0.05, 9.81, 0.08],
        [0.05, 9.81, 0.07],
        [0.06, 9.80, 0.08],
        [0.05, 9.82, 0.07],
        [0.06, 9.81, 0.08]
    ]
}

def create_model():
    """Create and compile the RNN model"""
    model = Sequential([
        SimpleRNN(64, input_shape=(5, 3), return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(len(ACTIVITIES), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_training_data():
    """Create training data from sample patterns"""
    X = []
    y = []
    
    for activity_idx, (activity, base_data) in enumerate(SAMPLE_TRAINING_DATA.items()):
        # Use base data
        X.append(base_data)
        y.append(activity_idx)
        
        # Add variations with controlled noise
        for _ in range(30):  # Increased variations
            variation = np.array(base_data) + np.random.normal(
                0, 
                0.1 if activity in ['Sitting', 'Standing', 'Laying'] else 0.3,  # Less noise for stationary activities
                size=np.array(base_data).shape
            )
            X.append(variation)
            y.append(activity_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert labels to one-hot encoding
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(ACTIVITIES))
    
    return X, y_one_hot

def analyze_sequence(data):
    """Analyze a sequence of sensor readings to determine activity patterns"""
    # Convert to numpy array
    data = np.array(data)
    
    # Calculate basic statistics
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    max_vals = np.max(data, axis=0)
    min_vals = np.min(data, axis=0)
    range_vals = max_vals - min_vals
    
    # Calculate additional features
    gravity_vector = np.mean(data, axis=0)
    movement_intensity = np.mean(np.abs(data), axis=0)
    total_variation = np.sum(np.abs(np.diff(data, axis=0)), axis=0)
    
    # Calculate pattern match scores (0 to 1) instead of binary matches
    pattern_scores = {
        'Walking': (
            min(1.0, max(0.0, (
                (np.mean(std) > 0.8 and np.mean(std) < 5.0) * 0.4 +  # Moderate variation
                (np.any((mean > 0.5) & (mean < 10.0))) * 0.3 +      # Reasonable mean values
                (range_vals[1] > 2.0) * 0.3                         # Y-axis movement
            )))
        ),
        'Running': (
            min(1.0, max(0.0, (
                (np.mean(std) > 3.0) * 0.4 +                        # High variation
                (np.any(np.abs(data) > 10.0)) * 0.3 +              # High acceleration
                (range_vals[1] > 4.0) * 0.3                         # Large Y-axis movement
            )))
        ),
        'Sitting': (
            min(1.0, max(0.0, (
                (np.all(std < 0.4)) * 0.3 +                         # Low movement
                (abs(gravity_vector[2]) > 9.3) * 0.3 +              # Strong Z-axis gravity
                (1.0 < abs(gravity_vector[1]) < 2.0) * 0.4          # Forward tilt
            )))
        ),
        'Standing': (
            min(1.0, max(0.0, (
                (np.all(std < 0.3)) * 0.3 +                         # Very low movement
                (abs(gravity_vector[2]) > 9.5) * 0.4 +              # Very strong Z-axis gravity
                (abs(gravity_vector[1]) < 0.7) * 0.3                # Minimal tilt
            )))
        ),
        'Laying': (
            min(1.0, max(0.0, (
                (np.all(std < 0.3)) * 0.3 +                         # Low movement
                (abs(gravity_vector[1]) > 9.3) * 0.4 +              # Strong Y-axis gravity
                (abs(gravity_vector[2]) < 0.7) * 0.3                # Minimal Z-axis gravity
            )))
        )
    }
    
    # Calculate activity-specific metrics
    metrics = {
        'movement_intensity': movement_intensity.tolist(),
        'total_variation': total_variation.tolist(),
        'gravity_vector': gravity_vector.tolist(),
        'std_deviation': std.tolist(),
        'range': range_vals.tolist()
    }
    
    return pattern_scores, metrics

def preprocess_input_data(data):
    """Preprocess input data for prediction"""
    data = np.array(data)
    
    # If we have fewer than 5 points, pad with the last value
    if len(data) < 5:
        padding = np.tile(data[-1], (5 - len(data), 1))
        data = np.vstack([data, padding])
    
    # If we have more than 5 points, create sliding windows
    if len(data) > 5:
        windows = []
        for i in range(len(data) - 4):
            windows.append(data[i:i+5])
        data = np.array(windows)
    else:
        data = data.reshape(1, 5, 3)
    
    return data

# Initialize model
model_path = os.path.join(os.path.dirname(__file__), 'har_rnn_model.h5')
try:
    model = load_model(model_path)
    logger.info("Loaded existing model")
except (OSError, IOError):
    logger.info("Model file not found, creating new model")
    X, y = create_training_data()
    model = create_model()
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    model.save(model_path)
    logger.info("Created and saved new model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        logger.debug(f"Received request data: {data}")
        
        sensor_data = data.get('sensor_data', [])
        logger.debug(f"Extracted sensor data: {sensor_data}")
        
        if not sensor_data:
            return jsonify({'error': 'No sensor data provided'}), 400
        
        # Analyze patterns in the data
        pattern_scores, metrics = analyze_sequence(sensor_data)
        
        # Preprocess the data into windows
        processed_windows = preprocess_input_data(sensor_data)
        
        # Make predictions for each window
        predictions = []
        for window in processed_windows:
            window_reshaped = window.reshape(1, *window.shape)
            pred = model.predict(window_reshaped, verbose=0)
            predictions.append(pred[0])
        
        # Average the predictions
        avg_prediction = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_prediction)
        base_confidence = float(avg_prediction[predicted_class])
        
        # Calculate final confidence using both model prediction and pattern analysis
        predicted_activity = ACTIVITIES[predicted_class]
        pattern_confidence = pattern_scores[predicted_activity]
        
        # Weighted average of model confidence and pattern confidence
        final_confidence = (base_confidence * 0.6 + pattern_confidence * 0.4)
        
        # Adjust confidence based on number of samples
        if len(sensor_data) < 5:
            final_confidence *= 0.8  # Reduce confidence for very few samples
        
        # Add uncertainty for very similar activities
        if predicted_activity in ['Sitting', 'Standing']:
            # Check if the other stationary activity has similar confidence
            other_activity = 'Standing' if predicted_activity == 'Sitting' else 'Sitting'
            other_idx = ACTIVITIES.index(other_activity)
            other_confidence = float(avg_prediction[other_idx])
            
            if abs(other_confidence - base_confidence) < 0.2:
                final_confidence *= 0.8  # Reduce confidence when activities are ambiguous
        
        # Calculate confidence for all activities
        all_confidences = {}
        for i, activity in enumerate(ACTIVITIES):
            all_confidences[activity] = {
                'model_confidence': float(avg_prediction[i]),
                'pattern_score': pattern_scores[activity],
                'final_confidence': float(avg_prediction[i]) * 0.6 + pattern_scores[activity] * 0.4
            }
        
        result = {
            'activity': predicted_activity,
            'confidence': min(0.99, final_confidence),  # Cap at 99% to acknowledge uncertainty
            'num_samples': len(sensor_data),
            'pattern_score': pattern_confidence,
            'metrics': metrics,
            'all_confidences': all_confidences
        }
        logger.info(f"Prediction result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 