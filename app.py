from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Activity labels
ACTIVITIES = ['Walking', 'Running', 'Sitting', 'Standing', 'Laying']

# Sample training data with more distinctive patterns
SAMPLE_TRAINING_DATA = {
    'Walking': [
        [0.69, 10.8, -2.03], [6.85, 7.44, -0.5], [0.93, 5.63, -0.5],
        [1.25, 8.45, -1.2], [0.75, 9.12, -1.8]
    ],
    'Running': [
        [2.15, 15.2, -3.45], [8.75, 12.3, -2.1], [3.25, 14.8, -2.8],
        [4.12, 13.5, -3.2], [2.85, 15.0, -3.0]
    ],
    'Sitting': [  # More distinctive sitting pattern (slight forward tilt)
        [0.12, 1.5, 9.65], [0.10, 1.4, 9.67], [0.11, 1.6, 9.64],
        [0.13, 1.5, 9.66], [0.12, 1.4, 9.65]
    ],
    'Standing': [  # Vertical position with minimal tilt
        [0.15, 0.20, 9.81], [0.14, 0.18, 9.82], [0.16, 0.19, 9.81],
        [0.15, 0.21, 9.80], [0.14, 0.20, 9.81]
    ],
    'Laying': [  # Horizontal position (z-axis close to 0, y-axis close to gravity)
        [0.05, 9.81, 0.08], [0.06, 9.82, 0.07], [0.05, 9.81, 0.09],
        [0.06, 9.80, 0.08], [0.05, 9.81, 0.07]
    ]
}

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

def create_training_data():
    """Create training data from sample data with more variations"""
    X = []
    y = []
    
    # Use fixed window size of 5 for consistency
    window_size = 5
    
    for activity_idx, (activity, data) in enumerate(SAMPLE_TRAINING_DATA.items()):
        # Use the base data as is
        X.append(data)
        y.append(activity_idx)
        
        # Add noise variations with controlled randomness
        base_data = np.array(data)
        for _ in range(30):  # Increased number of variations
            # Add small random variations that preserve the basic pattern
            if activity in ['Sitting', 'Standing', 'Laying']:
                # Less noise for stationary activities
                noise = np.random.normal(0, 0.05, base_data.shape)
            else:
                # More noise for dynamic activities
                noise = np.random.normal(0, 0.1, base_data.shape)
            
            noisy_data = base_data + noise
            X.append(noisy_data)
            y.append(activity_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode the labels
    y = to_categorical(y, num_classes=len(ACTIVITIES))
    
    logger.info(f"Created training data with shape: X={X.shape}, y={y.shape}")
    return X, y

def create_model():
    """Create a new RNN model"""
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
    
    logger.info("Created new model")
    return model

# Try to load existing model or create and train a new one
try:
    model = load_model('har_rnn_model.h5')
    logger.info("Loaded existing model")
except Exception as e:
    logger.info(f"Creating and training new model: {e}")
    model = create_model()
    
    # Create and prepare training data
    X_train, y_train = create_training_data()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Save the trained model
    model.save('har_rnn_model.h5')
    logger.info(f"Model training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}")

def preprocess_input_data(data):
    """Preprocess the input data for prediction"""
    try:
        # Convert to numpy array
        sensor_data = np.array(data, dtype=np.float32)
        logger.debug(f"Input data shape: {sensor_data.shape}")
        
        # Create sliding windows if we have more than 5 points
        windows = []
        if len(sensor_data) > 5:
            for i in range(0, len(sensor_data) - 4):
                windows.append(sensor_data[i:i+5])
        else:
            # Pad if less than 5 points
            if len(sensor_data) < 5:
                pad_length = 5 - len(sensor_data)
                sensor_data = np.pad(sensor_data, ((0, pad_length), (0, 0)), 'constant')
            windows = [sensor_data]
        
        windows = np.array(windows)
        logger.debug(f"Preprocessed data shape: {windows.shape}")
        return windows
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
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
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True) 