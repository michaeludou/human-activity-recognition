from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import pandas as pd
import os
import logging
import sys
import gc
import time
from functools import wraps
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set timeout for model operations (in seconds)
MODEL_TIMEOUT = 30

def timeout_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

# Define activities
ACTIVITIES = ['Walking', 'Running', 'Sitting', 'Standing', 'Laying']

# Sample training data for each activity
SAMPLE_TRAINING_DATA = {
    'Walking': [
        [0.69, 10.8, -2.03],
        [6.85, 7.44, -0.5],
        [0.93, 5.63, -0.5],
        [1.25, 8.45, -1.2],
        [0.75, 9.12, -1.8]
    ],
    'Running': [
        [2.15, 15.2, -3.45],
        [8.75, 12.3, -2.1],
        [3.25, 14.8, -2.8],
        [4.12, 13.5, -3.2],
        [2.85, 15.0, -3.0]
    ],
    'Sitting': [
        [0.12, 1.5, 9.65],
        [0.10, 1.4, 9.64],
        [0.11, 1.45, 9.63],
        [0.13, 1.48, 9.65],
        [0.12, 1.47, 9.64]
    ],
    'Standing': [
        [0.15, 0.20, 9.81],
        [0.14, 0.18, 9.82],
        [0.16, 0.19, 9.81],
        [0.15, 0.21, 9.80],
        [0.14, 0.20, 9.81]
    ],
    'Laying': [
        [0.05, 9.81, 0.08],
        [0.06, 9.82, 0.07],
        [0.05, 9.81, 0.09],
        [0.06, 9.80, 0.08],
        [0.05, 9.81, 0.07]
    ]
}

# Global model variable
model = None

def cleanup_memory():
    """Clean up memory to prevent leaks"""
    try:
        gc.collect()
        tf.keras.backend.clear_session()
        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.error(f"Error during memory cleanup: {str(e)}")
        logger.error(traceback.format_exc())

def create_model():
    """Create and compile an optimized RNN model"""
    try:
        logger.info("Creating optimized model...")
        model = Sequential([
            # Input layer with batch normalization
            SimpleRNN(64, input_shape=(None, 3), return_sequences=True),
            Dropout(0.2),
            # Second RNN layer
            SimpleRNN(32),
            Dropout(0.2),
            # Dense layers for classification
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(len(ACTIVITIES), activation='softmax')
        ])
        
        # Use a lower learning rate for better stability
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model created successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_training_data():
    """Create optimized training data with clear distinctions between activities"""
    try:
        logger.info("Creating training data...")
        X = []
        y = []
        
        # Define characteristic patterns for each activity
        activity_patterns = {
            'Walking': {
                'y_range': (5.0, 12.0),    # Moderate vertical movement
                'z_range': (-3.0, -0.5),   # Forward tilt
                'noise': 0.2
            },
            'Running': {
                'y_range': (12.0, 18.0),   # High vertical movement
                'z_range': (-4.0, -2.0),   # Forward tilt
                'noise': 0.3
            },
            'Sitting': {
                'y_range': (0.5, 2.0),     # Low vertical movement
                'z_range': (9.5, 9.8),     # Upward orientation
                'noise': 0.05
            },
            'Standing': {
                'y_range': (0.0, 0.3),     # Minimal movement
                'z_range': (9.7, 9.9),     # Straight upward
                'noise': 0.02
            },
            'Laying': {
                'y_range': (9.7, 9.9),     # Sideways gravity
                'z_range': (-0.2, 0.2),    # Minimal vertical
                'noise': 0.02
            }
        }
        
        for activity_idx, (activity, base_data) in enumerate(SAMPLE_TRAINING_DATA.items()):
            pattern = activity_patterns[activity]
            
            # Generate variations
            num_variations = 50  # Reduced number but more distinct patterns
            
            for _ in range(num_variations):
                # Create sequence of 5-10 measurements
                sequence_length = np.random.randint(5, 11)
                variation = []
                
                for _ in range(sequence_length):
                    # Generate characteristic pattern for this activity
                    x = np.random.normal(0, pattern['noise'])
                    y = np.random.uniform(*pattern['y_range'])
                    z = np.random.uniform(*pattern['z_range'])
                    
                    # Add activity-specific patterns
                    if activity == 'Walking':
                        # Add rhythmic pattern
                        x += np.sin(_ * 0.5) * 0.5
                        y += np.sin(_ * 0.5) * 2.0
                    elif activity == 'Running':
                        # Add stronger rhythmic pattern
                        x += np.sin(_ * 0.8) * 1.0
                        y += np.sin(_ * 0.8) * 3.0
                    
                    variation.append([x, y, z])
                
                X.append(variation)
                y.append(activity_idx)
        
        X = np.array(X)
        y = np.array(y)
        y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(ACTIVITIES))
        
        logger.info(f"Training data created: X shape {X.shape}, y shape {y_one_hot.shape}")
        return X, y_one_hot
    except Exception as e:
        logger.error(f"Error creating training data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@timeout_handler
def initialize_model():
    """Initialize or load the model with improved training"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'har_rnn_model.keras')
        logger.info(f"Attempting to load model from {model_path}")
        
        # Try to load existing model first
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                logger.info("Successfully loaded existing model")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {str(e)}")
                logger.info("Creating new model...")
        
        # If loading fails or model doesn't exist, create new one
        logger.info("Creating new model with improved training...")
        X, y = create_training_data()
        model = create_model()
        
        # Train with early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X, y,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1,
            shuffle=True
        )
        
        # Save the model
        model.save(model_path)
        logger.info(f"Created and saved model to {model_path}")
        
        # Log training results
        val_accuracy = history.history['val_accuracy'][-1]
        logger.info(f"Final validation accuracy: {val_accuracy:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def index():
    return render_template('index.html')

def calibrate_confidence(prediction, sensor_data):
    """Calibrate the confidence score based on multiple factors"""
    try:
        # Get the raw confidence from the model
        raw_confidence = float(prediction[np.argmax(prediction)])
        
        # Calculate pattern match score
        pattern_score = calculate_pattern_match_score(sensor_data, ACTIVITIES[np.argmax(prediction)])
        
        # Calculate data quality score
        data_quality = min(1.0, len(sensor_data) / 10.0)  # Normalize to 1.0 for 10+ samples
        
        # Calculate signal quality score
        signal_quality = calculate_signal_quality(sensor_data)
        
        # Combine scores with weights
        calibrated_confidence = (
            0.4 * raw_confidence +  # Model prediction
            0.3 * pattern_score +   # Pattern matching
            0.2 * data_quality +    # Data quality
            0.1 * signal_quality    # Signal quality
        )
        
        return min(1.0, calibrated_confidence)  # Cap at 1.0
    except Exception as e:
        logger.error(f"Error in confidence calibration: {str(e)}")
        return raw_confidence  # Fallback to raw confidence

def calculate_pattern_match_score(sensor_data, activity):
    """Calculate how well the sensor data matches the expected pattern for an activity"""
    try:
        # Calculate basic statistics
        avg_vertical = np.mean(np.abs(sensor_data[:, 1]))
        avg_horizontal = np.mean(np.abs(sensor_data[:, 0]))
        avg_gravity = np.mean(sensor_data[:, 2])
        
        # Define expected patterns for each activity
        patterns = {
            'Walking': {
                'y_range': (5.0, 12.0),    # Moderate vertical movement
                'z_range': (-3.0, -0.5),   # Forward tilt
                'x_range': (0.0, 2.0)      # Side-to-side movement
            },
            'Running': {
                'y_range': (12.0, 18.0),   # High vertical movement
                'z_range': (-4.0, -2.0),   # Strong forward tilt
                'x_range': (0.0, 3.0)      # More side-to-side movement
            },
            'Sitting': {
                'y_range': (0.5, 2.0),     # Low vertical movement
                'z_range': (9.5, 9.8),     # Upward orientation
                'x_range': (0.0, 0.5)      # Minimal side movement
            },
            'Standing': {
                'y_range': (0.0, 0.3),     # Minimal movement
                'z_range': (9.7, 9.9),     # Straight upward
                'x_range': (0.0, 0.3)      # Minimal side movement
            },
            'Laying': {
                'y_range': (9.7, 9.9),     # Sideways gravity
                'z_range': (-0.2, 0.2),    # Minimal vertical
                'x_range': (0.0, 0.3)      # Minimal side movement
            }
        }
        
        pattern = patterns[activity]
        
        # Calculate match scores for each axis
        y_score = 1.0 - min(1.0, abs(avg_vertical - np.mean(pattern['y_range'])) / 5.0)
        z_score = 1.0 - min(1.0, abs(avg_gravity - np.mean(pattern['z_range'])) / 5.0)
        x_score = 1.0 - min(1.0, abs(avg_horizontal - np.mean(pattern['x_range'])) / 2.0)
        
        # Combine scores with weights
        pattern_score = 0.4 * y_score + 0.4 * z_score + 0.2 * x_score
        
        return pattern_score
    except Exception as e:
        logger.error(f"Error in pattern match calculation: {str(e)}")
        return 0.5  # Return neutral score on error

def calculate_signal_quality(sensor_data):
    """Calculate the quality of the sensor signal"""
    try:
        # Calculate signal-to-noise ratio
        signal_power = np.mean(np.square(sensor_data))
        noise_power = np.var(sensor_data)
        snr = signal_power / (noise_power + 1e-6)  # Add small constant to avoid division by zero
        
        # Normalize SNR to 0-1 range
        snr_score = min(1.0, snr / 10.0)
        
        # Calculate signal stability
        stability = 1.0 - min(1.0, np.std(sensor_data) / 5.0)
        
        # Combine scores
        signal_quality = 0.6 * snr_score + 0.4 * stability
        
        return signal_quality
    except Exception as e:
        logger.error(f"Error in signal quality calculation: {str(e)}")
        return 0.5  # Return neutral score on error

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received request data: {data}")
        
        if not data or 'sensor_data' not in data:
            logger.error("Invalid request: No sensor data provided")
            return jsonify({'error': 'No sensor data provided'}), 400

        sensor_data = np.array(data['sensor_data'])
        logger.info(f"Converted sensor data to numpy array with shape: {sensor_data.shape}")
        
        if len(sensor_data) == 0:
            logger.error("Empty sensor data array")
            return jsonify({'error': 'Empty sensor data'}), 400

        # Initialize model if not already initialized
        global model
        if model is None:
            logger.info("Model not initialized, initializing now...")
            if not initialize_model():
                logger.error("Failed to initialize model")
                return jsonify({'error': 'Failed to initialize model'}), 500

        try:
            # Calculate basic metrics
            avg_acceleration = {
                'x': float(np.mean(sensor_data[:, 0])),
                'y': float(np.mean(sensor_data[:, 1])),
                'z': float(np.mean(sensor_data[:, 2]))
            }
            
            peak_acceleration = {
                'x': float(np.max(np.abs(sensor_data[:, 0]))),
                'y': float(np.max(np.abs(sensor_data[:, 1]))),
                'z': float(np.max(np.abs(sensor_data[:, 2])))
            }
            
            variability = {
                'x': float(np.std(sensor_data[:, 0])),
                'y': float(np.std(sensor_data[:, 1])),
                'z': float(np.std(sensor_data[:, 2]))
            }

            # Calculate movement metrics
            total_movement = np.sqrt(np.sum(sensor_data**2, axis=1))
            movement_intensity = float(np.mean(total_movement))
            vertical_movement = float(np.mean(np.abs(sensor_data[:, 1])))
            horizontal_movement = float(np.mean(np.sqrt(sensor_data[:, 0]**2 + sensor_data[:, 2]**2)))
            gravity_alignment = float(np.mean(np.abs(sensor_data[:, 2] - 9.81)))

            # Calculate pattern match scores for each activity
            pattern_scores = {
                'Walking': calculate_pattern_match_score(sensor_data, 'Walking'),
                'Running': calculate_pattern_match_score(sensor_data, 'Running'),
                'Sitting': calculate_pattern_match_score(sensor_data, 'Sitting'),
                'Standing': calculate_pattern_match_score(sensor_data, 'Standing'),
                'Laying': calculate_pattern_match_score(sensor_data, 'Laying')
            }

            # Get model prediction
            logger.info("Making model prediction...")
            prediction = model.predict(sensor_data.reshape(1, -1, 3))
            predicted_activity = ACTIVITIES[np.argmax(prediction)]
            logger.info(f"Predicted activity: {predicted_activity}")
            
            # Calculate confidence and probabilities
            raw_scores = {}
            for activity in ACTIVITIES:
                model_score = prediction[0][ACTIVITIES.index(activity)]
                pattern_score = pattern_scores[activity]
                raw_scores[activity] = (model_score * 0.7) + (pattern_score * 0.3)
            
            # Normalize scores to get probabilities
            total_score = sum(raw_scores.values())
            probabilities = {activity: score/total_score for activity, score in raw_scores.items()}
            
            # Calculate overall confidence
            confidence = probabilities[predicted_activity]
            
            # Get pattern match confidence level
            pattern_confidence = 'High' if pattern_scores[predicted_activity] > 0.8 else \
                               'Medium' if pattern_scores[predicted_activity] > 0.6 else 'Low'

            # Calculate signal quality based on confidence (as percentage)
            confidence_percentage = confidence * 100
            signal_quality = 'Good' if confidence_percentage >= 80 else \
                           'Fair' if confidence_percentage >= 60 else 'Poor'

            # Calculate data reliability based on pattern match (as percentage)
            pattern_match_percentage = pattern_scores[predicted_activity] * 100
            data_reliability = 'High' if pattern_match_percentage >= 90 else \
                             'Medium' if pattern_match_percentage >= 70 else 'Low'

            response = {
                'prediction': predicted_activity,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in probabilities.items()},
                'metrics': {
                    'pattern_scores': {k: float(v) for k, v in pattern_scores.items()},
                    'acceleration': {
                        'average': {
                            'x': f"{avg_acceleration['x']:.2f}",
                            'y': f"{avg_acceleration['y']:.2f}",
                            'z': f"{avg_acceleration['z']:.2f}"
                        },
                        'peak': {
                            'x': f"{peak_acceleration['x']:.2f}",
                            'y': f"{peak_acceleration['y']:.2f}",
                            'z': f"{peak_acceleration['z']:.2f}"
                        },
                        'variability': {
                            'x': f"{variability['x']:.2f}",
                            'y': f"{variability['y']:.2f}",
                            'z': f"{variability['z']:.2f}"
                        }
                    },
                    'movement_metrics': {
                        'intensity': f"{movement_intensity:.2f}",
                        'vertical_movement': f"{vertical_movement:.2f}",
                        'horizontal_movement': f"{horizontal_movement:.2f}",
                        'gravity_alignment': f"{gravity_alignment:.2f}"
                    }
                },
                'activity_info': {
                    'pattern_match': {
                        'confidence': pattern_confidence,
                        'description': describe_pattern_match(sensor_data, predicted_activity)
                    }
                }
            }

            logger.info(f"Successfully generated response: {response}")
            return jsonify(response)

        except Exception as e:
            logger.error(f"Error during prediction processing: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing prediction: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error handling request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def describe_pattern_match(sensor_data, activity):
    """Generate a description of how well the sensor data matches the activity pattern"""
    pattern_score = calculate_pattern_match_score(sensor_data, activity)
    
    if pattern_score > 0.8:
        return f"The sensor data strongly matches the expected pattern for {activity.lower()}."
    elif pattern_score > 0.6:
        return f"The sensor data shows a good match with the expected pattern for {activity.lower()}."
    elif pattern_score > 0.4:
        return f"The sensor data shows some characteristics of {activity.lower()}, but with some variations."
    else:
        return f"The sensor data shows significant differences from the expected pattern for {activity.lower()}."

if __name__ == '__main__':
    app.debug = True  # Enable debug mode
    app.jinja_env.auto_reload = True  # Enable template auto-reload
    app.config['TEMPLATES_AUTO_RELOAD'] = True  # Force template reloading
    app.run(host='0.0.0.0', port=5000)
