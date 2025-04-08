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
    """Create and compile a minimal RNN model"""
    try:
        logger.info("Creating new minimal model...")
        model = Sequential([
            SimpleRNN(4, input_shape=(None, 3)),  # Changed to accept variable length sequences
            Dense(4, activation='relu'),
            Dense(len(ACTIVITIES), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Minimal model created successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_training_data():
    """Create minimal training data from sample patterns"""
    try:
        logger.info("Creating minimal training data...")
        X = []
        y = []
        
        for activity_idx, (activity, base_data) in enumerate(SAMPLE_TRAINING_DATA.items()):
            # Use base data
            X.append(base_data)
            y.append(activity_idx)
            
            # Add minimal variations
            for _ in range(5):  # Very minimal variations
                variation = np.array(base_data) + np.random.normal(
                    0, 
                    0.1 if activity in ['Sitting', 'Standing', 'Laying'] else 0.3,
                    size=np.array(base_data).shape
                )
                X.append(variation)
                y.append(activity_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to one-hot encoding
        y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(ACTIVITIES))
        logger.info(f"Minimal training data created: X shape {X.shape}, y shape {y_one_hot.shape}")
        return X, y_one_hot
    except Exception as e:
        logger.error(f"Error creating training data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@timeout_handler
def initialize_model():
    """Initialize or load the model"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'har_rnn_model.keras')
        logger.info(f"Attempting to load model from {model_path}")
        
        if os.path.exists(model_path):
            logger.info("Loading existing model...")
            model = load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.info("Model file not found, creating new model")
            X, y = create_training_data()
            model = create_model()
            # Minimal training for faster initialization
            model.fit(X, y, epochs=5, batch_size=4, verbose=1)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            logger.info(f"Created and saved new model to {model_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with optimized processing"""
    try:
        # Get and validate input data
        data = request.get_json()
        logger.info(f"Received request data: {data}")

        if not data or "sensor_data" not in data:
            logger.error("Invalid request: No sensor data provided")
            return jsonify({"error": "Missing sensor_data"}), 400

        # Initialize model if not already initialized
        global model
        if model is None:
            logger.info("Model not initialized, initializing now...")
            if not initialize_model():
                return jsonify({"error": "Failed to initialize model"}), 500

        sensor_data = np.array(data["sensor_data"])
        logger.info(f"Received sensor data shape: {sensor_data.shape}")

        # Validate input shape
        if sensor_data.ndim != 2 or sensor_data.shape[1] != 3:
            logger.error(f"Invalid sensor data shape: {sensor_data.shape}")
            return jsonify({"error": "Expected 2D array with shape (timesteps, 3)"}), 400

        # Reshape for RNN (batch_size, timesteps, features)
        input_data = sensor_data.reshape((1, sensor_data.shape[0], 3))
        logger.info(f"Input reshaped to: {input_data.shape}")

        # Make prediction with timing
        start_time = time.time()
        try:
            prediction = model.predict(input_data, verbose=0)[0]
            prediction_time = time.time() - start_time
            logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
            logger.info(f"Raw prediction: {prediction}")

            # Clean up memory after prediction
            cleanup_memory()

            # Process results
            result = {
                'prediction': ACTIVITIES[np.argmax(prediction)],
                'confidence': float(np.max(prediction)),
                'probabilities': {
                    activity: float(prob) 
                    for activity, prob in zip(ACTIVITIES, prediction)
                },
                'prediction_time': prediction_time
            }
            
            logger.info(f"Final prediction: {result['prediction']} with confidence {result['confidence']}")
            return jsonify(result)

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            cleanup_memory()
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        cleanup_memory()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize model before starting the app
    if initialize_model():
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("Failed to initialize model")
