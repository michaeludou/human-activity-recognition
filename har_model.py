import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess the data
def preprocess_data(df, sequence_length=50):
    # Convert activity labels to numerical values
    label_encoder = LabelEncoder()
    df['activity_encoded'] = label_encoder.fit_transform(df['activity'])
    
    # Get unique activities for later use
    activities = label_encoder.classes_
    
    # Group data by user and activity
    sequences = []
    labels = []
    
    for (user, activity), group in df.groupby(['user', 'activity']):
        # Extract sensor data
        sensor_data = group[['x-axis', 'y-axis', 'z-axis']].values
        
        # Create sequences
        for i in range(0, len(sensor_data) - sequence_length + 1, sequence_length):
            sequence = sensor_data[i:i + sequence_length]
            sequences.append(sequence)
            labels.append(group['activity_encoded'].iloc[0])
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)
    
    # One-hot encode labels
    y = to_categorical(y)
    
    return X, y, activities

# Build RNN model
def build_rnn_model(input_shape, num_classes):
    model = Sequential([
        SimpleRNN(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Load and preprocess data
    df = load_data('time_series_data_human_activities.csv')
    X, y, activities = preprocess_data(df)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build and train the model
    model = build_rnn_model(input_shape=(X.shape[1], X.shape[2]), 
                           num_classes=y.shape[1])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest accuracy: {test_acc:.4f}')
    
    # Save the model
    model.save('har_rnn_model.h5')
    print("Model saved as 'har_rnn_model.h5'")

if __name__ == "__main__":
    main() 