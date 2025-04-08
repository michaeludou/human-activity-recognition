# Human Activity Recognition Web Application

A real-time human activity recognition system with an interactive web interface. The application uses accelerometer data to predict and analyze human activities such as walking, running, sitting, standing, and laying.

## Features

- **Real-time Activity Prediction**: Analyzes accelerometer data to predict human activities
- **Interactive Web Interface**: Modern, responsive design with real-time updates
- **Sample Activities**: Pre-configured sample data for testing different activities
- **Detailed Analysis**: 
  - Signal Quality Assessment
  - Data Reliability Metrics
  - Movement Analysis
  - Acceleration Analysis
  - Activity Probabilities

## Technologies Used

- **Backend**:
  - Python 3.x
  - Flask
  - NumPy
  - Scikit-learn (for ML model)

- **Frontend**:
  - HTML5
  - CSS3
  - JavaScript
  - Bootstrap 5.1.3
  - Font Awesome 6.0.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/human-activity-recognition.git
cd human-activity-recognition
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### 1. Testing Server Connection
- Click the "Test Server Connection" button to verify the server is running properly

### 2. Entering Sensor Data
- Use the "Add Data Point" button to add new sensor inputs
- Each data point requires X, Y, and Z axis accelerometer values
- Use the sample activity buttons for quick testing:
  - Walking
  - Running
  - Sitting
  - Standing
  - Laying

### 3. Making Predictions
- Click "Predict Activity" to analyze the sensor data
- View results in the following sections:
  - Predicted Activity
  - Confidence Score
  - Pattern Match
  - Analysis Details
  - Acceleration Analysis
  - Activity Probabilities

### 4. Understanding Results

#### Signal Quality Indicators
- **Good** (≥80%): High confidence in prediction
- **Fair** (≥60%): Moderate confidence
- **Poor** (<60%): Low confidence

#### Data Reliability
- **High** (≥90%): Strong pattern match
- **Medium** (≥70%): Moderate pattern match
- **Low** (<70%): Weak pattern match

#### Movement Analysis
- Intensity
- Vertical Movement
- Horizontal Movement
- Gravity Alignment

## Sample Data Ranges

| Activity | Sample Range |
|----------|-------------|
| Walking | [0.69, 10.8, -2.03] to [0.75, 9.12, -1.8] |
| Running | [2.15, 15.2, -3.45] to [2.85, 15.0, -3.0] |
| Sitting | [0.12, 1.5, 9.65] to [0.12, 1.4, 9.65] |
| Standing | [0.15, 0.20, 9.81] to [0.14, 0.20, 9.81] |
| Laying | [0.05, 9.81, 0.08] to [0.05, 9.81, 0.07] |

## Project Structure

```
human-activity-recognition/
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Main web interface
├── static/
│   └── model/            # ML model files
├── requirements.txt      # Python dependencies
└── README.md            # Documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Bootstrap for the responsive design framework
- Font Awesome for the icons
- The scientific community for activity recognition research 