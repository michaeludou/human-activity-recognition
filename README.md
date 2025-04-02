# Human Activity Recognition Web App

A Flask web application for human activity recognition using RNN and pattern analysis.

## Features

- Real-time activity prediction
- Multiple sensor data input
- Detailed confidence metrics
- Pattern analysis
- Sample data for different activities

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Deployment

The application is deployed on Render and can be accessed at: [https://har-web-app.onrender.com](https://har-web-app.onrender.com)

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `requirements.txt`: Python dependencies
- `Procfile`: Deployment configuration
- `har_rnn_model.h5`: Trained model (generated on first run)

## Notes

- The model will be automatically trained on first run if not present
- Sample data is provided for testing different activities
- Confidence scores are calculated based on both model prediction and pattern analysis 