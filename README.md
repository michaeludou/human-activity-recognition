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

## Deployment on Render

1. Create a GitHub repository and push your code
2. Go to [Render](https://render.com)
3. Click "New +" and select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - Name: Choose a name for your service
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
6. Click "Create Web Service"

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