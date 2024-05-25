# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import joblib
# import numpy as np
# import warnings

# # Suppress specific warnings from sklearn
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# app = Flask(__name__)
# # Set up CORS to allow requests from your frontend
# # CORS(app, resources={r"/predict": {"origins": "https://flight-route.vercel.app"}})
# CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# # Load the trained models and encoders for the safety model
# rf_model_safety = joblib.load('model.pkl')
# label_encoders_safety = joblib.load('label_encoders.pkl')

# # Load the trained model for efficiency
# lr_model_efficiency = joblib.load('lr_model_efficiency.pkl')

# # Function to calculate combined score
# def calculate_combined_score(safety_score, efficiency_score, safety_weight=0.8, efficiency_weight=0.2):
#     combined_score = safety_weight * safety_score + efficiency_weight * efficiency_score
#     return combined_score

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         print("Received request:", request.json)
#         data = request.json
#         route = data['route']

#         try:
#             # Predict safety score for the route
#             weather_encoded = label_encoders_safety['Weather'].transform([route['weather']])[0]
#             maintenance_history_encoded = label_encoders_safety['Maintenance History & Health Metrics  (MH)'].transform([route['maintenance_history']])[0]
#             features_safety = np.array([
#                 [
#                     weather_encoded, 
#                     route['visibility'], 
#                     route['turbulence_intensity'], 
#                     route['wind_shear'], 
#                     route['air_traffic_density'], 
#                     route['precipitation'], 
#                     route['pilot_experience'], 
#                     route['forecast_accuracy'], 
#                     maintenance_history_encoded
#                 ]
#             ])
#             safety_score = rf_model_safety.predict(features_safety)[0]

#             # Predict efficiency score for the route
#             features_efficiency = np.array([
#                 [
#                     route['fuel_consumption'], 
#                     route['air_traffic_congestion'], 
#                     route['no_step_climbs'], 
#                     route['aircraft_load_factor'], 
#                     route['projected_flight_time'], 
#                     route['maintenance_status']
#                 ]
#             ])
#             efficiency_score = lr_model_efficiency.predict(features_efficiency)[0]

#             # Calculate combined score
#             combined_score = calculate_combined_score(safety_score, efficiency_score)

#             # Return the scores
#             return jsonify({
#                 'safety_score': safety_score,
#                 'efficiency_score': efficiency_score,
#                 'combined_score': combined_score
#             })

#         except KeyError as e:
#             return jsonify({'error': f"Missing key in the input data: {str(e)}"}), 400

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import warnings

# Suppress specific warnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)
# Set up CORS to allow requests from your frontend
CORS(app, resources={r"/predict": {"origins": "https://flight-route.vercel.app"}})


# Load the trained models and encoders for the safety model
rf_model_safety = joblib.load('model.pkl')
label_encoders_safety = joblib.load('label_encoders.pkl')

# Load the trained model for efficiency
lr_model_efficiency = joblib.load('lr_model_efficiency.pkl')

# Load the trained models and encoders for the health model
rf_model_health = joblib.load('rf_model_health.pkl')
label_encoder_health = joblib.load('label_encoder_health.pkl')

# Function to calculate combined score
def calculate_combined_score(safety_score, efficiency_score, safety_weight=0.8, efficiency_weight=0.2):
    combined_score = safety_weight * safety_score + efficiency_weight * efficiency_score
    return combined_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("Received request:", request.json)
        data = request.json
        route = data['route']

        try:
            # Predict safety score for the route
            weather_encoded = label_encoders_safety['Weather'].transform([route['weather']])[0]
            maintenance_history_encoded = label_encoders_safety['Maintenance History & Health Metrics  (MH)'].transform([route['maintenance_history']])[0]
            features_safety = np.array([
                [
                    weather_encoded, 
                    route['visibility'], 
                    route['turbulence_intensity'], 
                    route['wind_shear'], 
                    route['air_traffic_density'], 
                    route['precipitation'], 
                    route['pilot_experience'], 
                    route['forecast_accuracy'], 
                    maintenance_history_encoded
                ]
            ])
            safety_score = rf_model_safety.predict(features_safety)[0]

            # Predict efficiency score for the route
            features_efficiency = np.array([
                [
                    route['fuel_consumption'], 
                    route['air_traffic_congestion'], 
                    route['no_step_climbs'], 
                    route['aircraft_load_factor'], 
                    route['projected_flight_time'], 
                    route['maintenance_status']
                ]
            ])
            efficiency_score = lr_model_efficiency.predict(features_efficiency)[0]

             # Predict health score for the route
            aircraft_model_encoded = label_encoder_health.transform([route['Aircraft_Model']])[0]
            features_health = np.array([
                [
                    aircraft_model_encoded, 
                    route['engine_temperature'], 
                    route['engine_vibration_levels'], 
                    route['oil_pressure'], 
                    route['hydraulic_system_pressure'], 
                    route['electrical_system_voltage'], 
                    route['oil_temperature']
                ]
            ])
            health_score = rf_model_health.predict(features_health)[0]
            # Calculate combined score
            combined_score = calculate_combined_score(safety_score, efficiency_score)

            # Return the scores
            return jsonify({
                'safety_score': safety_score,
                'efficiency_score': efficiency_score,
                'combined_score': combined_score,
                'health_score': health_score

            })

        except KeyError as e:
            return jsonify({'error': f"Missing key in the input data: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)

