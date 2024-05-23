# from flask import Flask, request, render_template
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the trained models and encoders for safety model
# rf_model_safety = joblib.load('model.pkl')
# label_encoders_safety = joblib.load('label_encoders.pkl')

# # Load the trained model for efficiency
# lr_model_efficiency = joblib.load('lr_model_efficiency.pkl')

# # Function to calculate combined score
# def calculate_combined_score(safety_score, efficiency_score, safety_weight=0.8, efficiency_weight=0.2):
     
#     # Calculate combined score
#     combined_score = safety_weight * safety_score + efficiency_weight * efficiency_score
#     return combined_score



# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Extract data for the route
#         route = {}
#         route['airline'] = request.form['airline']
#         route['departure_airport'] = request.form['departure_airport']
#         route['arrival_airport'] = request.form['arrival_airport']
#         route['route_via'] = request.form['route_via']
#         route['weather'] = request.form['weather']
#         route['visibility'] = float(request.form['visibility'])
#         route['turbulence_intensity'] = float(request.form['turbulence_intensity'])
#         route['wind_shear'] = float(request.form['wind_shear'])
#         route['air_traffic_density'] = float(request.form['air_traffic_density'])
#         route['precipitation'] = float(request.form['precipitation'])
#         route['pilot_experience'] = float(request.form['pilot_experience'])
#         route['forecast_accuracy'] = float(request.form['forecast_accuracy'])
#         route['maintenance_history'] = request.form['maintenance_history']
#         route['fuel_consumption'] = float(request.form['fuel_consumption'])
#         route['air_traffic_congestion'] = float(request.form['air_traffic_congestion'])
#         route['no_step_climbs'] = float(request.form['no_step_climbs'])
#         route['aircraft_load_factor'] = float(request.form['aircraft_load_factor'])
#         route['projected_flight_time'] = float(request.form['projected_flight_time'])
#         route['maintenance_status'] = float(request.form['maintenance_status'])

#         # Predict safety score for the route
#         weather_encoded = label_encoders_safety['Weather'].transform([route['weather']])[0]
#         maintenance_history_encoded = label_encoders_safety['Maintenance History & Health Metrics  (MH)'].transform([route['maintenance_history']])[0]
#         features_safety = np.array([[weather_encoded, route['visibility'], route['turbulence_intensity'], route['wind_shear'], route['air_traffic_density'], 
#                                      route['precipitation'], route['pilot_experience'], route['forecast_accuracy'], maintenance_history_encoded]])
#         safety_score = rf_model_safety.predict(features_safety)[0]

#         # Predict efficiency score for the route
#         efficiency_score = lr_model_efficiency.predict([[route['fuel_consumption'], route['air_traffic_congestion'], route['no_step_climbs'], 
#                                                          route['aircraft_load_factor'], route['projected_flight_time'], 
#                                                          route['maintenance_status']]])[0]

#         # Calculate combined score
#         combined_score = calculate_combined_score(safety_score, efficiency_score)

#         # Render the result template with the route and scores
#         return render_template('result.html', route=route, safety_score=safety_score, efficiency_score=efficiency_score, combined_score=combined_score)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Load the trained models and encoders for safety model
rf_model_safety = joblib.load('model.pkl')
label_encoders_safety = joblib.load('label_encoders.pkl')

# Load the trained model for efficiency
lr_model_efficiency = joblib.load('lr_model_efficiency.pkl')

# Function to calculate combined score
def calculate_combined_score(safety_score, efficiency_score, safety_weight=0.8, efficiency_weight=0.2):
    # Calculate combined score
    combined_score = safety_weight * safety_score + efficiency_weight * efficiency_score
    return combined_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        route = data['route']

        # Predict safety score for the route
        weather_encoded = label_encoders_safety['Weather'].transform([route['weather']])[0]
        maintenance_history_encoded = label_encoders_safety['Maintenance History & Health Metrics  (MH)'].transform([route['maintenance_history']])[0]
        features_safety = np.array([[weather_encoded, route['visibility'], route['turbulence_intensity'], route['wind_shear'], route['air_traffic_density'], route['precipitation'], route['pilot_experience'], route['forecast_accuracy'], maintenance_history_encoded]])
        safety_score = rf_model_safety.predict(features_safety)[0]

        # Predict efficiency score for the route
        efficiency_score = lr_model_efficiency.predict([[route['fuel_consumption'], route['air_traffic_congestion'], route['no_step_climbs'], route['aircraft_load_factor'], route['projected_flight_time'], route['maintenance_status']]])[0]

        # Calculate combined score
        combined_score = calculate_combined_score(safety_score, efficiency_score)

        # Return the scores
        return jsonify({
            'safety_score': safety_score,
            'efficiency_score': efficiency_score,
            'combined_score': combined_score
        })

if __name__ == '__main__':
    app.run(debug=True)

