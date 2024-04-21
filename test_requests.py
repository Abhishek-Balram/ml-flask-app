# This file is for testing making requests to the Flask API endpoint
import requests

# Example vehicle data with 3 data points
vehicle_data = {
    'Cylinders': [4, 6, 8],
    'Displacement': [155.0, 160.0, 165.5],
    'Horsepower': [93.0, 130.0, 98.0],
    'Weight': [2500.0, 3150.0, 2600.0],
    'Acceleration': [15.0, 14.0, 16.0],
    'Model Year': [81, 80, 78],
    'Origin': [3, 2, 1]
}

url = 'http://127.0.0.1:9696/predict'
r = requests.post(url, json = vehicle_data)
print(r.text.strip())