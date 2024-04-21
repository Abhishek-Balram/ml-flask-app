# This file is for testing making requests to the Flask API endpoint
import requests

url = 'http://127.0.0.1:9696/'
r = requests.post(url, json = vehicle_data)
print(r.text.strip())