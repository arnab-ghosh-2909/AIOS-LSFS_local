import requests
import os

GEMINI_KEY = "AIzaSyBafqtWKTBH3QZ-EveVO1437zdfOUtJ03E"

url = "https://generativelanguage.googleapis.com/v1beta/models"
params = {
    "key": GEMINI_KEY
}

resp = requests.get(url, params=params)
if resp.status_code != 200:
    print("Error:", resp.status_code, resp.text)
else:
    data = resp.json()
    print("Available models:")
    for m in data.get("models", []):
        print(" •", m.get("name"))
