from django.shortcuts import render
import joblib
import numpy as np
import os

# Load model once
MODEL_PATH = os.path.join(os.path.dirname(__file__), "knn_house_price_model.pkl")
model = joblib.load(MODEL_PATH)

def home(request):
    predicted_price = None

    if request.method == "POST":
        # Get form data
        features = [
            float(request.POST["Square_Footage"]),
            float(request.POST["Num_Bedrooms"]),
            float(request.POST["Num_Bathrooms"]),
            float(request.POST["Year_Built"]),
            float(request.POST["Lot_Size"]),
            float(request.POST["Garage_Size"]),
            float(request.POST["Neighborhood_Quality"])
        ]

        # Predict
        predicted_price = model.predict([features])[0]

    return render(request, "predictor/index.html", {"predicted_price": predicted_price})
