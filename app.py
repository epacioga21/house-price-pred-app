from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
from predict import predict_price

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Set a secret key for session management

# Load your trained model
model = joblib.load("model.joblib")

DEFAULTS = {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotFrontage": 65.0,
    "LotArea": 8450,
    "Street": "Pave",
    "Alley": "NA",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "CollgCr",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "OverallQual": 5,
    "OverallCond": 5,
    "YearBuilt": 1980,
    "YearRemodAdd": 1970,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0.0,
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "Unf",
    "BsmtFinSF1": 0,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 0,
    "TotalBsmtSF": 0,
    "Heating": "GasA",
    "HeatingQC": "TA",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": 856,
    "2ndFlrSF": 0,
    "LowQualFinSF": 0,
    "GrLivArea": 856,
    "BsmtFullBath": 0,
    "BsmtHalfBath": 0,
    "FullBath": 1,
    "HalfBath": 0,
    "Bedroom": 3,
    "Kitchen": 1,
    "KitchenQual": "TA",
    "TotRmsAbvGrd": 5,
    "Functional": "Typ",
    "Fireplaces": 0,
    "FireplaceQu": "NA",
    "GarageType": "Attchd",
    "GarageYrBlt": 1970,
    "GarageFinish": "Unf",
    "GarageCars": 1,
    "GarageArea": 200,
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "WoodDeckSF": 0,
    "OpenPorchSF": 0,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "PoolQC": "NA",
    "Fence": "NA",
    "MiscFeature": "NA",
    "MiscVal": 0,
    "MoSold": 6,
    "YrSold": 2008,
    "SaleType": "WD",
    "SaleCondition": "Normal"
}


@app.route('/')
def index():
    # Home page with a Start button
    return render_template('index.html')


@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Collect user input or use default values
        input_values = {}
        for key, default in DEFAULTS.items():
            value = request.form[key] if key in request.form else default

            if isinstance(default, int):
                input_values[key] = int(value)
            elif isinstance(default, float):
                input_values[key] = float(value)
            else:
                input_values[key] = value

        # Save input values to the session
        session['input_values'] = input_values

        # Redirect to the result page
        return redirect(url_for('result'))

    return render_template('form.html')


@app.route('/result')
def result():
    # Retrieve input values from the session
    input_values = session.get('input_values', DEFAULTS)

    # Make prediction
    prediction = predict_price(input_values)
    predicted_price = round(np.exp(prediction), 2)

    # Render result page with the prediction
    return render_template('result.html', price=predicted_price)


if __name__ == '__main__':
    app.run(debug=True)
