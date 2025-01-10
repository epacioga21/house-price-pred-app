import joblib
import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

model = joblib.load("model.joblib")


def predict_price(features):
    """
    Functie care prezice pretul unei case cu datele introduse de utilizator
    :param features:
    :return pret:
    """
    features_df = pd.DataFrame([features])

    # Umplem valorile lipsa cu valoarea intalnita cel mai des doar dacă există valori
    for col in features_df.columns:
        if features_df[col].isna().sum() > 0:
            mode_value = features_df[col].mode()
            if not mode_value.empty:  # Verificăm dacă există valori în mod
                features_df[col].fillna(mode_value[0], inplace=True)

    # Crearea caracteristicilor noi
    features_df["HasAlley"] = 1 * (features_df["Alley"] != 'NA')
    features_df["HasBasement"] = 1 * (features_df["BsmtExposure"] != 'NA')
    features_df['TotalSF'] = features_df['TotalBsmtSF'] + features_df['1stFlrSF'] + features_df['2ndFlrSF']
    features_df['Total_Bathrooms'] = (features_df['FullBath'] + (0.5 * features_df['HalfBath']) +
                                      features_df['BsmtFullBath'] + (0.5 * features_df.get('BsmtHalfBath', 0)))
    features_df['Hasfireplace'] = features_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    features_df['HasOpenPorch'] = (features_df['OpenPorchSF'] == 0) * 1
    features_df['HasEnclosedPorch'] = (features_df['EnclosedPorch'] == 0) * 1
    features_df['Has3SsnPorch'] = (features_df['3SsnPorch'] == 0) * 1
    features_df['HasScreenPorch'] = (features_df['ScreenPorch'] == 0) * 1
    features_df["HasFence"] = 1 * (features_df["Fence"] != 'NA')
    features_df['Haspool'] = features_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    # Folosim functia get_dummies din pandas
    features_df = pd.get_dummies(features_df, drop_first=True)

    # Adaugam datele necesare modelului
    model_columns = model.feature_names_

    # Cream un DataFrame pentru coloanele lipsă
    missing_cols = [col for col in model_columns if col not in features_df.columns]
    for col in missing_cols:
        features_df[col] = 0  # Adaugă coloanele lipsă cu valori 0

    # Reordonam dataframe-ul pentru a se potrivi cu modelul
    features_df = features_df[model_columns]

    predicted_price = model.predict(features_df)
    return predicted_price[0]

