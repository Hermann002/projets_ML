import streamlit as st
import joblib
import numpy as np

# Titre de l'application
st.title("Prédiction de l'adoption d'un produit")

# Chargement du modèle sauvegardé
model = joblib.load("model.pkl")

# Création des widgets pour entrer les données
age = st.number_input("Âge", min_value=18, max_value=100, value=30)
experience = st.number_input("Expérience (années)", min_value=0, max_value=50, value=5)
income = st.number_input("Revenu annuel ($)", min_value=0, max_value=1000000, value=50000)
zip_code = st.number_input("Code postal", min_value=10000, max_value=99999, value=12345)
family = st.selectbox("Taille de la famille", options=[1, 2, 3, 4], index=0)
cc_avg = st.number_input("Dépenses mensuelles moyennes par carte de crédit ($)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
education = st.selectbox("Niveau d'éducation", options=[1, 2, 3], index=0)  # 1: Bas, 2: Moyen, 3: Elevé
mortgage = st.number_input("Hypothèque ($)", min_value=0, max_value=1000000, value=0)
securities_account = st.checkbox("Compte en valeurs mobilières")
cd_account = st.checkbox("Compte à terme")
online = st.checkbox("Utilise Internet")
credit_card = st.checkbox("Possède une carte de crédit")

# Bouton pour effectuer la prédiction
if st.button("Prédire"):
    # Préparation des données pour le modèle
    input_data = np.array([
        age, experience, income, zip_code, family, cc_avg, education, mortgage,
        int(securities_account), int(cd_account), int(online), int(credit_card)
    ]).reshape(1, -1)

    # Prédiction avec le modèle chargé
    prediction = model.predict(input_data)

    # Affichage du résultat
    if prediction[0] == 1:
        st.success("Le client adoptera probablement le produit 🎉")
    else:
        st.error("Le client n'adoptera probablement pas le produit 😔")