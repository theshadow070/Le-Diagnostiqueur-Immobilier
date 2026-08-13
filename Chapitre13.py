import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="Diagnostiqueur Médical", page_icon="🏥", layout="wide")
st.title("🏠 Le Diagnostiqueur Immobilier")
st.write("Renseignez les caractéristiques de la maison pour obtenir une estimation du prix.")

@st.cache_resource
def charger_modele() :
    return joblib.load("modele_immobilier_dakar.pkl")

modele = charger_modele()

data = fetch_california_housing()
toutes_features = pd.DataFrame(data.data, columns=data.feature_names)
minimum = toutes_features.min()
maximum = toutes_features.max()
moyenne = toutes_features.mean()
entree = moyenne.copy()

st.header("Mesures Principales")


for nom_feature in ["MedInc", "HouseAge", "AveRooms"] :
    valeur1 = st.slider(nom_feature, min_value=minimum[nom_feature], max_value=maximum[nom_feature], 
                         value=float(moyenne[nom_feature]))
    entree[nom_feature] = valeur1

for nom_feature in ["AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"] :
    valeur2 = st.number_input(nom_feature, min_value=minimum[nom_feature], max_value=maximum[nom_feature], 
                         value=float(moyenne[nom_feature]))
    entree[nom_feature] = valeur2


X_utilisateur = pd.DataFrame([entree])

if st.button("Obtenir le prix") :
    prix_estime = modele.predict(X_utilisateur)[0]
    prix_usd = prix_estime * 100_000
    prix_franc = prix_usd * 600
    st.write(f"Prix estimé en dollar : {prix_usd:,.0f}$")
    st.write(f"Équivalent indicatif : {prix_franc:,.0f}FCFA")
    st.caption("⚠️ Cette estimation est indicative et ne constitue pas une expertise immobilière.")
