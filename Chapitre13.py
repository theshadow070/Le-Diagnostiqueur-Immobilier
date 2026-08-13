import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="L'Estimateur Immobilier", page_icon="🏠",layout="wide")
st.title("🏠 L'Estimateur Immobilier")
st.write("Renseignez les caractéristiques de la maison pour obtenir une estimation du prix.")

# Charger le modèle sauvegardé.
@st.cache_resource
def charger_modele() :
    return joblib.load("modele_immobilier_dakar.pkl")

modele = charger_modele()

# Récupérer les noms, minimums, maximums et moyennes des 8 features.
data = fetch_california_housing()
toutes_features = pd.DataFrame(data.data, columns=data.feature_names)
minimum = toutes_features.min()
maximum = toutes_features.max()
moyenne = toutes_features.mean()

# Commencer avec les valeurs moyennes des 8 features.
entree = moyenne.copy()

st.header("Mesures Principales")

# Les trois premières features sont saisies avec un curseur.
for nom_feature in ["MedInc", "HouseAge", "AveRooms"] :
    valeur = st.slider(nom_feature, min_value=minimum[nom_feature], max_value=maximum[nom_feature], 
                         value=float(moyenne[nom_feature]))
    entree[nom_feature] = valeur

# Les cinq autres features sont saisies avec un champ numérique.
for nom_feature in ["AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"] :
    valeur = st.number_input(nom_feature, min_value=minimum[nom_feature], max_value=maximum[nom_feature], 
                         value=float(moyenne[nom_feature]))
    entree[nom_feature] = valeur

# Transformer les valeurs saisies en DataFrame pour le modèle.
X_utilisateur = pd.DataFrame([entree])

if st.button("Obtenir le prix") :
    prix_estime = modele.predict(X_utilisateur)[0]
    # La cible du California Housing est exprimée en centaines de milliers de dollars.
    prix_usd = prix_estime * 100_000

    # Conversion illustrative en FCFA.
    prix_franc = prix_usd * 600
    
    st.write(f"Prix estimé en dollars : {prix_usd:,.0f} $")
    st.write(f"Équivalent indicatif : {prix_franc:,.0f} FCFA")
    st.caption("⚠️ Cette estimation est indicative et ne constitue pas une expertise immobilière.")
    
