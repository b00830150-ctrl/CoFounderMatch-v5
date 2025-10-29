import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import plotly.express as px

# ---- Setup ----
st.set_page_config(page_title="CoFounderMatch V3", layout="wide")
st.title("CoFounderMatch V3 🚀")

# Charger profils existants
try:
    with open("profiles.json", "r") as f:
        profiles = json.load(f)
except FileNotFoundError:
    profiles = []

# ---- Ajouter profil ----
with st.form("add_profile"):
    st.subheader("Ajoute ton profil")
    name = st.text_input("Nom")
    description = st.text_area("Description courte")
    skills = st.text_input("Compétences (séparées par des virgules)")
    goals = st.text_input("Objectifs")
    submitted = st.form_submit_button("Ajouter mon profil")

    if submitted:
        profile = {
            "name": name,
            "description": description,
            "skills": [s.strip() for s in skills.split(",")],
            "goals": goals
        }
        profiles.append(profile)
        with open("profiles.json", "w") as f:
            json.dump(profiles, f, indent=4)
        st.success(f"Profil de {name} ajouté !")

# ---- Matchmaking ----
st.subheader("Matchs possibles")
if profiles:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    current_user = st.selectbox("Choisis ton profil pour matcher", [p['name'] for p in profiles])
    
    user_profile = next(p for p in profiles if p['name'] == current_user)
    user_embedding = model.encode(user_profile['description'])

    matches = []
    for p in profiles:
        if p['name'] != current_user:
            score = util.cos_sim(user_embedding, model.encode(p['description'])).item()
            matches.append({"name": p['name'], "score": score, "skills": p['skills']})

    matches = sorted(matches, key=lambda x: x['score'], reverse=True)

    for m in matches:
        st.write(f"**{m['name']}** - Compatibilité: {m['score']:.2f}")
        st.write("Compétences :", ", ".join(m['skills']))

    # ---- Graphique radar ----
    st.subheader("Radar de compétences")
    if matches:
        all_skills = list({skill for m in matches for skill in m['skills']})
        df = pd.DataFrame(0, index=[m['name'] for m in matches], columns=all_skills)
        for m in matches:
            for skill in m['skills']:
                df.at[m['name'], skill] = 1
        fig = px.line_polar(df.reset_index(), r=df.values.flatten(), theta=df.columns.tolist()*len(df),
                            line_close=True, markers=True)
        st.plotly_chart(fig)

# ---- Chat interne ----
st.subheader("Chat interne")
chat_with = st.selectbox("Choisis avec qui chatter", [p['name'] for p in profiles if p['name'] != current_user] if profiles else [])

# Charger messages existants
try:
    with open("messages.json", "r") as f:
        messages = json.load(f)
except FileNotFoundError:
    messages = []

# Message de succès (optionnel)
if 'message' in locals():
    st.success(message)

