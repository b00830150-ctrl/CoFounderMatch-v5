import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import openai

# -------------------------
# CONFIGURATION
# -------------------------
st.set_page_config(page_title="🤝 CoFounderMatch", page_icon="🤝", layout="wide")

openai.api_key = st.secrets.get("OPENAI_API_KEY")

# -------------------------
# DATA + MODEL
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

profiles = [
    {"name": "Alice", "skills": "AI, data science, Python, research", "personality": "analytical, introverted, reliable", "domain": "tech", "strengths": [8, 7, 6, 9, 5, 6, 8, 7], "email": "alice@example.com"},
    {"name": "Ben", "skills": "marketing, branding, storytelling, design", "personality": "creative, extroverted, energetic", "domain": "marketing", "strengths": [5, 9, 8, 6, 7, 7, 8, 6], "email": "ben@example.com"},
    {"name": "Chloe", "skills": "finance, strategy, fundraising, management", "personality": "structured, ambitious, calm", "domain": "business", "strengths": [9, 7, 6, 8, 5, 8, 6, 9], "email": "chloe@example.com"},
    {"name": "David", "skills": "software engineering, backend, AI, product", "personality": "logical, humble, focused", "domain": "tech", "strengths": [8, 6, 9, 7, 5, 8, 7, 6], "email": "david@example.com"},
    {"name": "Emma", "skills": "UX, communication, product design", "personality": "empathetic, visionary, adaptable", "domain": "design", "strengths": [6, 8, 7, 6, 9, 7, 8, 7], "email": "emma@example.com"},
]

profiles_df = pd.DataFrame(profiles)

# -------------------------
# USER INTERFACE
# -------------------------
st.title("🤝 CoFounderMatch — Find Your Perfect Startup Partner")

st.sidebar.header("🎯 Your Profile")

your_name = st.sidebar.text_input("Your name:")
your_email = st.sidebar.text_input("Your email address:")
your_skills = st.sidebar.text_area("Your skills and experience:")
your_personality = st.sidebar.text_area("Your personality and working style:")
your_domain = st.sidebar.selectbox("Your main domain:", ["tech", "marketing", "business", "design", "other"])

st.sidebar.markdown("### 🧩 Rate your key strengths (1–10)")
your_strengths = [
    st.sidebar.slider("Technical expertise", 1, 10, 7),
    st.sidebar.slider("Creativity", 1, 10, 7),
    st.sidebar.slider("Teamwork", 1, 10, 7),
    st.sidebar.slider("Leadership", 1, 10, 7),
    st.sidebar.slider("Empathy", 1, 10, 7),
    st.sidebar.slider("Adaptability", 1, 10, 7),
    st.sidebar.slider("Strategic thinking", 1, 10, 7),
    st.sidebar.slider("Communication", 1, 10, 7),
]

# -------------------------
# SAVE USER IN DATABASE
# -------------------------
if your_name and your_skills and your_personality:
    new_profile = {
        "name": your_name,
        "skills": your_skills,
        "personality": your_personality,
        "domain": your_domain,
        "strengths": your_strengths,
        "email": your_email or "N/A",
    }

    if your_name not in profiles_df["name"].values:
        profiles_df.loc[len(profiles_df)] = new_profile

# -------------------------
# MATCHING
# -------------------------
if st.sidebar.button("🔍 Find Matches"):
    if your_skills.strip() == "" or your_personality.strip() == "":
        st.warning("⚠️ Please fill in your skills and personality!")
    else:
        your_text = your_skills + " " + your_personality
        your_embedding = model.encode(your_text, convert_to_tensor=True)

        profiles_df["similarity"] = profiles_df.apply(
            lambda row: util.cos_sim(
                your_embedding,
                model.encode(row["skills"] + " " + row["personality"], convert_to_tensor=True)
            ).item(),
            axis=1
        )

        results = profiles_df.sort_values(by="similarity", ascending=False).reset_index(drop=True)

        st.subheader("💡 Your Top Matches")

        for _, row in results.iterrows():
            with st.expander(f"👤 {row['name']} — {row['domain'].capitalize()} ({row['similarity']:.2f})"):
                st.write(f"**Skills:** {row['skills']}")
                st.write(f"**Personality:** {row['personality']}")
                st.write(f"**Email contact:** {row['email']}")

                # --- Radar Chart ---
                labels = [
                    "Technical", "Creativity", "Teamwork", "Leadership",
                    "Empathy", "Adaptability", "Strategy", "Communication"
                ]
                user_values = np.array(your_strengths)
                match_values = np.array(row["strengths"])

                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                user_values = np.concatenate((user_values, [user_values[0]]))
                match_values = np.concatenate((match_values, [match_values[0]]))
                angles += angles[:1]

                fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                ax.plot(angles, user_values, label="You", linewidth=2)
                ax.fill(angles, user_values, alpha=0.25)
                ax.plot(angles, match_values, label=row["name"], linewidth=2)
                ax.fill(angles, match_values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels)
                ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
                st.pyplot(fig)

                # --- OpenAI Summary ---
                if openai.api_key:
                    prompt = f"Summarize in 2 sentences why {your_name or 'You'} and {row['name']} would make great cofounders based on their skills and personalities."
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a startup mentor."},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        summary = response["choices"][0]["message"]["content"]
                        st.info(f"🧠 Compatibility summary:\n{summary}")
                    except Exception as e:
                        st.warning(f"Could not generate AI summary: {e}")
                else:
                    st.warning("🔑 Add your OpenAI key in Streamlit secrets to enable AI insights.")
