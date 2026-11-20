import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from src.models.sauvegardes.function import recommander_hybride


######## Pour le lancement du fichier, se placer à la racine du projet ########



### Imports du modèle utilisé

@st.cache_resource # évite le rechargement à chaque clic
def charger_modele():
    # on récupère toutes nos données depuis 
    export = joblib.load('src/models/sauvegardes/modele_hybride.pkl')
    df = export['df']
    vecteurs_texte = export['vecteurs_texte']
    vecteurs_character = export['vecteurs_character']
    poids = export['poids']
    modele = SentenceTransformer(export['modele'])
    return df, vecteurs_texte, vecteurs_character, poids, modele

df, vecteurs_texte, vecteurs_character, poids, modele = charger_modele()



### Préparation des listes

familles = sorted(df['family'].dropna().unique())

saisons = {'spring' : 'Spring', 
           'summer' : 'Summer', 
           'autumn' : 'Autumn', 
           'winter': 'Winter'}

moments = {'day' : 'Day', 
           'evening': 'Evening', 
           'night': 'Night'}

intensites = {1: 'Very light', 2: 'Light', 3: 'Moderate', 4: 'Strong', 5: 'Very strong'}

liste_notes = sorted({note for cellule in df['notes'] for note in cellule.split('; ')})

noms_critere = {
    'semantique': 'Description',
    'famille': 'Fragrance family',
    'emotion': 'Character',
    'saison': 'Season and time',
    'intensite': 'Intensity'
}


### Les questions

st.title('Find your perfect perfume')

description_remplie = st.text_area('Describe the perfume you are looking for.', placeholder= 'Ex: A warm, woody and comforting perfume')

familles_choisies = st.multiselect('Which fragrance families do you like ?', options = familles)

emotions_choisies = st.text_input('What emotions or atmosphere should it evoke ?', placeholder = 'Ex: cosy, elegant, fresh')

saisons_choisies = st.multiselect('In which seasons would you like to wear it ?', options = list(saisons), format_func=lambda s: saisons[s])

moments_choisis = st.multiselect('At what time of the day ?',  options = list(moments), format_func=lambda m: moments[m])

intensite_choisie = st.select_slider('What intensity are you looking for?', options = list(intensites), value = 3, format_func=lambda i: f"{i} - {intensites[i]}")

notes_detestees = st.multiselect('Are there any notes you dislike ?', options = liste_notes)


if st.button('Find my perfume'):

    profil = {
        'description': description_remplie,
        'familles': familles_choisies,
        'emotions': emotions_choisies,
        'saisons': saisons_choisies,
        'moments': moments_choisis,
        'intensite': intensite_choisie,
        'notes_detestees': notes_detestees,
    }

    recommandations = recommander_hybride(profil, df, modele, vecteurs_texte, vecteurs_character, poids, noms_critere, n=3)


### Affichage des recommandations

    st.title('Your recommendations')

    # Parfum numéro 1
    principal = recommandations.iloc[0]

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
             st.markdown(f"### {principal['name']} <span style='font-size: 16px; color: gray'> by {principal['house']}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown("<p style='text-align: right; color: gray; font-size: 1.5rem'>#1</p>", unsafe_allow_html=True)
        st.write(f"**Fragrance family :** {principal['family']}")
        st.markdown('**Why this perfume ?**')
        st.write(principal['explication'])
        st.markdown(f"[Buy it here !]({principal['url']})")

    # les 2 autres recommandations

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            col3, col4 = st.columns(2)
            with col3:
                st.write(f'**{recommandations.iloc[1]['name']}**')
                st.caption(f'**By {recommandations.iloc[1]['house']}**')
            with col4:
                st.markdown("<p style='text-align: right; color: gray; font-size: 1rem'>#2</p>", unsafe_allow_html=True)
            st.markdown(f"[Buy it here !]({recommandations.iloc[1]['url']})")

    with col2:
            with st.container(border=True):
                col3, col4 = st.columns(2)
                with col3:
                    st.write(f'**{recommandations.iloc[2]['name']}**')
                    st.caption(f'**By {recommandations.iloc[2]['house']}**')
                with col4:
                    st.markdown("<p style='text-align: right; color: gray; font-size: 1rem'>#3</p>", unsafe_allow_html=True)
                st.markdown(f"[Buy it here !]({recommandations.iloc[2]['url']})")


### Affichage des graphes

    st.subheader('Learn more about your results')

    criteres = ['semantique', 'famille', 'emotion', 'saison', 'intensite']
    labels = ['Description', 'Family', 'Emotion', 'Season', 'Intensity']
    couleurs = ["#6087b6", "#a360b6", "#61a44f"]
    couleurs_criteres = ["#6087b6", "#a360b6", "#61a44f", "#eda100", "#6250d6"] 
    
    col1, col2 = st.columns(2)

    with col1:

        fig_radar = go.Figure()
        for i, (_, parfum) in enumerate(recommandations.iterrows()):
            valeurs = [parfum[f's_{c}'] for c in criteres]
            fig_radar.add_trace(go.Scatterpolar(
                r=valeurs + valeurs[:1],
                theta= labels + labels[:1],
                name=f"#{i+1} {parfum['name']}",
                line=dict(color=couleurs[i], width=2),
                fill='toself', opacity=0.6,
                hovertemplate='%{theta} : %{r:.2f}<extra>' + parfum['name'] + '</extra>',
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0,1], tickvals=[0.25, 0.5, 0.75, 1], tickfont=dict(size=10, color='gray'), gridcolor='#e1e0d9')), legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),margin=dict(l=40, r=40, t=20, b=20), height=420)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        noms = [f"#{i+1} {n}" for i, n in enumerate(recommandations['name'])]
        fig_score = go.Figure()
        for c, label, couleur in zip(criteres, labels, couleurs_criteres):
            fig_score.add_trace(go.Bar(
                y=noms,
                x=poids[c]*recommandations[f's_{c}'],
                name=label, orientation='h',
                marker=dict(color=couleur, line=dict(color='white', width=2)),
                hovertemplate=label + ' : %{x:.2f}<extra></extra>',
            ))
            fig_score.update_layout(
                barmode='stack',
                yaxis=dict(autorange='reversed'),
                xaxis=dict(title='Score', gridcolor='#e1e0d9'),
                legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center'),
                margin=dict(l=10, r=10, t=20, b=20), height=420,
            )
        st.plotly_chart(fig_score, use_container_width=True)