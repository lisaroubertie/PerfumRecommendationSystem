from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Dans ce fichier, on recopie toutes les fonctions nécessaires à l'utilisation du modèle pour s'en servir dans l'interface

def normaliser(scores):
    ecart = scores.max() - scores.min()
    if ecart == 0:  # tous les scores identiques : on renvoie une valeur neutre
        return np.full(len(scores), 0.5)
    return (scores - scores.min()) / ecart


def criteres_principaux(parfum, poids, noms_criteres):
    contributions = {c: poids[c] * parfum[f"s_{c}"] for c in poids}
    meilleurs = sorted(contributions, key=contributions.get, reverse=True)[:2]
    return [noms_criteres[c] for c in meilleurs]

colonnes_scores = ["s_semantique", "s_famille", "s_emotion", "s_saison", "s_intensite"]

def expliquer(parfum):
    phrases = []
    if parfum["s_semantique"] >= 0.7:
        phrases.append(f"Your description is really close to this perfume (similarity {parfum['cosinus_brut']:.2f}).")
    if parfum["s_famille"] == 1 and parfum["family"] != '':
        phrases.append(f"It is from the family {parfum['family']} that you chose.")
    elif parfum["s_famille"] == 0.5 and parfum["secondary_family"] != '':
        phrases.append(f"Its secondary family ({parfum['secondary_family']}) is part of your choice.")
    if parfum["s_emotion"] >= 0.7 and parfum['character'] != 0:
        phrases.append(f"Its character ({parfum['character'].replace('; ', ', ')}) matches the astmosphere you are looking for.")
    if parfum["s_saison"] >= 0.7 and parfum['seasons'] != '':
        phrases.append(f"It is suitable for {parfum['seasons'].replace('; ', ', ')}.")
    if parfum["s_intensite"] == 1 and parfum['concentration'] != '':
        phrases.append(f"It's intensity ({parfum['concentration']}) matches your expectations.")
    if not phrases:
        phrases.append("It offers the best compromise between your criteria.")
    return " ".join(phrases)

def recommander_hybride(profil, df, modele, vecteurs_texte, vecteurs_character, poids, noms_critere, n=3):
    # 1. Filtre des notes détestées
    masque = pd.Series(False, index=df.index)
    for note in profil.get("notes_detestees", []):
        masque = masque | df["notes"].str.contains(note, case=False, na=False)
    candidats = df[~masque].copy()

    # 2. Sous-scores (0,5 si la question est sans réponse)
    if profil.get("description"):
        cos = cosine_similarity(modele.encode([profil["description"]]), vecteurs_texte[candidats.index])[0]
        candidats["s_semantique"], candidats["cosinus_brut"] = normaliser(cos), cos
    else:
        candidats["s_semantique"], candidats["cosinus_brut"] = 0.5, 0.0

    if profil.get("familles"):
        principale = candidats["family"].isin(profil["familles"])
        secondaire = candidats["secondary_family"].isin(profil["familles"])
        candidats["s_famille"] = np.where(principale, 1.0, np.where(secondaire, 0.5, 0.0))
    else:
        candidats["s_famille"] = 0.5

    if profil.get("emotions"):
        cos = cosine_similarity(modele.encode([profil["emotions"]]), vecteurs_character[candidats.index])[0]
        candidats["s_emotion"] = normaliser(cos)
    else:
        candidats["s_emotion"] = 0.5

    saisons, moments = profil.get("saisons", []), profil.get("moments", [])
    s_saison = candidats[[f"season_{s}" for s in saisons]].mean(axis=1) if saisons else 0.5
    s_moment = candidats[[f"time_{m}" for m in moments]].mean(axis=1) if moments else 0.5
    candidats["s_saison"] = 0.7 * s_saison + 0.3 * s_moment

    if profil.get("intensite") is not None:
        candidats["s_intensite"] = (1 - (candidats["intensity"] - profil["intensite"]).abs() / 4).fillna(0.5)
    else:
        candidats["s_intensite"] = 0.5

    # 3. Score final
    candidats["score"] = sum(poids[c] * candidats[f"s_{c}"] for c in poids)

    # 4. Top n avec critères principaux et explications
    top = candidats.sort_values(by="score", ascending=False).head(n).copy()
    top["criteres_principaux"] = top.apply(lambda parfum: criteres_principaux(parfum, poids, noms_critere), axis=1)
    top["explication"] = top.apply(expliquer, axis=1)
    return top[["name", "house", "family", "score", "url"] + colonnes_scores + ["criteres_principaux", "explication"]]