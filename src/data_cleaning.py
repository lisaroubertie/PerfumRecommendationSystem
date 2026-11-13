# Nettoyage du dataset de parfums

import unicodedata
from pathlib import Path

import pandas as pd


# Etape 1 : Chargement des données

Root = Path(__file__).resolve().parent.parent
INPUT_PATH = Root / "datasets" / "original" / "perfumes.csv"
OUTPUT_PATH = Root / "datasets" / "clean" / "perfumes_clean.csv"

# On travaille sur une copie pour pouvoir comparer avant / après
df_original = pd.read_csv(INPUT_PATH)
df = df_original.copy()
print(f"Dataset original : {df.shape[0]} lignes, {df.shape[1]} colonnes")


# Etape 2 : Traiter les valeurs manquantes

print("\n=== 2. Valeurs manquantes ===")
manquants = (df.isna().mean() * 100).round(2)
print(manquants[manquants > 0].sort_values(ascending=False))

#On supprime les colonnes perfumers, year et popularity_rank car elles ne sont pas pertinentes, + de 50% de valeurs manquantes.
df = df.drop(columns=["year", "perfumers", "popularity_rank"])


# 2.1 Valeurs Notes de tête / cœur / fond manquantes (6,15 %)
# On vérifie si la colonne 'notes' est renseignée

sans_pyramide = df["top_notes"].isna()
print("\nParfums sans pyramide :", sans_pyramide.sum())
print("Parmi eux, parfums dont la colonne 'notes' est remplie :", df.loc[sans_pyramide, "notes"].notna().sum())

# Cela signifie que pour ces parfums, on a la liste complète des notes mais pas la pyramide de notes (top, heart, base). Ainsi, on remplit les colonnes de notes manquantes avec une chaîne vide"

for col in ["top_notes", "heart_notes", "base_notes"]:
    df[col] = df[col].fillna("")

# 2.2 Famille secondaire manquante (27,32 %)
# Tous les parfums n'ont pas de famille secondaire : on remplace les valeurs manquantes par une chaîne vide
df["secondary_family"] = df["secondary_family"].fillna("")

# 2.3 Concentration manquante (16,97 %)
# Impossible à deviner la valeur de façon fiable -> on met "unknown" pour les valeurs manquantes
df["concentration"] = df["concentration"].fillna("unknown")

# Etape 3 : Nettoyer et uniformiser les valeurs textuelles

print("\n=== 3. Harmonisation des textes ===")

# Ici, on normalise les textes pour éviter les problèmes d'encodage, apostrophes et espaces superflus

def nettoyer_texte(texte):
    if pd.isna(texte):
        return ""
    texte = unicodedata.normalize("NFC", str(texte))               
    texte = texte.replace("\u2019", "'").replace("\u2018", "'")    
    return " ".join(texte.split()) 

# On applique la fonction de nettoyage à toutes les colonnes de type texte                                

colonnes_texte = df.select_dtypes(include=["object", "string"]).columns
for col in colonnes_texte:
    df[col] = df[col].apply(nettoyer_texte)
print("Textes harmonisés")

# Etape 4 : Structurer les variables multiples

print("\n=== 4. Variables multiples ===")

# Pour les colonnes multiples, on supprime les doublons et on met tout en minuscules

def nettoyer_liste(cellule):
    if not cellule:
        return ""
    elements = [e.strip().lower() for e in cellule.split(";") if e.strip()]
    return "; ".join(dict.fromkeys(elements))  # supprime les doublons en gardant l'ordre

# On applique la fonction de nettoyage aux colonnes multiples
colonnes_multiples = ["character", "seasons", "times_of_day",
                      "top_notes", "heart_notes", "base_notes", "notes"]
for col in colonnes_multiples:
    df[col] = df[col].apply(nettoyer_liste)

# On crée des colonnes binaires pour les saisons pour faciliter l'analyse et la recommandation
# En effet, un parfum peut être associé à plusieurs saisons, donc on crée une colonne par saison avec 1 si le parfum est associé à cette saison, 0 sinon
for saison in ["spring", "summer", "autumn", "winter"]:
    df[f"season_{saison}"] = df["seasons"].str.contains(saison).astype(int)

# pareil pour les moments de la journée

for moment in ["day", "evening", "night"]:
    df[f"time_{moment}"] = df["times_of_day"].str.contains(moment).astype(int)

# Etape 5 : Evaluer la pertinence des variables

print("\n=== 5. Pertinence des variables ===")

# 5.1 house_id fait doublon avec house -> on supprime
assert df.groupby("house_id")["house"].nunique().max() == 1
df = df.drop(columns=["house_id"])

# 5.2 Doublons fonctionnels : on regarde les parfums ayant le même nom et la même maison
# Mais on ne supprime pas les doublons car il peut s'agir de parfums différents (ex : eau de toilette et eau de parfum)

cle = df["name"].str.lower() + " | " + df["house"].str.lower()
print("Doublons fonctionnels :")
print(df[cle.duplicated(keep=False)][["id", "name", "house", "concentration"]])
print("Descriptions dupliquées :", df["description"].str.lower().duplicated().sum())

# 5.3 Intensité (1 à 5) déduite de la concentration,
# on crée une nouvelle colonne 'intensity' pour faciliter l'analyse et la recommandation

INTENSITE = {"eau-de-cologne": 1, "eau-de-toilette": 2, "eau-de-parfum": 3,
             "parfum": 4, "elixir": 4, "extrait": 5}
df["intensity"] = df["concentration"].map(INTENSITE).astype("Int64")  # vide si 'unknown'


# Etape 6 : Préparer les données textuelles (pour les embeddings)

print("\n=== 6. Préparation des textes ===")

# 6.1 Qualité des descriptions 
# On regarde la longueur des descriptions pour voir si elles sont suffisantes pour l'analyse sémantique
longueur = df["description"].str.len()
print(f"Longueur des descriptions : {longueur.min()} à {longueur.max()} caractères "
      f"(moyenne {longueur.mean():.0f})")
print("Les plus courtes :", df.loc[longueur.nsmallest(3).index, "description"].tolist())


# 6.2 Texte enrichi : pour les descriptions courtes, on les complète avec les autres champs descriptifs. 
# C'est cette colonne que les modèles (méthode Lisa, Sentence Transformers, hybride) vont vectoriser.

def construire_texte(p):
    lister = lambda s: s.replace("; ", ", ")

    # Famille principale + famille secondaire si elle existe
    familles = p["family"]
    if p["secondary_family"] != "":
        familles = familles + " and " + p["secondary_family"]

    morceaux = [p["description"],
                f"Olfactory family: {familles}.",
                f"Character: {lister(p['character'])}."]

    if p["top_notes"] != "":   # le parfum a une pyramide de notes
        morceaux += [f"Top notes: {lister(p['top_notes'])}.",
                     f"Heart notes: {lister(p['heart_notes'])}.",
                     f"Base notes: {lister(p['base_notes'])}."]
    else:                      # sinon on utilise la liste complète des notes
        morceaux.append(f"Notes: {lister(p['notes'])}.")

    morceaux.append(f"Seasons: {lister(p['seasons'])}. Time of day: {lister(p['times_of_day'])}.")
    return " ".join(morceaux)

df["text_full"] = df.apply(construire_texte, axis=1)
print(f"Longueur moyenne de text_full : {df['text_full'].str.len().mean():.0f} caractères")
print("Exemple :", df.loc[0, "text_full"])

# Ainsi, on a un texte enrichi pour chaque parfum, qui sera utilisé pour l'analyse sémantique et la recommandation.

# Etape 7 : Sauvegarder le dataset nettoyé

print("\n=== 7. Sauvegarde du dataset nettoyé ===")


colonnes_finales = [
    "id", "name", "house", "url",
    "family", "secondary_family", "character",
    "top_notes", "heart_notes", "base_notes", "notes",
    "concentration", "intensity", "seasons", "times_of_day",
    "season_spring", "season_summer", "season_autumn", "season_winter",
    "time_day", "time_evening", "time_night",
    "description", "text_full",
]

df_clean = df[colonnes_finales]

# Contrôles : aucune ligne perdue, identifiants uniques, plus de valeurs manquantes
assert len(df_clean) == len(df_original)
assert df_clean["id"].is_unique
assert df_clean.drop(columns=["intensity"]).isna().sum().sum() == 0

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("\n=== Résultat ===")
print(f"Lignes : {df_original.shape[0]} -> {df_clean.shape[0]}")
print(f"Colonnes : {df_original.shape[1]} -> {df_clean.shape[1]}")
print(f"Valeurs manquantes : {df_original.isna().sum().sum()} -> {df_clean.isna().sum().sum()}")
print(f"Fichier enregistré : {OUTPUT_PATH}")

# Remarque : il reste 251 valeurs manquantes qui correspond aux valeurs "unknown" de la colonne "INTENSITY"