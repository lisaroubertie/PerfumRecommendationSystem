# Projet final - Big Data & Data Science - A3MSI
## Système intelligent de recommandations contextuelles de parfum basé sur l'analyse sémantique

Groupe : Lisa Roubertie, Aurélie Nass, Sara Ji-Lo

### Dataset

Le dataset à télécharger est parfumes.csv, disponible sur le lien suivant : `https://github.com/Scento-com/perfume-dataset`.

Il est a placer au chemin suivant, en local : `datasets/original/perfumes.csv`


### Nettoyage du dataset

Nous avons commencé de projet par une exploration de données, suivie d'un nettoyage du dataset. Ce nettoyage est fait en exécutant le fichier `data_cleaning.py`. Il créera automatiquement, en local également, un dataset nommé parfumes_clean.csv (enregistré au chemin : `datasets/clean/parfumes_clean.csv`).


### Les modèles

Pour que chacune de nous puissions essayer de mettre en place un modèle, nous avons pris la décision de créer 3 modèles différents, puis de les comparer à la fin pour garder le meilleur pour notre interface.
Nous avons donc un modèle utilisant Word2Vec avec la similarité cosinus, un modèle avec Sentence Transformers utilisant SBERT et un modèle hybride créer avec le Sentence Transformers ainsi que les critères structurés avec poids.


### Comparaison des modèles et enregistrement

Après comparaison, comme nous nous y attendions, le modèle Word2Vec n'est pas bon pour la recommandation contextuelle, et c'est finalement notre modèle hybride qui était le meilleur.
Nous avons donc exporté les fonctions nécessaires pour faire tourner le modèle (`fichier fonction.py`) et exporté le modèle en format .pkl (`fichier modele_hybride.pkl`).


### L'interface

Notre interface a été faite sur Streamlit. Un questionnaire simple est proposé à l'utilisateur qui peut ensuite lancer l'analyse et voir ses recommandations personnalisées. Un top 3 apparait, ainsi que 2 graphiques permettant d'analyser les résultats proposés selon les catégories choisies par l'utilisateur.

Pour lancer l'interface, placez-vous à la racine du projet et exécutez `streamlit run app/interface.py`. Assurez-vous d'avoir téléchargez les fichiers correspondants au modèle au préalable, ainsi que les datasets.
