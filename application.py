### 1. Importation des librairies et chargement des données
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

# Configuration générale de l'application
st.set_page_config(
    page_title="💼 Data Science Salary Dashboard",
    page_icon="📊",
    layout="wide"
)

# Chargement des données
df = pd.read_csv("H:/SD3/SAE 601/Projet/Projet/ds_salaries.csv")

# Style CSS léger pour un rendu professionnel
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .stMetric {
        background-color: #f9fafb;
        padding: 12px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


### 2. Exploration visuelle des données
#votre code 

st.title("📊 Visualisation des Salaires en Data Science")
st.markdown("Analyse interactive des salaires en Data Science (2023) 🚀")

# Sidebar pour les options d'affichage
st.sidebar.title("🎛️ Paramètres d’analyse")
show_data = st.sidebar.checkbox("👀 Afficher un aperçu des données")

# KPI principaux
st.subheader("📌 Indicateurs clés")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Salaire moyen ($)", f"{int(df.salary_in_usd.mean()):,}")
col2.metric("📈 Salaire max ($)", f"{int(df.salary_in_usd.max()):,}")
col3.metric("📉 Salaire min ($)", f"{int(df.salary_in_usd.min()):,}")
col4.metric("📊 Nb d'observations", df.shape[0])

if show_data:
    st.dataframe(df.head())

# Statistiques générales avec describe pandas
st.subheader("📌 Statistiques générales")
st.write(df.describe())

st.info("""
🧠 **Interprétation :**

**Période couverte :** 2020 → 2023, avec une majorité de données en 2022–2023  
**Nombre d’observations :** 3 755 profils en Data Science  

💰 **Salaires**  
- Salaire moyen : ~137 600 $  
- Salaire médian : 135 000 $  
  → le marché est globalement bien rémunéré  
- Forte dispersion :  
  - minimum 5 132 $  
  - maximum 450 000 $  
  👉 présence d’énormes écarts selon le poste, le pays et l’expérience
""")



### 3. Distribution des salaires en France par rôle et niveau d'expérience, uilisant px.box et st.plotly_chart
#votre code 

st.subheader("📈 Distribution des salaires en France")

df_france = df[df["company_location"] == "FR"]

fig_fr = px.box(
    df_france,
    x="experience_level",
    y="salary_in_usd",
    color="job_title",
    title="💼 Salaires en France par expérience et rôle"
)

fig_fr.update_layout(template="plotly_white")

st.plotly_chart(fig_fr, use_container_width=True)

st.info("""
🧠 **Interprétation :**

- Les salaires augmentent avec l’expérience : les seniors (SE) gagnent plus que les juniors (EN).  
- Les postes techniques comme Machine Learning Engineer affichent des salaires plus élevés et plus variables.  
- Certains rôles, comme Data Analyst, ont des salaires plus stables et généralement plus bas.  
- Le rôle influence fortement le salaire au-delà de l’expérience.
""")


### 4. Analyse des tendances de salaires :
#### Salaire moyen par catégorie : en choisisant une des : ['experience_level', 'employment_type', 'job_title', 'company_location'], utilisant px.bar et st.selectbox 

st.subheader("📊 Salaire moyen par catégorie")

categorie = st.selectbox(
    "Choisissez une catégorie d'analyse",
    ['experience_level', 'employment_type', 'job_title', 'company_location']
)

salary_cat = df.groupby(categorie)["salary_in_usd"].mean().reset_index()

fig_cat = px.bar(
    salary_cat,
    x=categorie,
    y="salary_in_usd",
    color=categorie,
    title=f"💰 Salaire moyen par {categorie}"
)

fig_cat.update_layout(template="plotly_white", xaxis_tickangle=-30)

st.plotly_chart(fig_cat, use_container_width=True)

st.info("""
🧠 **Interprétation :**

- Le niveau **EX (Executive)** affiche le salaire moyen le plus élevé (~195 000 $).  
- Les profils **SE (Senior)** suivent avec environ 153 000 $.  
- Les niveaux **MI (Mid-level)** et **EN (Entry-level)** ont des salaires moyens plus bas, autour de 105 000 $ et 80 000 $.  
- L’écart important entre EX et les autres niveaux souligne la forte prime pour les cadres dirigeants.  
- La progression des salaires entre MI et SE n’est pas strictement linéaire, reflétant des variations selon les rôles.
""")



### 5. Corrélation entre variables
# Sélectionner uniquement les colonnes numériques pour la corrélation
#votre code 

st.subheader("🔗 Corrélations entre variables numériques")

df_num = df.select_dtypes(include=np.number)

# Calcul de la matrice de corrélation
#votre code

corr = df_num.corr()

# Affichage du heatmap avec sns.heatmap
#votre code 

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap="RdBu_r", fmt=".2f", ax=ax)
st.pyplot(fig)

st.info("""
🧠 **Interprétation :**

* 📈 **Tendance du marché :** La corrélation positive entre `work_year` et `salary_in_usd` (0.23) confirme que les salaires en Data Science augmentent globalement chaque année.
* 🏠 **Flexibilité vs Paie :** La corrélation quasi nulle (-0.06) entre le télétravail (`remote_ratio`) et le salaire indique que **travailler à distance ne pénalise pas la rémunération**.
* 🔄 **Évolution du Remote :** Le lien négatif (-0.24) entre l'année et le télétravail suggère un léger recul du "100% remote" au profit de modèles hybrides ou présentiels entre 2020 et 2023.
* ⚠️ **Note technique :** La variable `salary` brute n'est pas corrélée au reste car elle mélange plusieurs devises. Seule la colonne `salary_in_usd` permet une analyse comparative fiable.
""")


### 6. Analyse interactive des variations de salaire
# Une évolution des salaires pour les 10 postes les plus courants
# count of job titles pour selectionner les postes
# calcule du salaire moyen par an
#utilisez px.line
#votre code 

st.subheader("📉 Évolution des salaires pour les postes les plus courants")

top_jobs = df["job_title"].value_counts().head(10).index
df_top_jobs = df[df["job_title"].isin(top_jobs)]

salary_trend = df_top_jobs.groupby(
    ["work_year", "job_title"]
)["salary_in_usd"].mean().reset_index()

fig_line = px.line(
    salary_trend,
    x="work_year",
    y="salary_in_usd",
    color="job_title",
    title="📈 Évolution du salaire moyen par poste"
)

fig_line.update_layout(template="plotly_white")

st.plotly_chart(fig_line, use_container_width=True)

st.info("""
🧠 **Interprétation :**

* 📈 **Croissance généralisée :** La majorité des postes (Data Analyst, Data Engineer, Data Scientist) affiche une progression constante depuis 2020, confirmant la forte demande du marché.
* 🚀 **Top Performers :** Les rôles de **Data Science Manager** et **Applied Scientist** dominent le haut du graphique, franchissant la barre des 200 000 $ en 2023.
* ⚡ **Volatilité vs Stabilité :** Certains postes comme le *Research Scientist* montrent des variations importantes, tandis que le poste de **Data Analyst** suit une hausse régulière mais reste en bas de l'échelle salariale du secteur.
* 🔍 **Convergence :** En 2023, on observe un regroupement de plusieurs métiers (Machine Learning Engineer, Analytics Engineer) autour de la zone 150k-175k $, indiquant une standardisation des salaires pour les profils techniques intermédiaires.
""")

### 7. Salaire médian par expérience et taille d'entreprise
# utilisez median(), px.bar
#votre code 

st.subheader("🏢 Salaire médian par expérience et taille d'entreprise")

median_salary = df.groupby(
    ["experience_level", "company_size"]
)["salary_in_usd"].median().reset_index()

fig_median = px.bar(
    median_salary,
    x="experience_level",
    y="salary_in_usd",
    color="company_size",
    barmode="group",
    title="💼 Salaire médian selon l'expérience et la taille d'entreprise"
)

fig_median.update_layout(template="plotly_white")

st.plotly_chart(fig_median, use_container_width=True)

st.info("""
🧠 **Interprétation** :  

* 🏢 **Le paradoxe des PME :** Contre toute attente, pour les niveaux **EN (Entry)** et **MI (Mid)**, ce sont les entreprises de taille moyenne (**M**) qui offrent souvent les meilleurs salaires médians, dépassant même les grandes structures (**L**).
* 👑 **Exécutifs (EX) :** Le salaire médian explose en entreprise moyenne et petite pour les cadres dirigeants, probablement car ces profils y portent des responsabilités critiques et transverses.
* 📉 **Petites entreprises (S) :** Elles restent globalement les moins compétitives sur les salaires, particulièrement pour les profils seniors (**SE**), où l'écart avec les entreprises **M** et **L** est le plus marqué.
* 📈 **Progression :** Peu importe la taille de l'entreprise, le passage au niveau Senior ou Executive garantit une augmentation significative du pouvoir d'achat.
""")


### 8. Ajout de filtres dynamiques
#Filtrer les données par salaire utilisant st.slider pour selectionner les plages 
#votre code 

st.subheader("🎚️ Filtrage par plage de salaire")

min_salary, max_salary = st.slider(
    "Sélectionnez la plage de salaire ($)",
    int(df.salary_in_usd.min()),
    int(df.salary_in_usd.max()),
    (50000, 200000)
)

df_salary_filtered = df[
    (df.salary_in_usd >= min_salary) &
    (df.salary_in_usd <= max_salary)
]

st.write(f"📊 Nombre d'observations : {df_salary_filtered.shape[0]}")


### 9.  Impact du télétravail sur le salaire selon le pays

st.subheader("🏠 Impact du télétravail selon le pays")

fig_remote = px.box(
    df_salary_filtered,
    x="remote_ratio",
    y="salary_in_usd",
    color="company_location",
    title="💻 Télétravail et salaire par pays"
)

fig_remote.update_layout(template="plotly_white")

st.plotly_chart(fig_remote, use_container_width=True)

st.info("""
🧠 **Interprétation** :  

* 🌍 **Standard Mondial :** Le télétravail (100%) est largement adopté dans presque tous les pays analysés, avec des niveaux de rémunération souvent identiques, voire supérieurs, au présentiel.
* 🇺🇸 **Domination US :** Les États-Unis (US) affichent les boîtes à moustaches les plus hautes, quel que soit le ratio de télétravail, confirmant leur position de leader sur les salaires tech.
* 🇪🇺 **Disparités Européennes :** En France (FR), Allemagne (DE) ou Espagne (ES), le télétravail est bien présent, mais les médianes restent souvent plus basses que les standards anglo-saxons (US/CA).
* ⚖️ **Équilibre :** L'absence de chute drastique des salaires à 100% de remote montre que le secteur de la Data valorise le résultat plutôt que la présence physique.
""")

### 10. Filtrage avancé des données avec deux st.multiselect, un qui indique "Sélectionnez le niveau d'expérience" et l'autre "Sélectionnez la taille d'entreprise"
#votre code 

st.subheader("🧩 Filtrage avancé")

exp_filter = st.multiselect(
    "Sélectionnez le niveau d'expérience",
    df["experience_level"].unique(),
    default=df["experience_level"].unique()
)

size_filter = st.multiselect(
    "Sélectionnez la taille d'entreprise",
    df["company_size"].unique(),
    default=df["company_size"].unique()
)

df_advanced = df[
    (df["experience_level"].isin(exp_filter)) &
    (df["company_size"].isin(size_filter))
]

st.dataframe(df_advanced.head())

st.success("🎯 Application prête ! Analyse complète, professionnelle et interactive 🚀")
