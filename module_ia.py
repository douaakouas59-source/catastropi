"""
=============================================================================
MODULE IA - SYSTEME INTELLIGENT DE GESTION DES CATASTROPHES
=============================================================================
Projet S8 - ING4 Groupe 1-2
Université M'hamed Bougara - Boumerdès

Ce module contient:
1. detecter_fake_news() - Détection des fausses alertes (CamemBERT + TF-IDF)
2. predire_niveau_risque() - Prédiction du niveau de risque (avec heure et météo)
3. estimer_impact() - Estimation de l'impact
4. prioriser_interventions() - Priorisation des interventions
5. optimiser_allocation_ressources() - Optimisation allocation ressources
6. aider_planification() - Aide à la planification/scénarios

Dépendances: pip install pandas scikit-learn openpyxl numpy transformers torch shap requests
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests as http_requests
import warnings
import json
warnings.filterwarnings('ignore')

# Optionnel: CamemBERT et SHAP (graceful fallback)
try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers non installé. Détection Fake News en mode TF-IDF uniquement.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ shap non installé. Explicabilité en mode heuristique.")


# =============================================================================
# CLASSE PRINCIPALE DU MODULE IA
# =============================================================================

class ModuleIA:
    """
    Classe principale regroupant toutes les fonctionnalités du module IA
    pour la gestion intelligente des catastrophes.
    """
    
    def __init__(self, chemin_fichier_excel="historique_catastrophes.xlsx"):
        """
        Initialisation du module IA.
        
        Args:
            chemin_fichier_excel (str): Chemin vers le fichier Excel 
                                        contenant l'historique des catastrophes
        """
        self.chemin_excel = chemin_fichier_excel
        self.modele_fake_news = None
        self.vectorizer_fake_news = None
        self.modele_risque = None
        self.modele_impact = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.donnees_historiques = None
        self.camembert_pipeline = None
        self.shap_explainer_risque = None
        
        # Clé OpenWeatherMap (gratuite)
        self.OWM_API_KEY = "demo"  # Remplacer par une vraie clé
        
        # Charger et préparer les données
        self._charger_donnees_historiques()
        self._entrainer_modeles()
        self._init_camembert()
    
    # =========================================================================
    # INITIALISATION CamemBERT
    # =========================================================================
    def _init_camembert(self):
        """Tente de charger le pipeline CamemBERT pour analyse sémantique."""
        if HAS_TRANSFORMERS:
            try:
                print("🧠 Chargement du modèle CamemBERT (première fois peut prendre du temps)...")
                self.camembert_pipeline = hf_pipeline(
                    "text-classification",
                    model="cmarkea/distilcamembert-base-sentiment",
                    top_k=None
                )
                print("   ✅ CamemBERT chargé avec succès !")
            except Exception as e:
                print(f"   ⚠️ Impossible de charger CamemBERT: {e}")
                self.camembert_pipeline = None
        else:
            self.camembert_pipeline = None

    # =========================================================================
    # FONCTION UTILITAIRE - Chargement des données
    # =========================================================================
    
    def _charger_donnees_historiques(self):
        """
        Charge les données historiques depuis le fichier Excel.
        Si le fichier n'existe pas, crée un jeu de données d'exemple.
        """
        try:
            self.donnees_historiques = pd.read_excel(self.chemin_excel)
            print(f"✅ Données historiques chargées: {len(self.donnees_historiques)} enregistrements")
        except FileNotFoundError:
            print("⚠️ Fichier Excel non trouvé. Création de données d'exemple...")
            self.donnees_historiques = self._creer_donnees_exemple()
    
    def _creer_donnees_exemple(self):
        """
        Crée un jeu de données d'exemple pour le développement et les tests.
        Ces données simulent l'historique des catastrophes passées.
        """
        np.random.seed(42)
        
        # Types de catastrophes
        types_catastrophe = [
            "Inondation", "Incendie", "Séisme", "Accident chimique", 
            "Tempête", "Glissement de terrain", "Tsunami", "Éruption volcanique"
        ]
        
        # Niveaux de risque
        niveaux_risque = ["Faible", "Moyen", "Élevé", "Critique"]
        
        # Zones géographiques (Wilayas d'Algérie)
        zones = [
            "Alger", "Oran", "Constantine", "Boumerdès", "Béjaïa",
            "Tizi Ouzou", "Blida", "Tipaza", "Chlef", "Annaba"
        ]
        
        # Générer 200 enregistrements historiques
        n_enregistrements = 200
        
        donnees = {
            "id": range(1, n_enregistrements + 1),
            "type_catastrophe": np.random.choice(types_catastrophe, n_enregistrements),
            "zone": np.random.choice(zones, n_enregistrements),
            "mois": np.random.randint(1, 13, n_enregistrements),
            "population_zone": np.random.randint(50000, 500000, n_enregistrements),
            "superficie_affectee_km2": np.random.uniform(0.5, 50, n_enregistrements).round(2),
            "duree_heures": np.random.uniform(1, 168, n_enregistrements).round(1),
            "nb_victimes_reel": np.random.randint(0, 500, n_enregistrements),
            "nb_blesses_reel": np.random.randint(0, 1000, n_enregistrements),
            "nb_sinistres_reel": np.random.randint(0, 2000, n_enregistrements),
            "degats_materiaux_da": np.random.randint(100000, 100000000, n_enregistrements),
            "equipes_necessaires": np.random.randint(2, 50, n_enregistrements),
            "ambulances_necessaires": np.random.randint(1, 30, n_enregistrements),
            "niveau_risque": np.random.choice(niveaux_risque, n_enregistrements, 
                                               p=[0.3, 0.35, 0.25, 0.1]),
            "alerte_prealable_heures": np.random.randint(0, 48, n_enregistrements),
            "infrastructures_endommagees": np.random.randint(0, 50, n_enregistrements),
            "evacuation_necessaire": np.random.choice([0, 1], n_enregistrements, p=[0.6, 0.4])
        }
        
        df = pd.DataFrame(donnees)
        
        # Sauvegarder en Excel
        df.to_excel("historique_catastrophes.xlsx", index=False)
        print("✅ Fichier 'historique_catastrophes.xlsx' créé avec succès!")
        
        return df
    
    # =========================================================================
    # ENTRAINEMENT DES MODÈLES
    # =========================================================================
    
    def _entrainer_modeles(self):
        """
        Entraîne tous les modèles IA nécessaires:
        - Modèle de détection de fake news
        - Modèle de prédiction du niveau de risque
        - Modèle d'estimation de l'impact
        """
        print("\n" + "="*60)
        print("🔧 ENTRAINEMENT DES MODÈLES IA")
        print("="*60)
        
        self._entrainer_modele_fake_news()
        self._entrainer_modele_risque()
        self._entrainer_modele_impact()
        
        print("\n✅ Tous les modèles sont entraînés et prêts!")
    
    # =========================================================================
    # FONCTION 1: DÉTECTION DES FAKE NEWS
    # =========================================================================
    
    def _entrainer_modele_fake_news(self):
        """
        Entraîne le modèle de détection de fausses nouvelles
        en utilisant un dataset enrichi de phrases françaises/algériennes.
        """
        print("\n📱 Entraînement du modèle de détection Fake News...")
        
        # ============================================================
        # DATASET ENRICHI: 50+ rumeurs locales algériennes
        # ============================================================
        fake_texts = [
            "URGENT: Inondation massive à Alger, des milliers de morts, le gouvernement cache la vérité!",
            "CATASTROPHE: Séisme de magnitude 9 à Oran, la ville est entièrement détruite!",
            "ATTENTION: Accident nucléaire à Boumerdès, évacuez immédiatement la région!",
            "INFO EXCLUSIVE: Tempête de sable géante va détruire toute l'Algérie demain!",
            "ALERT: Le volcan dormant de Tizi Ouzou va entrer en éruption dans 2 heures!",
            "BREAKING: Tsunami de 100 mètres va frapper la côte algérienne ce soir!",
            "URGENT: Épidémie mortelle non identifiée à Constantine, milliers de contaminés!",
            "SECRET: Le gouvernement cache un accident chimique majeur à Blida!",
            "CATASTROPHE TOTALE: Inondation qui va submerger tout le nord de l'Algérie!",
            "ATTENTION: Invasion de sauterelles géantes va détruire toutes les récoltes!",
            "FAKE: Tremblement de terre va frapper Alger à l'heure exacte demain!",
            "URGENT: Les barrages vont tous céder simultanément, évacuez!",
            "ATTENTION: Température de 60 degrés prévue, danger de mort!",
            "EXCLUSIF: Accident nucléaire caché par les autorités depuis des mois!",
            "ALERTE: Des météorites vont tomber sur l'Algérie cette semaine!",
            "DANGER: Le séisme d'hier était en réalité une explosion nucléaire secrète!",
            "INFO: Les secours ne viendront jamais, sauvez-vous par vous-mêmes!",
            "CATASTROPHE: Tous les ponts d'Alger se sont effondrés simultanément!",
            "URGENT: L'eau potable est empoisonnée dans tout le nord du pays!",
            "ATTENTION: Une vague de chaleur va tuer des millions de personnes!",
            "EXCLUSIF: On nous cache la vraie magnitude du séisme, c'était 8.5!",
            "FUYEZ: Un raz de marée de 50 mètres va frapper Jijel dans 1 heure!",
            "SCANDALE: Les pompiers refusent d'intervenir, corruption totale!",
            "URGENT: Explosion d'une usine chimique à Skikda, nuage toxique de 200km!",
            "COMPLOT: Les autorités laissent les gens mourir volontairement à Chlef!",
            "ALERTE ROUGE: Le barrage de Beni Haroun va exploser cette nuit!",
            "INFO CACHÉE: Des dizaines de milliers de morts à Boumerdès, on ne dit rien!",
            "DANGER MORTEL: L'eau du robinet contient du poison dans 10 wilayas!",
            "RÉVÉLATION: Le gouvernement a provoqué le séisme avec des bombes!",
            "ATTENTION: Un tsunami nucléaire va toucher toute la Méditerranée!",
            "URGENT: 50 000 morts dans le séisme d'Alger, les médias mentent!",
            "BREAKING: Un astéroïde va percuter l'Algérie dans 48 heures, c'est confirmé!",
            "CATASTROPHE: La terre va s'ouvrir et engloutir toute la ville de Sétif!",
            "EXCLUSIF: Des radiations mortelles détectées dans toute la Kabylie!",
            "URGENT: Tempête cataclysmique va raser 20 wilayas cette nuit!",
            "FUYEZ IMMÉDIATEMENT: Le Sahara va être sous les eaux demain matin!",
            "COMPLOT MONDIAL: Les chemtrails provoquent les inondations en Algérie!",
            "DANGER: Des milliers de scorpions géants envahissent Ghardaïa!",
            "ALERTE: L'armée bombarde des civils sous couvert de catastrophe naturelle!",
            "SCANDALE: Les hôpitaux refusent les victimes du séisme par manque de budget!",
            "URGENT: Volcan sous-marin en éruption au large d'Annaba, évacuation totale!",
            "CATASTROPHE IMMINENTE: Tous les bâtiments d'Alger vont s'effondrer!",
            "EXCLUSIF: Des milliers de poissons morts empoisonnés sur la côte oranaise!",
            "FAKE NEWS: Tremblement de terre prévu à 14h32 exactement demain à Blida!",
            "COMPLOT: Le barrage est fissuré depuis des années, les autorités savaient!",
            "ATTENTION MAXIMALE: Glissement de terrain géant va emporter la Casbah!",
            "PANIQUE: Gaz toxique invisible se propage dans toute la wilaya de Tipaza!",
            "URGENT: Les pompiers sont en grève, personne ne vient éteindre le feu!",
            "RÉVÉLATION CHOC: 10 000 disparus non comptabilisés après la crue de Bab el Oued!",
            "DANGER DE MORT: Ne sortez pas, pluie acide qui brûle la peau à Annaba!",
        ]
        
        # ============================================================
        # DATASET ENRICHI: 100+ phrases réelles de sources officielles algériennes
        # ============================================================
        real_texts = [
            "Alerte météo: Prévisions de pluies abondantes dans la région d'Alger pour les prochaines 24 heures.",
            "Un séisme de magnitude 4.5 a été ressenti ce matin dans la wilaya de Boumerdès, aucun dégât signalé.",
            "Les services de protection civile interviennent pour un incendie de forêt près de Tizi Ouzou.",
            "Alerte inondation: Le niveau de l'Oued El Harrach a dépassé le seuil d'alerte.",
            "Un accident de la circulation impliquant un camion citerne a provoqué une fuite de carburant sur l'autoroute Est-Ouest.",
            "La protection civile demande aux citoyens de rester vigilants face aux risques de glissements de terrain.",
            "Incendie déclaré dans un immeuble résidentiel à Oran, les secours sont sur place.",
            "Les autorités locales ont activé le plan ORSEC suite aux intempéries dans la région de Béjaïa.",
            "Un glissement de terrain a bloqué la route nationale RN12 entre Blida et Médéa.",
            "La direction de la météorologie annonce des vents forts pouvant atteindre 80 km/h sur le littoral.",
            "Evacuation préventive de 50 familles suite à la montée des eaux dans la commune de Tipaza.",
            "Les équipes de secours ont maîtrisé l'incendie après 4 heures d'intervention à Chlef.",
            "Le centre anti-poison recommande des précautions suite aux inondations dans la région.",
            "Les pompiers sont intervenus pour secourir des personnes bloquées par les crues.",
            "La wilaya d'Annaba a mobilisé des moyens supplémentaires face à la montée des eaux.",
            "Un séisme de magnitude 3.2 a été enregistré au large de la côte de Béjaïa.",
            "Alerte: Risque de crues soudaines dans les oueds de la région des Aurès.",
            "Les services compétents surveillent l'évolution de la situation météorologique.",
            "Un accident industriel a causé une dispersion de fumées toxiques, les riverains sont invités à rester calfeutrés.",
            "La protection civile organise une journée de sensibilisation aux risques naturels.",
            # Sources type APS (Algérie Presse Service)
            "APS: Le CRAAG enregistre une secousse de magnitude 3.7 dans la wilaya de Médéa sans dégâts matériels.",
            "APS: La protection civile a procédé à l'extinction d'un feu de broussailles à El Tarf après une intervention de 3 heures.",
            "Communiqué officiel: Aucune victime signalée suite au séisme de magnitude 4.1 ressenti à Jijel.",
            "BMS: Bulletin météorologique spécial prévoyant des chutes de neige au-dessus de 1000m dans les Aurès.",
            "La direction de l'hydraulique confirme que le taux de remplissage des barrages est normal malgré les pluies.",
            "Le wali de Boumerdès préside une réunion de coordination pour évaluer les dégâts des intempéries.",
            "Protection Civile: Intervention achevée à Bab El Oued, 3 familles évacuées par mesure de précaution.",
            "ONM: Alerte orange pour vents forts pouvant atteindre 70 km/h sur les wilayas côtières de l'ouest.",
            "L'hôpital de Ain Defla renforce ses capacités d'accueil en prévision des urgences saisonnières.",
            "Le CRAAG dément formellement les rumeurs d'un séisme majeur imminent en Algérie.",
            "Communiqué: Feu de forêt maîtrisé à Kherrata après intervention conjointe protection civile et armée.",
            "Les services de sécurité appellent les citoyens à ne pas relayer les rumeurs non confirmées.",
            "BMS: Pluies orageuses attendues sur le centre et l'est du pays durant les prochaines 48 heures.",
            "Le ministère de l'Intérieur active le dispositif de prévention des feux de forêt pour la saison estivale.",
            "Rapport trimestriel: 45 interventions de la protection civile enregistrées dans la wilaya d'Oran.",
            "Communiqué préfectoral: Travaux de confortement des berges de l'oued Sebaou à Tizi Ouzou.",
            "La Gendarmerie nationale intervient pour sécuriser le périmètre d'un effondrement de terrain à Djelfa.",
            "APS: Session de formation aux gestes de premiers secours organisée à l'université de Constantine.",
            "Alerte jaune: Risque modéré d'incendie de forêt dans les wilayas de l'est pour les jours à venir.",
            "Le CRAAG étudie l'activité sismique accrue dans la région de Zemmouri pour émettre des recommandations.",
            "Protection civile d'Alger: Exercice de simulation d'évacuation réalisé avec succès au quartier Bab el Oued.",
            "APS: Le bilan des intempéries dans la wilaya de Mostaganem fait état de dégâts matériels légers.",
            "Communiqué: Réouverture de la RN5 après déblaiement d'un glissement de terrain à Lakhdaria.",
            "APS: La direction de l'environnement alerte sur les risques de pollution après les inondations.",
            "BMS: Températures pouvant dépasser 45°C dans les wilayas du sud pour la semaine prochaine.",
            "Opération de reboisement lancée dans la wilaya de Bouira pour prévenir l'érosion des sols.",
            "Le CHU Mustapha d'Alger met en place un dispositif d'urgence renforcé pour la période estivale.",
            "APS: La Marine nationale participe aux opérations de recherche au large de Skikda.",
            "Communiqué: Renforcement du dispositif de surveillance des oueds dans 12 wilayas du nord.",
            "Rapport mensuel de la protection civile: 312 interventions dont 45 liées aux catastrophes naturelles.",
            "APS: La Sonelgaz rétablit l'alimentation électrique après les coupures dues aux intempéries à Sétif.",
            "Le directeur de la protection civile inspecte les casernes de la wilaya de Tlemcen.",
            "Communiqué: Fermeture temporaire du port de pêche de Cherchell pour cause de forte houle.",
            "BMS: Alerte orange aux pluies torrentielles et risques d'inondations dans le bassin de la Soummam.",
            "APS: Le réseau national de surveillance sismique du CRAAG dispose de 68 stations opérationnelles.",
            "Exercice régional de simulation d'intervention en cas de séisme organisé à Blida.",
            "La direction de la santé de Batna met en alerte la médecine de catastrophe pour les risques sismiques.",
            "APS: Installation de nouveaux capteurs pluviométriques dans les wilayas du sud-est.",
            "Rapport du CRAAG: 156 secousses sismiques enregistrées sur le territoire national ce trimestre.",
            "Communiqué officiel: Aucun risque de débordement signalé au barrage de Keddara malgré les pluies.",
            "APS: Les pompiers de Saïda maîtrisent un incendie de steppe après 6 heures d'intervention.",
            "Les autorités de la wilaya de Mila organisent une campagne de sensibilisation aux risques sismiques.",
            "APS: Opération de curage des oueds et canalisations pluviales dans la commune d'Hussein Dey.",
            "Une secousse tellurique de 2.8 a été enregistrée dans la région de M'sila sans faire de victimes.",
            "Le service national de météorologie émet un bulletin d'alerte pour tempête de sable dans le sud.",
            "APS: La protection civile de Bejaia intervient pour un éboulement rocheux sur la route de Kherrata.",
            "Le wali de Mascara préside une cellule de crise suite aux précipitations exceptionnelles.",
            "Communiqué: Plan d'urgence activé dans les communes rurales de la wilaya de Ain Temouchent.",
            "APS: Trois familles relogées temporairement suite à un glissement de terrain à Constantine.",
            "La direction de l'agriculture de Tiaret évalue les pertes agricoles causées par la grêle.",
            "APS: Reprise progressive de la navigation maritime au port d'Arzew après la tempête.",
            "Communiqué DGPC: Révision du plan national de prévention et gestion des catastrophes naturelles.",
            "APS: Les hôpitaux de la wilaya de Jijel renforcent leurs stocks de médicaments d'urgence.",
            "Conférence sur la résilience urbaine face aux catastrophes naturelles organisée à l'USTHB Alger.",
            "APS: Équipement de la wilaya de Ghardaïa en matériel de pompage pour prévenir les inondations.",
            "Le ministère de la Santé lance une campagne de vaccination post-inondation dans le nord du pays.",
            "APS: Signature d'une convention entre la DGPC et l'université de Boumerdès pour la recherche sismique.",
            "Communiqué ONEDD: Surveillance renforcée de la qualité de l'air dans les zones industrielles de Skikda.",
            "Les sapeurs-pompiers d'Alger ont secouru 12 personnes bloquées dans un ascenseur lors de la coupure.",
            "APS: L'ANP déploie des moyens aériens pour surveiller les foyers d'incendie dans les forêts de Khenchela.",
            "Le CRAAG rassure la population: l'activité sismique observée est normale pour cette zone géologique.",
            "APS: Livraison de 500 tentes et 2000 couvertures à la wilaya de M'sila après les crues soudaines.",
            "Bulletin ONM: Conditions météo favorables au retour au calme dans les 24 prochaines heures.",
            "APS: Formation de 250 volontaires au secourisme dans les zones rurales de la wilaya de Bordj Bou Arreridj.",
            "Communiqué: L'état des pistes forestières est évalué avant la campagne de prévention des feux d'été.",
            "APS: Inauguration d'un centre régional de veille et d'alerte précoce aux catastrophes à Sétif.",
            "Le ministre de l'Intérieur salue le professionnalisme des agents de la protection civile.",
            "APS: Bilan positif de l'exercice de simulation tsunami réalisé dans la wilaya de Tipaza.",
            "Séisme de magnitude 3.5 enregistré dans la région de Bouira, aucune victime à déplorer.",
            "Distribution d'aides alimentaires aux familles sinistrées par les inondations dans la wilaya de Mostaganem.",
            "Lancement de travaux de réhabilitation des ponts endommagés par les intempéries sur la RN1.",
            "APS: Le parc national du Djurdjura rouvre après l'extinction complète de l'incendie qui a touché 200 hectares.",
            "Station sismique du CRAAG à Zemmouri: activité normale, pas de répliques significatives.",
            "Le directeur de la santé de Batna appelle à la vigilance face aux maladies hydriques post-inondation.",
        ]
        
        texts = fake_texts + real_texts
        labels = [1]*len(fake_texts) + [0]*len(real_texts)
        
        # Vectorisation du texte avec TF-IDF
        self.vectorizer_fake_news = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X = self.vectorizer_fake_news.fit_transform(texts)
        
        # Train test split pour éviter l'overfitting (5.1)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        
        # Entraînement du modèle SGDClassifier (supporte partial_fit pour apprentissage continu - 5.3)
        self.modele_fake_news = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
        self.modele_fake_news.fit(X_train, y_train)
        
        # Évaluation sur le test set (5.1)
        y_pred = self.modele_fake_news.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   ✅ Modèle Fake News entraîné (Accuracy sur test: {accuracy:.2%})")

    def renforcer_apprentissage_fake_news(self, texte, est_fake=False):
        """
        Apprentissage continu du modèle Fake News lors d'une validation/rejet Responsable. (5.3)
        """
        if self.modele_fake_news and self.vectorizer_fake_news:
            try:
                X_new = self.vectorizer_fake_news.transform([texte])
                y_new = [1 if est_fake else 0]
                self.modele_fake_news.partial_fit(X_new, y_new, classes=[0, 1])
                print(f"   🔄 Modèle IA mis à jour avec une nouvelle donnée (Est Fake: {est_fake})")
            except Exception as e:
                print(f"   ❌ Erreur lors de l'apprentissage continu: {str(e)}")
    
    def detecter_fake_news(self, texte_signalement, historique_utilisateur=None, signalements_recents=None):
        """
        =====================================================
        FONCTION 1: DÉTECTION DES FAUSSES NOUVELLES
        =====================================================
        
        Analyse un signalement de catastrophe pour déterminer
        s'il s'agit d'une fausse nouvelle ou d'un signalement légitime.
        Utilise CamemBERT (5.5) + TF-IDF + historique citoyen + recoupement.
        
        Args:
            texte_signalement (str): Le texte du signalement du citoyen
            historique_utilisateur (dict): { 'taux_fiabilite': 0-100 }
            signalements_recents (list): Autres signalements dans la même zone/groupe d'heures
            
        Returns:
            dict: Résultat de l'analyse contenant:
                - est_fake (bool): True si c'est une fake news
                - confiance (float): Pourcentage de confiance (0-100)
                - motifs (list): Liste des motifs de suspicion
                - recommandation (str): Action recommandée
        """
        # Indicateurs de fake news textuels
        indicateurs_suspects = {
            "mots_exageration": [
                "milliers de morts", "entièrement détruit", "catastrophe totale",
                "millions", "total", "géante", "massive", "apocalyptique",
                "dévastateur", "infernal", "monstre", "gigantesque", "anéanti", "effondrement total"
            ],
            "mots_urgence_excessive": [
                "immédiatement", "fuyez", "sauvez-vous", "URGENT", "BREAKING",
                "EXCLUSIF", "SECRET", "CATASTROPHE", "DANGER DE MORT", "VITE", "FIN DU MONDE"
            ],
            "chiffres_irrealistes": [
                "magnitude 9", "100 mètres", "60 degrés", "tous les ponts",
                "tous les barrages", "toute l'Algérie", "tout le nord", "1000 morts",
                "50 000 morts", "10 000 disparus"
            ],
            "mots_conspiration": [
                "cache", "caché", "secret", "vérité", "mente", "ment",
                "gouvernement cache", "autorités cachent", "ils ne veulent pas", "complot",
                "scandale", "révélation", "on nous cache"
            ]
        }
        
        # Analyser les indicateurs heuristiques
        texte_lower = texte_signalement.lower()
        motifs = []
        score_suspicion = 0
        
        for categorie, mots in indicateurs_suspects.items():
            for mot in mots:
                if mot.lower() in texte_lower:
                    score_suspicion += 1
                    motifs.append(f"Présence de terme suspect: '{mot}'")
        
        # Prédiction avec le modèle ML (TF-IDF + SGDClassifier)
        texte_vectorise = self.vectorizer_fake_news.transform([texte_signalement])
        probabilite_fake_ml = self.modele_fake_news.predict_proba(texte_vectorise)[0][1]
        
        # 🆕 (5.5) Analyse CamemBERT pour contexte sémantique
        score_camembert = 0.5  # neutre par défaut
        if self.camembert_pipeline:
            try:
                # CamemBERT sentiment: texte alarmiste/négatif = suspect
                cam_result = self.camembert_pipeline(texte_signalement[:512])
                if cam_result and len(cam_result) > 0:
                    results = cam_result[0] if isinstance(cam_result[0], list) else cam_result
                    for r in results:
                        label = r.get('label', '').lower()
                        score_val = r.get('score', 0)
                        # Texte très négatif = potentiellement alarmiste/fake
                        if '1' in label or 'negative' in label or 'star' in label:
                            score_camembert = score_val * 0.7
                        elif '5' in label or 'positive' in label:
                            score_camembert = 1.0 - score_val * 0.3
                    motifs.append(f"🧠 Analyse CamemBERT: sentiment = {score_camembert:.2f}")
            except Exception as e:
                motifs.append(f"CamemBERT non disponible: {str(e)[:50]}")

        # Ajustement avec historique citoyen (Scoring de fiabilité)
        ajustement_fiabilite = 0.0
        if historique_utilisateur:
            fiabilite = historique_utilisateur.get('taux_fiabilite', historique_utilisateur.get('score', 50))
            if fiabilite < 30:
                ajustement_fiabilite = 0.2
                motifs.append("Utilisateur à faible fiabilité (historique suspect)")
            elif fiabilite > 80:
                ajustement_fiabilite = -0.2
                motifs.append("Bonus de fiabilité (citoyen historiquement sûr)")

        # Ajustement avec cohérence (Recoupement)
        ajustement_coherence = 0.0
        if signalements_recents and len(signalements_recents) > 0:
            ajustement_coherence = -0.15
            motifs.append(f"Recoupement validé avec {len(signalements_recents)} autres signalements récents.")

        # Score combiné final (TF-IDF 40% + CamemBERT 20% + heuristiques 40%)
        score_base = (probabilite_fake_ml * 0.4) + (score_camembert * 0.2) + (min(score_suspicion / 5, 1) * 0.4)
        score_final = score_base + ajustement_fiabilite + ajustement_coherence
        
        # Borner entre 0 et 1
        score_final = max(0.0, min(1.0, score_final))
        
        est_fake = bool(score_final > 0.5)
        confiance = round(float(abs(score_final - 0.5) * 200), 1)
        if confiance > 100: confiance = 100.0
        
        # Score de fiabilité inversé (100 = très fiable, 0 = très suspect)
        score_fiabilite = round(float((1.0 - score_final) * 100), 1)
        
        # Recommandation
        if est_fake:
            recommandation = "⚠️ SIGNALEMENT REJETÉ: Forte probabilité de fausse nouvelle. Vérification manuelle requise."
        elif score_final > 0.35:
            recommandation = "🔍 SIGNALEMENT DOUTEUX: Analyse supplémentaire recommandée. Consulter d'autres sources."
        else:
            recommandation = "✅ SIGNALEMENT LÉGITIME: Le signalement semble fiable. Procéder à l'analyse IA."
        
        resultat = {
            "est_fake": bool(est_fake),
            "confiance": float(confiance),
            "score_fiabilite": float(score_fiabilite),
            "probabilite_ml": round(float(probabilite_fake_ml * 100), 1),
            "score_camembert": round(float(score_camembert * 100), 1),
            "score_indicateurs": int(score_suspicion),
            "motifs": motifs if motifs else ["Aucun motif de suspicion détecté"],
            "recommandation": recommandation,
            "statut": "FAUSSE NOUVELLE" if est_fake else "LÉGITIME"
        }
        
        return resultat
    
    # =========================================================================
    # FONCTION 2: PRÉDICTION DU NIVEAU DE RISQUE
    # =========================================================================
    
    def _entrainer_modele_risque(self):
        """
        Entraîne le modèle de prédiction du niveau de risque
        en utilisant les données historiques avec train_test_split (5.1).
        """
        print("\n📊 Entraînement du modèle de prédiction du niveau de risque...")
        
        if self.donnees_historiques is None or len(self.donnees_historiques) == 0:
            print("   ⚠️ Pas de données disponibles pour l'entraînement")
            return
        
        # Préparer les features
        features = [
            "type_catastrophe", "zone", "mois", "population_zone",
            "superficie_affectee_km2", "alerte_prealable_heures",
            "infrastructures_endommagees", "evacuation_necessaire"
        ]
        
        # Encoder les variables catégorielles
        df = self.donnees_historiques.copy()
        
        for col in ["type_catastrophe", "zone"]:
            self.label_encoders[col] = LabelEncoder()
            df[col + "_enc"] = self.label_encoders[col].fit_transform(df[col])
        
        features_enc = [
            "type_catastrophe_enc", "zone_enc", "mois", "population_zone",
            "superficie_affectee_km2", "alerte_prealable_heures",
            "infrastructures_endommagees", "evacuation_necessaire"
        ]
        
        X = df[features_enc].fillna(0)
        y = df["niveau_risque"]
        
        # Normaliser
        X_scaled = self.scaler.fit_transform(X)
        
        # Train test split (5.1)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Entraîner le modèle
        self.modele_risque = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.modele_risque.fit(X_train, y_train)
        
        # Évaluation sur le test set (5.1)
        y_pred = self.modele_risque.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   ✅ Modèle de risque entraîné (Accuracy sur test: {accuracy:.2%})")
        
        # (5.6) Initialiser SHAP si disponible
        if HAS_SHAP:
            try:
                self.shap_explainer_risque = shap.TreeExplainer(self.modele_risque)
                print("   ✅ SHAP TreeExplainer initialisé pour l'explicabilité")
            except Exception as e:
                print(f"   ⚠️ SHAP non initialisé: {e}")
                self.shap_explainer_risque = None
    
    def predire_niveau_risque(self, type_catastrophe, zone, mois, 
                              population_zone, superficie_affectee,
                              alerte_prealable=0, infrastructures_endommagees=0,
                              evacuation=False, heure=None):
        """
        =====================================================
        FONCTION 2: PRÉDICTION DU NIVEAU DE RISQUE
        =====================================================
        
        Prédit le niveau de risque d'une catastrophe.
        Niveaux possibles: Faible, Moyen, Élevé, Critique
        Prend en compte l'heure actuelle (5.4).
        
        Args:
            type_catastrophe (str): Type de la catastrophe
            zone (str): Zone géographique (wilaya)
            mois (int): Mois de l'année (1-12)
            population_zone (int): Population de la zone
            superficie_affectee (float): Superficie affectée en km²
            alerte_prealable (int): Heures d'alerte préalable
            infrastructures_endommagees (int): Nombre d'infrastructures touchées
            evacuation (bool): Évacuation nécessaire
            heure (int): Heure actuelle (0-23) pour facteur nuit (5.4)
            
        Returns:
            dict: Résultat de la prédiction
        """
        # (5.4) Si heure non fournie, prendre l'heure actuelle
        if heure is None:
            heure = datetime.now().hour
            
        # Préparer les données d'entrée
        try:
            type_enc = self.label_encoders["type_catastrophe"].transform([type_catastrophe])[0]
        except:
            type_enc = 0
            
        try:
            zone_enc = self.label_encoders["zone"].transform([zone])[0]
        except:
            zone_enc = 0
        
        donnees_entree = np.array([[
            type_enc, zone_enc, mois, population_zone,
            superficie_affectee, alerte_prealable,
            infrastructures_endommagees, int(evacuation)
        ]])
        
        # Normaliser
        donnees_scaled = self.scaler.transform(donnees_entree)
        
        # Prédiction
        if self.modele_risque:
            niveau_predit = self.modele_risque.predict(donnees_scaled)[0]
            probabilites = self.modele_risque.predict_proba(donnees_scaled)[0]
        else:
            niveau_predit = self._predire_risque_heuristique(
                type_catastrophe, population_zone, superficie_affectee, heure
            )
            probabilites = [0.25, 0.25, 0.25, 0.25]
        
        # (5.4) Ajustement nocturne
        if heure < 6 or heure >= 22:
            niveaux_ordre = ["Faible", "Moyen", "Élevé", "Critique"]
            idx = niveaux_ordre.index(niveau_predit) if niveau_predit in niveaux_ordre else 1
            if idx < len(niveaux_ordre) - 1:
                niveau_predit = niveaux_ordre[min(idx + 1, len(niveaux_ordre) - 1)]
        
        # Identifier les facteurs principaux
        facteurs = self._analyser_facteurs_risque(
            type_catastrophe, population_zone, superficie_affectee,
            alerte_prealable, infrastructures_endommagees, heure
        )
        
        # (5.6) SHAP explicabilité
        shap_raisons = []
        if HAS_SHAP and self.shap_explainer_risque:
            try:
                shap_values = self.shap_explainer_risque.shap_values(donnees_scaled)
                feature_names = [
                    "Type catastrophe", "Zone", "Mois", "Population zone",
                    "Superficie affectée", "Alerte préalable",
                    "Infrastructures endommagées", "Évacuation"
                ]
                # Trouver l'index de la classe prédite
                classes = list(self.modele_risque.classes_)
                class_idx = classes.index(niveau_predit) if niveau_predit in classes else 0
                sv = shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0]
                
                # Trier par importance absolue
                importance = [(feature_names[i], sv[i]) for i in range(len(sv))]
                importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for fname, fval in importance[:5]:
                    direction = "↑ augmente" if fval > 0 else "↓ diminue"
                    shap_raisons.append(f"📊 {fname}: {direction} le risque ({abs(fval):.3f})")
            except Exception as e:
                shap_raisons = [f"SHAP non disponible: {str(e)[:60]}"]
        
        # Générer des recommandations basées sur le niveau
        recommandations = self._generer_recommandations_risque(niveau_predit)
        
        # Formater les probabilités
        classes = self.modele_risque.classes_ if self.modele_risque else ["Faible", "Moyen", "Élevé", "Critique"]
        prob_dict = {classes[i]: float(probabilites[i] * 100) for i in range(len(classes))}
        
        return {
            "niveau_risque": niveau_predit,
            "probabilites": prob_dict,
            "confiance_max": float(max(probabilites) * 100),
            "facteurs_principaux": facteurs,
            "shap_raisons": shap_raisons,
            "recommandations": recommandations,
            "analyse_historique": self._get_contexte_historique(type_catastrophe, zone),
            "heure_analyse": heure,
            "facteur_nuit": heure < 6 or heure >= 22
        }
    
    def _predire_risque_heuristique(self, type_catastrophe, population, superficie, heure=12):
        """Prédiction de secours basée sur des règles heuristiques."""
        score = 0
        
        types_dangereux = ["Séisme", "Tsunami", "Accident chimique", "Éruption volcanique"]
        if type_catastrophe in types_dangereux:
            score += 2
            
        # (5.4) Facteur nuit aggravant
        if heure < 6 or heure >= 22:
            score += 1
        
        if population > 300000:
            score += 2
        elif population > 100000:
            score += 1
        
        if superficie > 20:
            score += 2
        elif superficie > 5:
            score += 1
        
        if score >= 5:
            return "Critique"
        elif score >= 3:
            return "Élevé"
        elif score >= 1:
            return "Moyen"
        else:
            return "Faible"
    
    def _analyser_facteurs_risque(self, type_catastrophe, population, superficie,
                                   alerte_prealable, infrastructures, heure=12):
        """Analyse les facteurs influençant le niveau de risque."""
        facteurs = []
        
        types_critiques = ["Séisme", "Tsunami", "Accident chimique", "Éruption volcanique"]
        if type_catastrophe in types_critiques:
            facteurs.append(f"⚠️ Type critique: {type_catastrophe}")
        
        if population > 300000:
            facteurs.append(f"⚠️ Zone très peuplée: {population:,} habitants")
        elif population > 100000:
            facteurs.append("⚡ Zone modérément peuplée")
        
        if superficie > 20:
            facteurs.append(f"⚠️ Zone affectée étendue: {superficie} km²")
        
        if alerte_prealable == 0:
            facteurs.append("🔴 Aucune alerte préalable - Risque maximum")
        elif alerte_prealable < 6:
            facteurs.append("🟡 Temps de réaction limité")
        else:
            facteurs.append("🟢 Alerte préalable suffisante")
        
        if infrastructures > 10:
            facteurs.append(f"⚠️ Infrastructures endommagées: {infrastructures}")
            
        # (5.4) Facteur nuit
        if heure < 6 or heure >= 22:
            facteurs.append(f"🌑 Nuit ({heure}h) : Visibilité et opérations complexes (Risque aggravé)")
        
        return facteurs if facteurs else ["📊 Analyse standard en cours"]
    
    def _generer_recommandations_risque(self, niveau):
        """Génère des recommandations basées sur le niveau de risque."""
        recommandations = {
            "Faible": [
                "Surveillance standard de la situation",
                "Mettre en alerte les équipes de proximité",
                "Préparer les ressources en cas d'évolution"
            ],
            "Moyen": [
                "Activer le niveau d'alerte orange",
                "Dépêcher des équipes d'évaluation sur site",
                "Préparer les plans d'évacuation préventive",
                "Informer les hôpitaux de proximité"
            ],
            "Élevé": [
                "🚨 ACTIVER LE PLAN ORSEC",
                "Déploiement immédiat des équipes de secours",
                "Évacuation préventive des zones à risque",
                "Mobiliser les hôpitaux et préparer les lits d'urgence",
                "Activer les canaux de communication d'urgence"
            ],
            "Critique": [
                "🔴 ALERTE ROUGE MAXIMALE",
                "Déclaration d'état d'urgence",
                "Mobilisation TOTALE de toutes les ressources disponibles",
                "Demande d'assistance nationale et internationale",
                "Évacuation immédiate et obligatoire des zones touchées",
                "Activation de tous les centres d'hébergement d'urgence"
            ]
        }
        return recommandations.get(niveau, ["Analyse en cours"])
    
    def _get_contexte_historique(self, type_catastrophe, zone):
        """Récupère le contexte historique pour ce type de catastrophe."""
        if self.donnees_historiques is None:
            return ["Aucune donnée historique disponible"]
        
        historique_type = self.donnees_historiques[
            self.donnees_historiques["type_catastrophe"] == type_catastrophe
        ]
        
        historique_zone = self.donnees_historiques[
            self.donnees_historiques["zone"] == zone
        ]
        
        contexte = []
        
        if len(historique_type) > 0:
            avg_victimes = historique_type["nb_victimes_reel"].mean()
            contexte.append(f"Catastrophes similaires: {len(historique_type)} cas historiques")
            contexte.append(f"Victimes moyennes (type): {avg_victimes:.0f}")
        
        if len(historique_zone) > 0:
            contexte.append(f"Catastrophes dans cette zone: {len(historique_zone)} cas")
        
        return contexte if contexte else ["Premier cas de ce type dans cette zone"]
    
    # =========================================================================
    # FONCTION 3: ESTIMATION DE L'IMPACT
    # =========================================================================
    
    def _entrainer_modele_impact(self):
        """
        Entraîne le modèle d'estimation de l'impact
        pour prédire le nombre de victimes et blessés.
        Utilise train_test_split (5.1).
        """
        print("\n📈 Entraînement du modèle d'estimation de l'impact...")
        
        if self.donnees_historiques is None or len(self.donnees_historiques) == 0:
            print("   ⚠️ Pas de données disponibles pour l'entraînement")
            return
        
        df = self.donnees_historiques.copy()
        
        # Encoder les variables catégorielles
        for col in ["type_catastrophe", "zone", "niveau_risque"]:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col + "_enc"] = self.label_encoders[col].fit_transform(df[col])
        
        features = [
            "type_catastrophe_enc", "zone_enc", "niveau_risque_enc",
            "population_zone", "superficie_affectee_km2", "duree_heures",
            "infrastructures_endommagees", "evacuation_necessaire"
        ]
        
        X = df[features].fillna(0)
        
        # Train test split (5.1)
        X_train, X_test, _, _ = train_test_split(X, df["nb_victimes_reel"], test_size=0.2, random_state=42)
        
        # Use indices from the split for all targets
        train_idx = X_train.index
        test_idx = X_test.index
        
        # Modèle pour les victimes
        y_victimes = df["nb_victimes_reel"]
        self.modele_impact_victimes = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.modele_impact_victimes.fit(X.loc[train_idx], y_victimes.loc[train_idx])
        
        # Modèle pour les blessés
        y_blesses = df["nb_blesses_reel"]
        self.modele_impact_blesses = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.modele_impact_blesses.fit(X.loc[train_idx], y_blesses.loc[train_idx])
        
        # Modèle pour les sinistrés
        y_sinistres = df["nb_sinistres_reel"]
        self.modele_impact_sinistres = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.modele_impact_sinistres.fit(X.loc[train_idx], y_sinistres.loc[train_idx])
        
        print(f"   ✅ Modèles d'impact entraînés (3 modèles, test_size=0.2)")
    
    def estimer_impact(self, type_catastrophe, zone, niveau_risque,
                       population_zone, superficie_affectee, duree_estimee_heures,
                       infrastructures_endommagees=0, evacuation=False):
        """
        =====================================================
        FONCTION 3: ESTIMATION DE L'IMPACT
        =====================================================
        
        Estime l'impact potentiel d'une catastrophe en termes de:
        - Nombre de victimes potentielles
        - Nombre de blessés estimés
        - Nombre de sinistrés (personnes déplacées)
        - Dégâts matériels estimés
        """
        # Préparer les données d'entrée
        try:
            type_enc = self.label_encoders["type_catastrophe"].transform([type_catastrophe])[0]
        except:
            type_enc = 0
            
        try:
            zone_enc = self.label_encoders["zone"].transform([zone])[0]
        except:
            zone_enc = 0
            
        try:
            risque_enc = self.label_encoders["niveau_risque"].transform([niveau_risque])[0]
        except:
            risque_enc = 1
        
        donnees_entree = np.array([[
            type_enc, zone_enc, risque_enc, population_zone,
            superficie_affectee, duree_estimee_heures,
            infrastructures_endommagees, int(evacuation)
        ]])
        
        # Prédictions
        try:
            victimes_estimees = max(0, int(self.modele_impact_victimes.predict(donnees_entree)[0]))
            blesses_estimes = max(0, int(self.modele_impact_blesses.predict(donnees_entree)[0]))
            sinistres_estimes = max(0, int(self.modele_impact_sinistres.predict(donnees_entree)[0]))
        except:
            facteur_risque = {"Faible": 0.001, "Moyen": 0.005, "Élevé": 0.02, "Critique": 0.05}
            facteur = facteur_risque.get(niveau_risque, 0.005)
            
            if evacuation:
                facteur *= 0.3
            
            victimes_estimees = max(0, int(population_zone * superficie_affectee / 100 * facteur))
            blesses_estimes = max(0, int(victimes_estimees * 3))
            sinistres_estimes = max(0, int(population_zone * superficie_affectee / 10 * facteur * 10))
        
        # Estimer les dégâts matériels
        degats_estimes = self._estimer_degats(
            type_catastrophe, niveau_risque, superficie_affectee, infrastructures_endommagees
        )
        
        # Déterminer la gravité globale
        gravite = self._determiner_gravite(victimes_estimees, blesses_estimes, sinistres_estimes)
        
        # Estimer les ressources requises
        ressources = self._estimer_ressources_requises(
            victimes_estimees, blesses_estimes, sinistres_estimes, gravite
        )
        
        return {
            "victimes_estimees": {
                "valeur": victimes_estimees,
                "intervalle_confiance": [max(0, int(victimes_estimees * 0.6)), 
                                         int(victimes_estimees * 1.4)],
                "niveau": self._categoriser_victimes(victimes_estimees)
            },
            "blesses_estimes": {
                "valeur": blesses_estimes,
                "intervalle_confiance": [max(0, int(blesses_estimes * 0.7)),
                                         int(blesses_estimes * 1.3)],
                "niveau": self._categoriser_blesses(blesses_estimes)
            },
            "sinistres_estimes": {
                "valeur": sinistres_estimes,
                "intervalle_confiance": [max(0, int(sinistres_estimes * 0.7)),
                                         int(sinistres_estimes * 1.3)]
            },
            "degats_materiaux_estimes": degats_estimes,
            "gravite_globale": gravite,
            "ressources_requises": ressources,
            "impact_evacuation": "L'évacuation réduit l'impact de ~70%" if evacuation else "Aucune évacuation - Impact maximal potentiel"
        }
    
    def _estimer_degats(self, type_catastrophe, niveau_risque, superficie, infrastructures):
        """Estime les dégâts matériels en Dinars Algériens."""
        cout_base = {
            "Inondation": 5000000,
            "Incendie": 8000000,
            "Séisme": 20000000,
            "Accident chimique": 15000000,
            "Tempête": 3000000,
            "Glissement de terrain": 6000000,
            "Tsunami": 25000000,
            "Éruption volcanique": 30000000
        }
        
        multiplicateur = {
            "Faible": 1,
            "Moyen": 3,
            "Élevé": 8,
            "Critique": 20
        }
        
        base = cout_base.get(type_catastrophe, 5000000)
        mult = multiplicateur.get(niveau_risque, 3)
        
        degats = base * mult * (superficie / 10) + (infrastructures * 500000)
        
        return {
            "estimation_da": int(degats),
            "estimation_en_millions_da": round(degats / 1000000, 2),
            "intervalle": [int(degats * 0.6), int(degats * 1.4)]
        }
    
    def _determiner_gravite(self, victimes, blesses, sinistres):
        """Détermine le niveau de gravité global."""
        score = 0
        
        if victimes > 100:
            score += 4
        elif victimes > 30:
            score += 3
        elif victimes > 10:
            score += 2
        elif victimes > 0:
            score += 1
        
        if blesses > 500:
            score += 3
        elif blesses > 100:
            score += 2
        elif blesses > 20:
            score += 1
        
        if sinistres > 1000:
            score += 3
        elif sinistres > 200:
            score += 2
        elif sinistres > 50:
            score += 1
        
        if score >= 8:
            return "CRITIQUE"
        elif score >= 5:
            return "TRÈS GRAVE"
        elif score >= 3:
            return "GRAVE"
        elif score >= 1:
            return "MODÉRÉ"
        else:
            return "MINEUR"
    
    def _categoriser_victimes(self, nb):
        if nb == 0:
            return "Aucune victime estimée"
        elif nb <= 5:
            return "Faible"
        elif nb <= 30:
            return "Modéré"
        elif nb <= 100:
            return "Élevé"
        else:
            return "Très élevé"
    
    def _categoriser_blesses(self, nb):
        if nb == 0:
            return "Aucun blessé estimé"
        elif nb <= 20:
            return "Faible"
        elif nb <= 100:
            return "Modéré"
        elif nb <= 500:
            return "Élevé"
        else:
            return "Très élevé"
    
    def _estimer_ressources_requises(self, victimes, blesses, sinistres, gravite):
        """Estime les ressources nécessaires basées sur l'impact."""
        equipes = max(2, (victimes + blesses) // 20 + 3)
        ambulances = max(1, (blesses // 5) + (victimes // 10) + 2)
        lits_hopitaux = blesses + victimes
        vehicules_logistique = max(1, sinistres // 100 + 2)
        
        ajustement = {"MINEUR": 1, "MODÉRÉ": 1.2, "GRAVE": 1.5, "TRÈS GRAVE": 2, "CRITIQUE": 3}
        mult = ajustement.get(gravite, 1)
        
        return {
            "equipes_secours": int(equipes * mult),
            "ambulances": int(ambulances * mult),
            "lits_hopitaux": int(lits_hopitaux * 1.2),
            "vehicules_logistique": int(vehicules_logistique * mult),
            "tentes_hebergement": max(0, int(sinistres / 10)),
            "kits_nourriture": int(sinistres * 3),
            "kits_eau_potable": int(sinistres * 5),
            "personnel_medical": max(2, int(blesses // 10) + 5)
        }
    
    # =========================================================================
    # FONCTION 4: PRIORISATION DES INTERVENTIONS
    # =========================================================================
    
    def prioriser_interventions(self, zones_touches, victimes_par_zone):
        """
        =====================================================
        FONCTION 4: PRIORISATION DES INTERVENTIONS
        =====================================================
        
        Classe les zones touchées selon leur urgence d'intervention.
        """
        zones_scorees = []
        alertes = []
        
        for zone in zones_touches:
            score = 0
            
            score += zone.get("victimes", 0) * 3
            score += zone.get("blesses_critiques", 0) * 10
            score += zone.get("blesses_graves", 0) * 5
            
            temps = zone.get("temps_heures", 0)
            if temps > 24:
                score += 50
            elif temps > 12:
                score += 30
            elif temps > 6:
                score += 15
            
            if not zone.get("accessible", True):
                score += 20
            
            population = zone.get("population", 100000)
            if population > 200000:
                score += 15
            
            justifications = []
            if zone.get("blesses_critiques", 0) > 0:
                justifications.append(f"🔴 {zone['blesses_critiques']} blessés critiques nécessitant intervention immédiate")
            if zone.get("victimes", 0) > 10:
                justifications.append(f"⚠️ {zone['victimes']} victimes signalées")
            if not zone.get("accessible", True):
                justifications.append("🚧 Zone inaccessible - Déblocage prioritaire")
            if temps > 12:
                justifications.append(f"⏰ {temps}h écoulées - Urgence temporelle")
            
            zones_scorees.append({
                "zone": zone.get("nom", "Zone inconnue"),
                "score_priorite": score,
                "niveau_urgence": self._get_niveau_urgence(score),
                "justifications": justifications,
                "donnees_originales": zone
            })
            
            if zone.get("blesses_critiques", 0) > 5:
                alertes.append(f"🚨 URGENCE: {zone['blesses_critiques']} blessés critiques à {zone.get('nom')}")
            if not zone.get("accessible", True) and zone.get("victimes", 0) > 0:
                alertes.append(f"🚨 ACCÈS BLOQUÉ: Zone {zone.get('nom')} inaccessible avec des victimes")
        
        zones_scorees.sort(key=lambda x: x["score_priorite"], reverse=True)
        
        ordre = [f"{i+1}. {z['zone']} (Score: {z['score_priorite']})" 
                 for i, z in enumerate(zones_scorees)]
        
        return {
            "zones_prioritaires": zones_scorees,
            "ordre_intervention": ordre,
            "nombre_zones": len(zones_scorees),
            "alertes_speciales": alertes if alertes else ["Aucune alerte spéciale"],
            "recommandation_globale": self._recommandation_globale(zones_scorees)
        }
    
    def _get_niveau_urgence(self, score):
        if score >= 200:
            return "🔴 CRITIQUE - Intervention immédiate"
        elif score >= 100:
            return "🟠 TRÈS URGENT"
        elif score >= 50:
            return "🟡 URGENT"
        elif score >= 20:
            return "🔵 MODÉRÉ"
        else:
            return "🟢 STANDARD"
    
    def _recommandation_globale(self, zones):
        """Génère une recommandation globale basée sur toutes les zones."""
        if not zones:
            return "Aucune zone à traiter"
        
        zones_critiques = [z for z in zones if z["score_priorite"] >= 200]
        
        if zones_critiques:
            return f"🚨 {len(zones_critiques)} zone(s) en état critique nécessitant une mobilisation maximale immédiate!"
        
        zones_urgentes = [z for z in zones if z["score_priorite"] >= 50]
        if zones_urgentes:
            return f"⚠️ {len(zones_urgentes)} zone(s) urgentes à traiter en priorité"
        
        return "✅ Situation sous contrôle - Interventions standard planifiées"
    
    # =========================================================================
    # FONCTION 5: OPTIMISATION DE L'ALLOCATION DES RESSOURCES
    # =========================================================================
    
    def optimiser_allocation_ressources(self, ressources_disponibles, zones_besoins):
        """
        =====================================================
        FONCTION 5: OPTIMISATION DE L'ALLOCATION DES RESSOURCES
        =====================================================
        """
        total_priorite = sum(z.get("priorite", 1) for z in zones_besoins)
        
        allocation = []
        ressources_utilisees = {
            "equipes_secours": 0, "ambulances": 0, "lits_hopitaux": 0,
            "vehicules_logistique": 0, "tentes": 0, "kits_nourriture": 0, "kits_eau": 0
        }
        
        alertes_deficit = []
        
        for zone in zones_besoins:
            proportion = zone.get("priorite", 1) / total_priorite if total_priorite > 0 else 0
            
            allocation_zone = {
                "zone": zone.get("nom"),
                "priorite": zone.get("priorite"),
                "proportion": round(proportion * 100, 1),
                "ressources_allouees": {}
            }
            
            types_ressources = ["equipes_secours", "ambulances", "lits_hopitaux",
                               "vehicules_logistique", "tentes", "kits_nourriture", "kits_eau"]
            
            for ressource in types_ressources:
                disponible = ressources_disponibles.get(ressource, 0)
                deja_alloue = ressources_utilisees.get(ressource, 0)
                reste = disponible - deja_alloue
                
                besoin = zone.get(f"{ressource}_necessaires", 0)
                
                if besoin > 0:
                    allocation_prop = int(disponible * proportion)
                    allocation_zone["ressources_allouees"][ressource] = min(allocation_prop, reste)
                else:
                    allocation_zone["ressources_allouees"][ressource] = 0
                
                ressources_utilisees[ressource] += allocation_zone["ressources_allouees"][ressource]
                
                if allocation_zone["ressources_allouees"][ressource] < besoin:
                    deficit = besoin - allocation_zone["ressources_allouees"][ressource]
                    if deficit > 0:
                        alertes_deficit.append(
                            f"⚠️ Déficit de {deficit} {ressource} pour {zone.get('nom')}"
                        )
            
            allocation.append(allocation_zone)
        
        ressources_restantes = {
            k: ressources_disponibles.get(k, 0) - v 
            for k, v in ressources_utilisees.items()
        }
        
        besoins_totaux = {}
        for zone in zones_besoins:
            for ressource in ressources_utilisees.keys():
                besoins_totaux[ressource] = besoins_totaux.get(ressource, 0) + \
                                           zone.get(f"{ressource}_necessaires", 0)
        
        total_couvert = 0
        total_besoin = 0
        for ressource, alloue in ressources_utilisees.items():
            total_couvert += alloue
            total_besoin += besoins_totaux.get(ressource, 0)
        
        taux_couverture = (total_couvert / total_besoin * 100) if total_besoin > 0 else 100
        
        return {
            "allocation_par_zone": allocation,
            "ressources_utilisees": ressources_utilisees,
            "ressources_restantes": ressources_restantes,
            "taux_couverture": round(taux_couverture, 1),
            "alertes_deficit": alertes_deficit if alertes_deficit else ["✅ Ressources suffisantes"],
            "recommandation": self._recommander_allocation(taux_couverture, alertes_deficit)
        }
    
    def _recommander_allocation(self, taux, alertes):
        """Génère des recommandations basées sur l'allocation."""
        if taux >= 90:
            return "✅ Allocation optimale - Toutes les zones sont correctement couvertes"
        elif taux >= 70:
            return "⚠️ Allocation acceptable mais des déficits mineurs existent"
        elif taux >= 50:
            return "🟠 Allocation insuffisante - Mobilisation de ressources supplémentaires requise"
        else:
            return "🔴 Déficit critique - Demande d'assistance immédiate nécessaire"
    
    # =========================================================================
    # FONCTION 6: AIDE À LA PLANIFICATION (SCÉNARIOS)
    # =========================================================================
    
    def aider_planification(self, scenarios, ressources_globales):
        """
        =====================================================
        FONCTION 6: AIDE À LA PLANIFICATION (SCÉNARIOS)
        =====================================================
        """
        plans = []
        
        for scenario in scenarios:
            risque = self.predire_niveau_risque(
                type_catastrophe=scenario.get("type_catastrophe", "Inondation"),
                zone=scenario.get("zone", "Alger"),
                mois=scenario.get("mois", 1),
                population_zone=scenario.get("population", 200000),
                superficie_affectee=scenario.get("superficie", 5)
            )
            
            impact = self.estimer_impact(
                type_catastrophe=scenario.get("type_catastrophe", "Inondation"),
                zone=scenario.get("zone", "Alger"),
                niveau_risque=risque["niveau_risque"],
                population_zone=scenario.get("population", 200000),
                superficie_affectee=scenario.get("superficie", 5),
                duree_estimee_heures=scenario.get("duree_estimee", 24)
            )
            
            plan = {
                "scenario": scenario.get("nom", "Scénario sans nom"),
                "niveau_risque": risque["niveau_risque"],
                "impact_estime": {
                    "victimes": impact["victimes_estimees"]["valeur"],
                    "blesses": impact["blesses_estimes"]["valeur"],
                    "sinistres": impact["sinistres_estimes"]["valeur"],
                    "gravite": impact["gravite_globale"]
                },
                "plan_intervention": self._generer_plan_intervention(
                    risque["niveau_risque"], 
                    impact,
                    scenario.get("type_catastrophe", "")
                ),
                "phases": self._definir_phases(risque["niveau_risque"]),
                "ressources_requises": impact["ressources_requises"],
                "probabilite_succes": self._estimer_probabilite_succes(
                    impact["ressources_requises"], 
                    ressources_globales
                )
            }
            
            plans.append(plan)
        
        scenario_recommande = max(plans, key=lambda x: x["impact_estime"]["victimes"])
        plan_global = self._generer_plan_global(plans, ressources_globales)
        
        return {
            "plans": plans,
            "scenario_recommande": scenario_recommande["scenario"],
            "plan_global": plan_global,
            "nombre_scenarios": len(scenarios),
            "recommandation_finale": f"Priorité au scénario '{scenario_recommande['scenario']}' - {scenario_recommande['niveau_risque']}"
        }
    
    def _generer_plan_intervention(self, niveau_risque, impact, type_catastrophe):
        """Génère un plan d'intervention détaillé."""
        plan = {
            "immediat": [
                "Déclencher l'alerte et mobiliser les équipes",
                "Établir le poste de commandement",
                "Première évaluation de la situation",
                "Sécuriser le périmètre de la zone",
                "Commencer les recherches et sauvetages"
            ],
            "court_terme": [
                "Évacuation des zones à risque",
                "Triage médical des blessés",
                "Mise en place des centres d'hébergement",
                "Distribution des premiers secours",
                "Coordination avec les hôpitaux"
            ],
            "moyen_terme": [
                "Recherche de survivants (si applicable)",
                "Soins médicaux continus",
                "Distribution alimentaire et en eau",
                "Évaluation des dégâts infrastructurels",
                "Restauration des services essentiels"
            ],
            "long_terme": [
                "Reconstruction des infrastructures",
                "Soutien psychologique aux victimes",
                "Aide financière aux sinistrés",
                "Évaluation et retour d'expérience",
                "Mise à jour des plans de prévention"
            ]
        }
        
        if type_catastrophe == "Inondation":
            plan["immediat"].insert(0, "Surveiller les niveaux d'eau et les crues")
            plan["court_terme"].append("Pompage des zones inondées")
        elif type_catastrophe == "Séisme":
            plan["immediat"].insert(0, "Évaluer les répliques sismiques potentielles")
            plan["immediat"].insert(1, "Déployer les équipes de recherche sous décombres")
        elif type_catastrophe == "Incendie":
            plan["immediat"].insert(0, "Déployer les moyens de lutte anti-incendie")
            plan["immediat"].append("Protéger les zones non touchées")
        elif type_catastrophe == "Accident chimique":
            plan["immediat"].insert(0, "Identifier le produit chimique impliqué")
            plan["immediat"].append("Mettre en place le périmètre de sécurité étendu")
            plan["court_terme"].insert(0, "Décontamination des personnes exposées")
        
        if niveau_risque in ["Élevé", "Critique"]:
            plan["immediat"].insert(0, "🚨 ACTIVER LE PLAN ORSEC NIVEAU ROUGE")
            plan["immediat"].append("Demander des renforts nationaux")
            if niveau_risque == "Critique":
                plan["immediat"].append("Demander l'assistance internationale")
        
        return plan
    
    def _definir_phases(self, niveau_risque):
        """Définit les phases de l'intervention selon le niveau de risque."""
        phases = {
            "Faible": [
                {"phase": "Phase 1", "duree": "2h", "action": "Évaluation"},
                {"phase": "Phase 2", "duree": "6h", "action": "Intervention légère"},
                {"phase": "Phase 3", "duree": "24h", "action": "Suivi et retour normal"}
            ],
            "Moyen": [
                {"phase": "Phase 1", "duree": "1h", "action": "Alerte et mobilisation"},
                {"phase": "Phase 2", "duree": "4h", "action": "Intervention principale"},
                {"phase": "Phase 3", "duree": "24h", "action": "Stabilisation"},
                {"phase": "Phase 4", "duree": "72h", "action": "Suivi et rétablissement"}
            ],
            "Élevé": [
                {"phase": "Phase 1", "duree": "30min", "action": "Alerte maximale"},
                {"phase": "Phase 2", "duree": "2h", "action": "Déploiement d'urgence"},
                {"phase": "Phase 3", "duree": "12h", "action": "Intervention intensive"},
                {"phase": "Phase 4", "duree": "48h", "action": "Stabilisation"},
                {"phase": "Phase 5", "duree": "7 jours", "action": "Rétablissement"}
            ],
            "Critique": [
                {"phase": "Phase 1", "duree": "Immédiat", "action": "Alerte nationale"},
                {"phase": "Phase 2", "duree": "1h", "action": "Mobilisation totale"},
                {"phase": "Phase 3", "duree": "6h", "action": "Intervention de masse"},
                {"phase": "Phase 4", "duree": "24h", "action": "Sauvetage intensif"},
                {"phase": "Phase 5", "duree": "72h", "action": "Stabilisation d'urgence"},
                {"phase": "Phase 6", "duree": "14 jours", "action": "Rétablissement progressif"}
            ]
        }
        return phases.get(niveau_risque, phases["Moyen"])
    
    def _estimer_probabilite_succes(self, besoins, ressources_disponibles):
        """Estime la probabilité de succès de l'intervention."""
        score = 0
        total = 0
        
        mapping = {
            "equipes_secours": "equipes_secours",
            "ambulances": "ambulances",
            "lits_hopitaux": "lits_hopitaux",
            "tentes_hebergement": "tentes",
            "kits_nourriture": "kits_nourriture",
            "kits_eau_potable": "kits_eau",
            "personnel_medical": "equipes_secours"
        }
        
        for besoin_key, ressource_key in mapping.items():
            if besoin_key in besoins:
                besoin = besoins[besoin_key]
                disponible = ressources_disponibles.get(ressource_key, 0)
                if besoin > 0:
                    ratio = min(disponible / besoin, 1.5)
                    score += ratio
                    total += 1
        
        if total > 0:
            prob = min((score / total) * 100, 95)
            return round(prob, 1)
        return 50.0
    
    def _generer_plan_global(self, plans, ressources_globales):
        """Génère un plan d'intervention global combinant tous les scénarios."""
        scenario_critique = max(plans, key=lambda x: x["impact_estime"]["victimes"])
        
        ressources_combinees = {}
        for plan in plans:
            for ressource, valeur in plan["ressources_requises"].items():
                if ressource not in ressources_combinees or valeur > ressources_combinees[ressource]:
                    ressources_combinees[ressource] = valeur
        
        deficit = {}
        for ressource, requis in ressources_combinees.items():
            disponible = ressources_globales.get(ressource, 0)
            if disponible < requis:
                deficit[ressource] = requis - disponible
        
        actions_globales = [
            "✅ Activer le centre de coordination unique",
            "✅ Mettre en place la communication inter-services",
            "✅ Préparer les rapports de situation réguliers (toutes les 2h)",
            "✅ Coordonner avec les autorités locales et nationales"
        ]
        
        if deficit:
            actions_globales.append("⚠️ DEMANDE DE RENFORTS - Déficits détectés:")
            for ressource, manque in deficit.items():
                actions_globales.append(f"   ❌ Manque: {manque} {ressource}")
        else:
            actions_globales.append("✅ Ressources actuelles suffisantes pour tous les scénarios")
        
        if scenario_critique["niveau_risque"] in ["Élevé", "Critique"]:
            actions_globales.append(f"🚨 Préparer le pire scénario: {scenario_critique['scenario']}")
        
        duree_estimee = "24-48h"
        if scenario_critique["niveau_risque"] == "Critique":
            duree_estimee = "7-14 jours"
        elif scenario_critique["niveau_risque"] == "Élevé":
            duree_estimee = "3-7 jours"
        elif scenario_critique["niveau_risque"] == "Moyen":
            duree_estimee = "48-72h"
        
        return {
            "scenario_principal": scenario_critique["scenario"],
            "niveau_risque_global": scenario_critique["niveau_risque"],
            "victimes_max_estimees": scenario_critique["impact_estime"]["victimes"],
            "gravite_globale": scenario_critique["impact_estime"]["gravite"],
            "ressources_totales_requises": ressources_combinees,
            "deficits_detectes": deficit if deficit else {},
            "actions_globales": actions_globales,
            "duree_estimee_totale": duree_estimee,
            "statut_preparation": "🟢 PRÊT" if not deficit else "🟠 RENFORTS NÉCESSAIRES"
        }

    # =========================================================================
    # NOUVELLES FONCTIONNALITÉS IA INTÉGRÉES
    # =========================================================================

    def analyser_temps_reel(self, stream_donnees):
        """Traiter les données dès qu'elles arrivent et mettre à jour les prédictions."""
        if len(stream_donnees) == 0:
            return {"statut": "En attente de flux de données..."}
            
        intensite_recentes = sum(d.get("nb_victimes_reel", 0) for d in stream_donnees)
        gravite_maj = "Stable"
        if intensite_recentes > 50: gravite_maj = "Escalade Critique"
        elif intensite_recentes > 10: gravite_maj = "Aggravation"
        
        return {
            "flux_analyse": len(stream_donnees),
            "tendance_risque": gravite_maj,
            "nouveaux_points_chauds": "Détectés" if intensite_recentes > 20 else "Aucun",
            "mise_a_jour_predictions": "Complète"
        }

    def recommander_consignes(self, type_catastrophe, niveau_gravite):
        """Générer automatiquement des consignes de sécurité adaptées."""
        consignes_base = {
            "Séisme": [
                "Couvrez-vous sous un meuble solide.",
                "Éloignez-vous des fenêtres, miroirs et murs extérieurs.",
                "Si vous êtes dehors, restez dans un endroit dégagé, loin des poteaux et immeubles."
            ],
            "Inondation": [
                "Rejoignez le point le plus haut possible (étage, colline).",
                "Coupez l'électricité, l'eau et le gaz.",
                "Ne marchez ni ne conduisez dans les zones inondées."
            ],
            "Incendie": [
                "Évacuez immédiatement les lieux en restant bas (la fumée monte).",
                "Ne jamais utiliser les ascenseurs.",
                "Fermez les portes derrière vous pour ralentir la propagation."
            ]
        }
        
        consignes_critiques = [
            "⚠️ ALERTE ROUGE : Exécutez le plan d'évacuation d'urgence !",
            "Emportez uniquement votre kit de survie (papiers, eau, médicaments essentiels)."
        ]
        
        reco = consignes_base.get(type_catastrophe, [
            "Restez calme et écoutez les instructions des autorités locales.",
            "Préparez un kit de survie basique."
        ])
        
        if niveau_gravite in ["CRITIQUE", "TRÈS GRAVE", "Critique"]:
            reco = consignes_critiques + reco
            
        return {
            "type_cible": type_catastrophe,
            "consignes": reco
        }

    def predire_propagation(self, type_catastrophe, loc_initiale, zone, vitesse_vent=10, temps_heures=6):
        """Prédiction de propagation pour incendies/inondations."""
        if type_catastrophe not in ["Incendie", "Inondation", "Feu de Forêt Massif"]:
            return {"message": "Propagation non applicable à ce type de catastrophe."}
            
        rayon_km = (vitesse_vent * temps_heures * 0.1) if type_catastrophe == "Incendie" else (temps_heures * 0.5)
        
        return {
            "localisation_initiale": loc_initiale,
            "zones_menacees": [f"Périmètre de +{rayon_km:.1f} km autour de {loc_initiale}"],
            "temps_avant_impact": f"{temps_heures} heures",
            "facteur_accel": "Vent fort" if vitesse_vent > 20 else "Normal"
        }

    # =========================================================================
    # (5.8) ITINÉRAIRES D'ÉVACUATION RÉELS VIA OSRM + NOMINATIM
    # =========================================================================
    def suggerer_evacuation(self, localisation_danger, loc_actuelle="Position Citoyen"):
        """
        Suggestion d'évacuation utilisant OSRM pour les vrais itinéraires (5.8).
        Fallback sur des données heuristiques si l'API est indisponible.
        """
        routes_sures = []
        temps_estime = "30-45 minutes si circulation fluide"
        distance_km = None
        
        # Tenter de géocoder la localisation via Nominatim
        try:
            nominatim_url = f"https://nominatim.openstreetmap.org/search?q={localisation_danger},+Algeria&format=json&limit=1"
            headers = {"User-Agent": "CATASTROPI-Module-IA/1.0"}
            resp = http_requests.get(nominatim_url, headers=headers, timeout=5)
            
            if resp.status_code == 200 and resp.json():
                loc = resp.json()[0]
                lat, lon = float(loc['lat']), float(loc['lon'])
                
                # Définir un point d'évacuation sûr (ex: hôpital le plus proche ou point élevé)
                safe_lat, safe_lon = lat + 0.05, lon + 0.05  # ~5km au NE
                
                # Appel OSRM pour le calcul d'itinéraire réel
                osrm_url = f"http://router.project-osrm.org/route/v1/driving/{lon},{lat};{safe_lon},{safe_lat}?overview=false"
                osrm_resp = http_requests.get(osrm_url, timeout=5)
                
                if osrm_resp.status_code == 200:
                    route_data = osrm_resp.json()
                    if route_data.get('routes'):
                        route = route_data['routes'][0]
                        duration_min = round(route['duration'] / 60, 1)
                        distance_km = round(route['distance'] / 1000, 1)
                        temps_estime = f"{duration_min} minutes ({distance_km} km par la route)"
                        routes_sures = [
                            f"Itinéraire OSRM calculé: {distance_km} km en {duration_min} min",
                            f"Coordonnées départ: {lat:.4f}, {lon:.4f}",
                            f"Coordonnées arrivée: {safe_lat:.4f}, {safe_lon:.4f}",
                            "Suivre les axes principaux, éviter les routes secondaires"
                        ]
        except Exception as e:
            print(f"   ⚠️ OSRM/Nominatim non disponible: {e}")
        
        # Fallback si pas de données OSRM
        if not routes_sures:
            routes_sures = [
                f"Prendre la N1 en direction de l'opposé de {localisation_danger}",
                "Éviter les axes secondaires risquant d'être encombrés",
                "Rejoindre le centre d'hébergement le plus proche (Zone Verte Nord)"
            ]
        
        return {
            "danger_identifie": localisation_danger,
            "loc_actuelle": loc_actuelle,
            "itineraires_recommandes": routes_sures,
            "temps_estime": temps_estime,
            "distance_km": distance_km,
            "source": "OSRM" if distance_km else "Heuristique"
        }

    # =========================================================================
    # (5.7) MÉTÉO EN TEMPS RÉEL VIA OPENWEATHERMAP
    # =========================================================================
    def obtenir_meteo(self, localisation):
        """
        Récupère les données météo en temps réel pour une localisation (5.7).
        Returns: dict avec vent, pluie, température, conditions.
        """
        meteo = {
            "temperature": None,
            "vent_kmh": None,
            "pluie_mm": None,
            "conditions": "Non disponible",
            "humidite": None,
            "source": "indisponible"
        }
        
        if self.OWM_API_KEY == "demo":
            # Mode démo: données simulées réalistes
            import random
            meteo = {
                "temperature": round(random.uniform(5, 42), 1),
                "vent_kmh": round(random.uniform(0, 80), 1),
                "pluie_mm": round(random.uniform(0, 50), 1),
                "conditions": random.choice(["Clair", "Nuageux", "Pluie", "Orage", "Vent fort"]),
                "humidite": random.randint(20, 95),
                "source": "simulation"
            }
            return meteo
        
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={localisation},DZ&appid={self.OWM_API_KEY}&units=metric&lang=fr"
            resp = http_requests.get(url, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                meteo["temperature"] = data["main"]["temp"]
                meteo["vent_kmh"] = round(data["wind"]["speed"] * 3.6, 1)  # m/s -> km/h
                meteo["pluie_mm"] = data.get("rain", {}).get("1h", 0)
                meteo["conditions"] = data["weather"][0]["description"] if data.get("weather") else "N/A"
                meteo["humidite"] = data["main"]["humidity"]
                meteo["source"] = "OpenWeatherMap"
        except Exception as e:
            print(f"   ⚠️ OpenWeatherMap non disponible: {e}")
        
        return meteo

    def scorer_fiabilite_citoyen(self, nb_valides, nb_rejetes):
        """
        =====================================================
        SCORING DE FIABILITÉ CITOYEN
        =====================================================
        Noter la fiabilité d'un citoyen basé sur ses précédents signalements.
        """
        total = nb_valides + nb_rejetes
        if total == 0:
            return {"score": 50, "niveau": "Inconnu (Nouveau)", "historique": total, "taux_fiabilite": 50}
            
        ratio = nb_valides / total
        score_base = ratio * 100
        
        if nb_valides >= 5: score_base = min(100, score_base + 10)
        if nb_rejetes >= 3: score_base = max(0, score_base - 20)
        
        score_final = int(score_base)
        if score_final > 80: n = "Excellente"
        elif score_final > 60: n = "Bonne"
        elif score_final > 40: n = "Moyenne"
        else: n = "Faible"
        
        return {
            "score": score_final,
            "niveau": n,
            "historique": total,
            "taux_fiabilite": score_final
        }

    # =========================================================================
    # FONCTION 7: APPRENTISSAGE EN TEMPS RÉEL
    # =========================================================================
    
    def ajouter_et_apprendre(self, nouvelle_donnee: dict):
        """
        Ajoute une catastrophe validée par un responsable dans le dataset 
        et relance l'entraînement des modèles pour qu'il soit plus précis avec le temps.
        """
        print(f"\n🔄 APPRENTISSAGE EN TEMPS RÉEL: Intégration d'une nouvelle catastrophe ({nouvelle_donnee.get('type')})")
        
        if self.donnees_historiques is None:
            print("⚠️ Impossible d'apprendre: Dataset historique non chargé.")
            return
            
        try:
            nouvelle_entree = pd.DataFrame({
                "id": [len(self.donnees_historiques) + 1],
                "type_catastrophe": [nouvelle_donnee.get("type", "Autre")],
                "zone": [nouvelle_donnee.get("zone", "CENTRE")],
                "mois": [datetime.now().month],
                "population_zone": [nouvelle_donnee.get("population_zone", 100000)],
                "superficie_affectee_km2": [5.0],
                "duree_heures": [24],
                "nb_victimes_reel": [nouvelle_donnee.get("nb_victimes_reel", 0)],
                "nb_blesses_reel": [0],
                "nb_sinistres_reel": [0],
                "degats_materiaux_da": [0],
                "equipes_necessaires": [5],
                "ambulances_necessaires": [2],
                "niveau_risque": [nouvelle_donnee.get("niveau_risque", "Moyen")],
                "alerte_prealable_heures": [0],
                "infrastructures_endommagees": [0],
                "evacuation_necessaire": [0]
            })
            
            self.donnees_historiques = pd.concat([self.donnees_historiques, nouvelle_entree], ignore_index=True)
            
            self.donnees_historiques.to_excel(self.chemin_excel, index=False)
            print(f"✅ Fichier {self.chemin_excel} mis à jour.")
            
            self._entrainer_modele_risque()
            self._entrainer_modele_impact()
            print("🧠 Modèles ré-entrainés avec succès !")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'apprentissage: {e}")


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    module = ModuleIA("historique_catastrophes.xlsx")
    
    # Test rapide
    result = module.detecter_fake_news("Séisme de magnitude 4 ressenti à Boumerdès ce matin, aucun dégât signalé.")
    print(f"\nTest Fake News: {result['statut']} (confiance: {result['confiance']}%)")
    
    result2 = module.detecter_fake_news("URGENT: Tsunami de 100 mètres va détruire toute l'Algérie, le gouvernement cache tout!")
    print(f"Test Fake News2: {result2['statut']} (confiance: {result2['confiance']}%)")
    
    meteo = module.obtenir_meteo("Alger")
    print(f"\nMétéo Alger: {meteo}")
    
    evac = module.suggerer_evacuation("Boumerdès")
    print(f"\nÉvacuation: {evac}")