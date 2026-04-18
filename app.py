"""
Plateforme CATASTROPI - Backend Flask
=====================================
"""

import os
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import random
import json
import csv
import io
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'super-secret-catastropi-key-2026'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

db = SQLAlchemy(app)
CORS(app)
jwt = JWTManager(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
ia_cache = {}

from module_ia import ModuleIA
print("Initialisation du Module IA...")
module_ia = ModuleIA("historique_catastrophes.xlsx")
print("Module IA prêt !")


# ==========================================
# MODÈLES DE DONNÉES
# ==========================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False) # ADMIN, RESPONSABLE, AGENT, CITOYEN
    avatar_url = db.Column(db.String(255), nullable=True) 
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    actif = db.Column(db.Boolean, default=True)
    competence = db.Column(db.String(100), nullable=True)
    localisation_actuelle = db.Column(db.String(255), nullable=True)
    
    # Fiabilité pour le citoyen
    nb_signalements_valides = db.Column(db.Integer, default=0)
    nb_signalements_rejetes = db.Column(db.Integer, default=0)
    fiabilite_score = db.Column(db.Integer, default=50)

    def to_dict(self):
        return {
            'id': self.id,
            'nom': self.nom,
            'email': self.email,
            'role': self.role,
            'actif': self.actif,
            'avatar_url': self.avatar_url,
            'competence': self.competence,
            'localisation_actuelle': self.localisation_actuelle,
            'fiabilite_score': self.fiabilite_score,
            'date_creation': self.date_creation.isoformat() if self.date_creation else None
        }

class Catastrophe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False) 
    zone = db.Column(db.String(50), nullable=False, default="CENTRE")
    localisation = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    date_declaration = db.Column(db.DateTime, default=datetime.utcnow)
    date_publication = db.Column(db.DateTime, nullable=True)
    statut = db.Column(db.String(20), default="SIGNALEE")
    niveau_risque = db.Column(db.String(20), nullable=True) 
    estimation_victimes = db.Column(db.Integer, nullable=True)
    declare_par_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    publie_par_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    nb_victimes = db.Column(db.Integer, default=0)
    score_ia = db.Column(db.Integer, nullable=True)
    commentaire_ia = db.Column(db.Text, nullable=True)
    
    image_url = db.Column(db.String(500), nullable=True)
    plan_ia_json = db.Column(db.Text, nullable=True)
    
    declare_par = db.relationship('User', foreign_keys=[declare_par_id])
    publie_par = db.relationship('User', foreign_keys=[publie_par_id])

    missions = db.relationship('Mission', backref='catastrophe_rel', cascade="all, delete-orphan")
    ressources = db.relationship('Ressource', backref='catastrophe_rel', cascade="all, delete-orphan")
    victimes_associees = db.relationship('Victime', backref='catastrophe_rel', cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'zone': self.zone,
            'localisation': self.localisation,
            'description': self.description,
            'date_declaration': self.date_declaration.isoformat() if self.date_declaration else None,
            'date_publication': self.date_publication.isoformat() if self.date_publication else None,
            'statut': self.statut,
            'niveau_risque': self.niveau_risque,
            'estimation_victimes': self.estimation_victimes,
            'declare_par': self.declare_par.to_dict() if self.declare_par else None,
            'publie_par': self.publie_par.nom if self.publie_par else None,
            'nb_victimes': self.nb_victimes,
            'image_url': self.image_url,
            'score_ia': self.score_ia,
            'commentaire_ia': self.commentaire_ia,
            'plan_ia': json.loads(self.plan_ia_json) if self.plan_ia_json else None,
            'agents_assignes': [{'nom': m.agent.nom, 'id': m.agent.id, 'statut': m.statut} for m in self.missions if m.agent],
            'ressources_assignees': [{'nom': r.nom, 'type': r.type, 'quantite': r.quantite} for r in self.ressources],
            'victimes': [v.to_dict() for v in self.victimes_associees]
        }

class IAAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    signalement_id = db.Column(db.Integer, db.ForeignKey('catastrophe.id'), nullable=False)
    confidence_score = db.Column(db.Integer, nullable=True)
    risk_level = db.Column(db.String(20), nullable=True)
    estimated_victims_min = db.Column(db.Integer, nullable=True)
    estimated_victims_max = db.Column(db.Integer, nullable=True)
    priority_zones = db.Column(db.Text, nullable=True)
    recommended_resources = db.Column(db.Text, nullable=True)
    suggested_agents = db.Column(db.Text, nullable=True)
    instructions = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default="pending")
    motif_rejet = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyzed_at = db.Column(db.DateTime, nullable=True)
    analysis_duration = db.Column(db.Float, nullable=True)
    responsable_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    signalement = db.relationship('Catastrophe', backref=db.backref('analyses_ia', lazy=True))
    responsable = db.relationship('User', foreign_keys=[responsable_id])

    def to_dict(self):
        return {
            'id': self.id,
            'signalement_id': self.signalement_id,
            'type_catastrophe': self.signalement.type if self.signalement else "",
            'localisation': self.signalement.localisation if self.signalement else "",
            'confidence_score': self.confidence_score,
            'risk_level': self.risk_level,
            'estimated_victims_min': self.estimated_victims_min,
            'estimated_victims_max': self.estimated_victims_max,
            'priority_zones': json.loads(self.priority_zones) if self.priority_zones else [],
            'recommended_resources': json.loads(self.recommended_resources) if self.recommended_resources else [],
            'suggested_agents': json.loads(self.suggested_agents) if self.suggested_agents else [],
            'instructions': self.instructions,
            'status': self.status,
            'motif_rejet': self.motif_rejet,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None,
            'analysis_duration': self.analysis_duration,
            'responsable_nom': self.responsable.nom if self.responsable else None
        }

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    message = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)
    catastrophe_id = db.Column(db.Integer, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'message': self.message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'read': self.read,
            'catastrophe_id': self.catastrophe_id
        }

class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    action = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='logs')

    def to_dict(self):
        return {
            'id': self.id,
            'user_email': self.user.email if self.user else "Système",
            'user_role': self.user.role if self.user else "SYS",
            'action': self.action,
            'timestamp': self.timestamp.isoformat()
        }

class Victime(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    etat = db.Column(db.String(20), nullable=False)
    localisation = db.Column(db.String(255), nullable=False)
    catastrophe_id = db.Column(db.Integer, db.ForeignKey('catastrophe.id'), nullable=False)
    date_enregistrement = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'nom': self.nom,
            'age': self.age,
            'etat': self.etat,
            'localisation': self.localisation,
            'catastrophe_id': self.catastrophe_id,
            'date_enregistrement': self.date_enregistrement.isoformat() if self.date_enregistrement else None
        }

class Ressource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False) 
    nom = db.Column(db.String(100), nullable=False)
    quantite = db.Column(db.Integer, default=1)
    catastrophe_id = db.Column(db.Integer, db.ForeignKey('catastrophe.id'), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'nom': self.nom,
            'quantite': self.quantite,
            'catastrophe_id': self.catastrophe_id
        }

class Mission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    catastrophe_id = db.Column(db.Integer, db.ForeignKey('catastrophe.id'), nullable=False)
    statut = db.Column(db.String(50), default="ASSIGNEE")
    instructions = db.Column(db.Text, nullable=True)
    date_assignation = db.Column(db.DateTime, default=datetime.utcnow)
    date_fin = db.Column(db.DateTime, nullable=True)
    
    agent = db.relationship('User')

    def to_dict(self):
        cat = db.session.get(Catastrophe, self.catastrophe_id)
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'agent_nom': self.agent.nom if self.agent else None,
            'catastrophe_id': self.catastrophe_id,
            'catastrophe_titre': cat.type if cat else None,
            'catastrophe_zone': cat.zone if cat else None,
            'localisation': cat.localisation if cat else None,
            'statut': self.statut,
            'instructions': self.instructions,
            'date_assignation': self.date_assignation.isoformat() if self.date_assignation else None,
            'date_fin': self.date_fin.isoformat() if self.date_fin else None
        }

class Besoin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mission_id = db.Column(db.Integer, db.ForeignKey('mission.id'), nullable=False)
    catastrophe_id = db.Column(db.Integer, db.ForeignKey('catastrophe.id'), nullable=False)
    type_besoin = db.Column(db.String(50), nullable=False)
    quantite = db.Column(db.String(50), nullable=False)
    urgence = db.Column(db.String(20), default="Haut")
    resolu = db.Column(db.Boolean, default=False)
    statut = db.Column(db.String(20), default="en_attente")  # en_attente, approuve, refuse
    motif_refus = db.Column(db.Text, nullable=True)
    declare_par_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        agent = db.session.get(User, self.declare_par_id) if self.declare_par_id else None
        return {
            'id': self.id,
            'mission_id': self.mission_id,
            'catastrophe_id': self.catastrophe_id,
            'type_besoin': self.type_besoin,
            'quantite': self.quantite,
            'urgence': self.urgence,
            'resolu': self.resolu,
            'statut': self.statut,
            'motif_refus': self.motif_refus,
            'agent_nom': agent.nom if agent else None,
            'date_creation': self.date_creation.isoformat() if self.date_creation else None
        }

# ==========================================
# UTILITAIRES
# ==========================================

def get_current_user():
    user_id = get_jwt_identity()
    return db.session.get(User, user_id)

def log_action(user_id, action_desc):
    l = Log(user_id=user_id, action=action_desc)
    db.session.add(l)
    db.session.commit()

def admin_required(fn):
    @jwt_required()
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user or user.role != 'ADMIN':
            return jsonify({'message': 'Accès non autorisé. Rôle ADMIN requis.'}), 403
        return fn(*args, **kwargs)
    return wrapper

def responsable_required(fn):
    @jwt_required()
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user or user.role not in ['ADMIN', 'RESPONSABLE']:
            return jsonify({'message': 'Accès non autorisé. Rôle RESPONSABLE ou plus requis.'}), 403
        return fn(*args, **kwargs)
    return wrapper

def agent_required(fn):
    @jwt_required()
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user or user.role not in ['ADMIN', 'RESPONSABLE', 'AGENT']:
            return jsonify({'message': 'Accès non autorisé. Rôle AGENT ou plus requis.'}), 403
        return fn(*args, **kwargs)
    return wrapper

def create_notification(user_id, message, catastrophe_id=None):
    n = Notification(user_id=user_id, message=message, catastrophe_id=catastrophe_id)
    db.session.add(n)
    db.session.commit()
    socketio.emit('notification', {'user_id': user_id, 'message': message, 'catastrophe_id': catastrophe_id})

# ==========================================
# ROUTES API & UI
# ==========================================

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# --- 1. AUTHENTIFICATION ---
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'message': 'Email et mot de passe requis'}), 400
        
    user = User.query.filter_by(email=data['email']).first()
    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({'message': 'Identifiants invalides'}), 401
    
    if not user.actif:
        return jsonify({'message': 'Compte désactivé'}), 403
        
    access_token = create_access_token(identity=str(user.id))
    log_action(user.id, "Connexion de l'utilisateur")
    return jsonify({
        'message': 'Connexion réussie',
        'token': access_token,
        'user': user.to_dict()
    }), 200

# --- 2. ADMIN: UTILISATEURS ET LOGS ---
@app.route('/api/admin/users', methods=['GET', 'POST'])
@admin_required
def manage_users():
    if request.method == 'GET':
        users = User.query.all()
        return jsonify([u.to_dict() for u in users]), 200
    
    if request.method == 'POST':
        data = request.json
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'message': 'Email déjà utilisé'}), 400
            
        u = User(
            nom=data['nom'],
            email=data['email'],
            password=generate_password_hash(data.get('password', 'password123')),
            role=data['role']
        )
        db.session.add(u)
        db.session.commit()
        log_action(get_jwt_identity(), f"Création utilisateur {u.email}")
        return jsonify(u.to_dict()), 201

@app.route('/api/admin/users/<int:user_id>', methods=['PUT', 'DELETE'])
@admin_required
def modify_user(user_id):
    u = db.session.get(User, user_id)
    if not u: return jsonify({'message': "Not found"}), 404
    
    if request.method == 'PUT':
        data = request.json
        if 'nom' in data: u.nom = data['nom']
        if 'role' in data: u.role = data['role']
        if 'actif' in data: u.actif = data['actif']
        db.session.commit()
        log_action(get_jwt_identity(), f"Modification utilisateur {u.email}")
        return jsonify(u.to_dict()), 200
        
    if request.method == 'DELETE':
        db.session.delete(u)
        db.session.commit()
        log_action(get_jwt_identity(), f"Suppression utilisateur {u.email}")
        return jsonify({'message': 'Supprimé'}), 200

@app.route('/api/admin/logs', methods=['GET'])
@admin_required
def get_logs():
    logs = Log.query.order_by(Log.timestamp.desc()).limit(100).all()
    return jsonify([l.to_dict() for l in logs]), 200

@app.route('/api/admin/users/import', methods=['POST'])
@admin_required
def import_users():
    if 'file' not in request.files:
        return jsonify({'message': 'Fichier manquant'}), 400
    file = request.files['file']
    if not file.filename.endswith(('.csv', '.xlsx')):
        return jsonify({'message': 'Format invalide (CSV ou XLSX attendu)'}), 400
    
    try:
        import pandas as pd
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
            
        for _, row in df.iterrows():
            if not User.query.filter_by(email=row['email']).first():
                u = User(nom=row['nom'], email=row['email'], password=generate_password_hash("pass123"), role=row['role'])
                db.session.add(u)
        db.session.commit()
        log_action(get_jwt_identity(), "Importation d'utilisateurs via fichier")
        return jsonify({'message': 'Import réussi'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_admin_stats():
    nb_users = User.query.count()
    nb_cats = Catastrophe.query.count()
    nb_rejetes = Catastrophe.query.filter_by(statut="REJETEE").count()
    nb_publiees = Catastrophe.query.filter_by(statut="PUBLIEE").count()
    
    zones = db.session.query(Catastrophe.zone, db.func.count(Catastrophe.id)).filter_by(statut='PUBLIEE').group_by(Catastrophe.zone).all()
    map_data = {z[0]: z[1] for z in zones}
    
    return jsonify({
        "users": nb_users,
        "catastrophes_totales": nb_cats,
        "rejetes": nb_rejetes,
        "publiees": nb_publiees,
        "victimes": db.session.query(db.func.sum(Catastrophe.nb_victimes)).scalar() or 0,
        "map_data": map_data
    }), 200

# --- 3. CATASTROPHES & DECLARATION CITOYENNE ---
@app.route('/api/catastrophes', methods=['GET'])
def get_all_catastrophes():
    statut = request.args.get('statut')
    query = Catastrophe.query.filter_by(statut=statut) if statut else Catastrophe.query
    cats = query.order_by(Catastrophe.id.desc()).all()
    results = []
    
    for c in cats:
        c_dict = c.to_dict()
        if c.statut == "SIGNALEE" and request.args.get('pre_analyze') == 'true':
            user = db.session.get(User, c.declare_par_id)
            hist = module_ia.scorer_fiabilite_citoyen(user.nb_signalements_valides if user else 0, user.nb_signalements_rejetes if user else 0)
            res_fake = module_ia.detecter_fake_news(c.description, hist, [])
            score_fiab = res_fake.get('score_fiabilite', 50)
            c_dict['ia_pre_analyse'] = {
                'score': score_fiab,
                'recommandation': res_fake['recommandation'],
                'est_fake': res_fake['est_fake']
            }
        results.append(c_dict)
        
    return jsonify(results), 200

@app.route('/api/declarer', methods=['POST'])
@jwt_required()
def declarer_catastrophe():
    data = request.json
    desc = data.get('description', 'Signalement citoyen')
    type_cat = data.get('type', 'Autre')
    zone = data.get('zone', 'CENTRE')
    localisation = data.get('localisation', 'Inconnue')
    
    user = get_current_user()
    
    historique_utilisateur = module_ia.scorer_fiabilite_citoyen(user.nb_signalements_valides, user.nb_signalements_rejetes)
    user.fiabilite_score = historique_utilisateur['score']
    
    signalements_recents = Catastrophe.query.filter_by(zone=zone).filter(
        Catastrophe.date_declaration >= datetime.utcnow() - timedelta(hours=2)
    ).all()
    
    resultat_fake = module_ia.detecter_fake_news(desc, historique_utilisateur, [s.to_dict() for s in signalements_recents])
    
    status = "SIGNALEE"
    score_ia_val = resultat_fake.get('score_fiabilite', 50)
    
    if resultat_fake['est_fake']:
        status = "REJETEE"
        user.nb_signalements_rejetes += 1
        db.session.commit()
        create_notification(user.id, f"Votre signalement '{type_cat}' a été rejeté par l'IA (Suspicion de Fake News).")
        log_action(user.id, "Signalement rejeté par IA")
    else:
        status = "SIGNALEE"
    
    new_cat = Catastrophe(
        type=type_cat,
        zone=zone,
        localisation=localisation,
        description=desc,
        declare_par_id=user.id,
        statut=status,
        score_ia=int(score_ia_val),
        commentaire_ia=resultat_fake.get('recommandation', '')
    )
    db.session.add(new_cat)
    db.session.commit()
    
    socketio.emit('dashboard_update', {'type': 'new_catastrophe', 'catastrophe': new_cat.to_dict()})
    
    if status == "SIGNALEE":
        log_action(user.id, f"Signalement '{type_cat}' créé. En attente validation responsable.")
        resps = User.query.filter_by(role='RESPONSABLE').all()
        for r in resps:
            create_notification(r.id, f"Nouveau signalement ({type_cat}) en attente de vérification.", new_cat.id)

    return jsonify(new_cat.to_dict()), 201

# --- 4. GESTION RESPONSABLE ---
import time

@app.route('/api/analyze_signal/<int:signal_id>', methods=['POST'])
@responsable_required
def analyze_signal(signal_id):
    start_time = time.time()
    catastrophe = db.session.get(Catastrophe, signal_id)
    if not catastrophe: return jsonify({'message': 'Not found'}), 404
    
    heure_actuelle = datetime.utcnow().hour
    
    user = db.session.get(User, catastrophe.declare_par_id)
    historique_utilisateur = module_ia.scorer_fiabilite_citoyen(user.nb_signalements_valides if user else 0, user.nb_signalements_rejetes if user else 0)
    signalements_recents = Catastrophe.query.filter_by(zone=catastrophe.zone).filter(
        Catastrophe.date_declaration >= datetime.utcnow() - timedelta(hours=2)
    ).all()
    resultat_fake = module_ia.detecter_fake_news(catastrophe.description, historique_utilisateur, [s.to_dict() for s in signalements_recents])
    confidence_score = int(resultat_fake.get('score_fiabilite', 50))
    
    pop_zone = 3000000 if catastrophe.zone == "NORD" else 1000000 if catastrophe.zone in ["EST", "OUEST", "CENTRE"] else 500000
    res_risque = module_ia.predire_niveau_risque(
        type_catastrophe=catastrophe.type,
        zone=catastrophe.zone,            
        mois=datetime.utcnow().month,
        population_zone=pop_zone,
        superficie_affectee=5.0,
        heure=heure_actuelle
    )
    niveau_risque = res_risque['niveau_risque']
    
    res_impact = module_ia.estimer_impact(
        type_catastrophe=catastrophe.type,
        zone=catastrophe.zone,
        niveau_risque=niveau_risque,
        population_zone=pop_zone,
        superficie_affectee=5.0,
        duree_estimee_heures=24
    )
    
    # Météo en temps réel (5.7)
    meteo = module_ia.obtenir_meteo(catastrophe.localisation)
    
    # Évacuation réelle (5.8)
    evacuation = module_ia.suggerer_evacuation(catastrophe.localisation)
    
    duration = round(time.time() - start_time, 2)
    
    consignes_ia = module_ia._generer_recommandations_risque(niveau_risque)
    
    # Section 4: Priorisation des interventions
    zones_touches = [
        {"nom": f"Zone Critique {catastrophe.localisation}", "victimes": res_impact['victimes_estimees']['valeur'], "blesses_critiques": max(0, res_impact['blesses_estimes']['valeur'] // 3), "blesses_graves": res_impact['blesses_estimes']['valeur'], "accessible": True, "temps_heures": 1, "population": pop_zone},
        {"nom": "Quartier adjacent Nord", "victimes": max(0, res_impact['victimes_estimees']['valeur'] // 4), "blesses_critiques": 2, "blesses_graves": 5, "accessible": True, "temps_heures": 2, "population": pop_zone // 5},
        {"nom": "Quartier adjacent Sud", "victimes": max(0, res_impact['victimes_estimees']['valeur'] // 6), "blesses_critiques": 0, "blesses_graves": 3, "accessible": False, "temps_heures": 4, "population": pop_zone // 8},
    ]
    priorisation = module_ia.prioriser_interventions(zones_touches, {})
    
    # Section 5: Optimisation des ressources
    ressources_disponibles = {
        "equipes_secours": 20, "ambulances": 15, "lits_hopitaux": 100,
        "vehicules_logistique": 10, "tentes": 50, "kits_nourriture": 500, "kits_eau": 800
    }
    zones_besoins = [
        {"nom": z["zone"], "priorite": z["score_priorite"], **{f"{k}_necessaires": v for k, v in res_impact['ressources_requises'].items()}} 
        for z in priorisation.get('zones_prioritaires', [])
    ]
    allocation = module_ia.optimiser_allocation_ressources(ressources_disponibles, zones_besoins)
    
    analysis_details = {
        "fake_news": {
            "motifs": resultat_fake.get('motifs', []),
            "confiance": confidence_score,
            "score_camembert": resultat_fake.get('score_camembert', 0),
            "score_indicateurs": resultat_fake.get('score_indicateurs', 0),
            "probabilite_ml": resultat_fake.get('probabilite_ml', 0),
            "statut": resultat_fake.get('statut', ''),
            "est_fake": resultat_fake.get('est_fake', False)
        },
        "risque_details": {
            "niveau": niveau_risque,
            "facteurs": res_risque.get('facteurs_principaux', []),
            "probabilites": res_risque.get('probabilites', {}),
            "shap_raisons": res_risque.get('shap_raisons', []),
            "facteur_nuit": res_risque.get('facteur_nuit', False),
            "heure": heure_actuelle,
            "recommandations": res_risque.get('recommandations', []),
            "analyse_historique": res_risque.get('analyse_historique', [])
        },
        "impact": {
            "victimes_estimees": res_impact['victimes_estimees'],
            "blesses_estimes": res_impact['blesses_estimes'],
            "sinistres_estimes": res_impact['sinistres_estimes'],
            "degats_materiaux_estimes": res_impact['degats_materiaux_estimes'],
            "gravite_globale": res_impact['gravite_globale'],
            "ressources_requises": res_impact['ressources_requises'],
            "impact_evacuation": res_impact.get('impact_evacuation', '')
        },
        "priorisation": {
            "zones_prioritaires": [{
                "zone": z['zone'],
                "score_priorite": z['score_priorite'],
                "niveau_urgence": z['niveau_urgence'],
                "justifications": z.get('justifications', [])
            } for z in priorisation.get('zones_prioritaires', [])],
            "ordre_intervention": priorisation.get('ordre_intervention', []),
            "alertes_speciales": priorisation.get('alertes_speciales', []),
            "recommandation_globale": priorisation.get('recommandation_globale', '')
        },
        "allocation": {
            "allocation_par_zone": allocation.get('allocation_par_zone', []),
            "taux_couverture": allocation.get('taux_couverture', 0),
            "alertes_deficit": allocation.get('alertes_deficit', []),
            "recommandation": allocation.get('recommandation', '')
        },
        "consignes": consignes_ia,
        "meteo": meteo,
        "evacuation": evacuation,
        "version_modele": "CATASTROPI IA v2.0 (TF-IDF + RF + OSRM)",
        "duree_analyse": duration
    }
    
    resp_user = get_current_user()
    
    # Build priority zones list from priorisation results
    pz_list = [z['zone'] for z in priorisation.get('zones_prioritaires', [])]
    if not pz_list:
        pz_list = [f"Zone Critique {catastrophe.localisation}", "Quartier adjacent A", "Quartier adjacent B"]
    
    analysis = IAAnalysis(
        signalement_id=signal_id,
        confidence_score=confidence_score,
        risk_level=niveau_risque,
        estimated_victims_min=max(0, res_impact['victimes_estimees']['valeur'] - 50),
        estimated_victims_max=res_impact['victimes_estimees']['valeur'] + 150,
        priority_zones=json.dumps(pz_list),
        recommended_resources=json.dumps(res_impact['ressources_requises']),
        suggested_agents=json.dumps(["Équipe Alpha", "Équipe Beta"]),
        instructions=json.dumps(analysis_details),
        status="pending",
        analyzed_at=datetime.utcnow(),
        analysis_duration=duration,
        responsable_id=resp_user.id if resp_user else None
    )
    db.session.add(analysis)
    db.session.commit()
    
    # Cache the result
    ia_cache[signal_id] = analysis.to_dict()
    
    return jsonify(analysis.to_dict()), 200

@app.route('/api/accept_analysis/<int:signal_id>', methods=['POST'])
@responsable_required
def accept_analysis(signal_id):
    analysis = IAAnalysis.query.filter_by(signalement_id=signal_id).order_by(IAAnalysis.id.desc()).first()
    if not analysis: return jsonify({'message': 'Analysis not found'}), 404
    catastrophe = db.session.get(Catastrophe, signal_id)
    
    resp_user = get_current_user()
    analysis.status = "accepted"
    analysis.responsable_id = resp_user.id if resp_user else None
    catastrophe.statut = "Validé par IA"
    
    plan = {
        "niveau_risque": analysis.risk_level,
        "victimes_estimees": analysis.estimated_victims_max,
        "ressources_requises": json.loads(analysis.recommended_resources)
    }
    catastrophe.niveau_risque = analysis.risk_level
    catastrophe.estimation_victimes = analysis.estimated_victims_max
    catastrophe.plan_ia_json = json.dumps(plan)
    
    citoyen = db.session.get(User, catastrophe.declare_par_id)
    if citoyen:
        citoyen.nb_signalements_valides += 1
        citoyen.fiabilite_score = module_ia.scorer_fiabilite_citoyen(citoyen.nb_signalements_valides, citoyen.nb_signalements_rejetes)['score']
    
    db.session.commit()
    
    # Apprentissage continu IA (5.3)
    module_ia.renforcer_apprentissage_fake_news(catastrophe.description, est_fake=False)
    
    create_notification(catastrophe.declare_par_id, f"Votre signalement '{catastrophe.type}' a été validé par l'IA.", catastrophe.id)
    log_action(get_jwt_identity(), f"A accepté l'analyse IA du signalement #{signal_id}")
    socketio.emit('dashboard_update', {'type': 'analysis_accepted', 'id': signal_id})
    return jsonify({'message': 'Accepted'}), 200

@app.route('/api/reject_analysis/<int:signal_id>', methods=['POST'])
@responsable_required
def reject_analysis(signal_id):
    analysis = IAAnalysis.query.filter_by(signalement_id=signal_id).order_by(IAAnalysis.id.desc()).first()
    if not analysis: return jsonify({'message': 'Analysis not found'}), 404
    catastrophe = db.session.get(Catastrophe, signal_id)
    
    motif = request.json.get('motif', 'Aucun motif fourni')
    resp_user = get_current_user()
    analysis.status = "rejected"
    analysis.motif_rejet = motif
    analysis.responsable_id = resp_user.id if resp_user else None
    catastrophe.statut = "REJETEE"
    
    citoyen = db.session.get(User, catastrophe.declare_par_id)
    if citoyen:
        citoyen.nb_signalements_rejetes += 1
        citoyen.fiabilite_score = module_ia.scorer_fiabilite_citoyen(citoyen.nb_signalements_valides, citoyen.nb_signalements_rejetes)['score']
        
    db.session.commit()
    
    # Apprentissage continu IA (5.3)
    module_ia.renforcer_apprentissage_fake_news(catastrophe.description, est_fake=True)
    
    create_notification(catastrophe.declare_par_id, f"Votre signalement a été rejeté. Motif : {motif}")
    log_action(get_jwt_identity(), f"A rejeté l'analyse IA du signalement #{signal_id}")
    socketio.emit('dashboard_update', {'type': 'analysis_rejected', 'id': signal_id})
    return jsonify({'message': 'Rejected'}), 200

@app.route('/api/reject_signal/<int:signal_id>', methods=['POST'])
@responsable_required
def reject_signal(signal_id):
    """Rejet direct d'un signalement depuis le tableau de validation."""
    catastrophe = db.session.get(Catastrophe, signal_id)
    if not catastrophe: return jsonify({'message': 'Not found'}), 404
    
    motif = request.json.get('motif', 'Rejet manuel par le responsable')
    resp_user = get_current_user()
    catastrophe.statut = "REJETEE"
    
    citoyen = db.session.get(User, catastrophe.declare_par_id)
    if citoyen:
        citoyen.nb_signalements_rejetes += 1
        citoyen.fiabilite_score = module_ia.scorer_fiabilite_citoyen(citoyen.nb_signalements_valides, citoyen.nb_signalements_rejetes)['score']
    
    db.session.commit()
    module_ia.renforcer_apprentissage_fake_news(catastrophe.description, est_fake=True)
    create_notification(catastrophe.declare_par_id, f"Votre signalement '{catastrophe.type}' a été rejeté. Motif : {motif}")
    log_action(get_jwt_identity(), f"Rejet direct du signalement #{signal_id}: {motif}")
    socketio.emit('dashboard_update', {'type': 'signal_rejected', 'id': signal_id})
    return jsonify({'message': 'Signalement rejeté', 'motif': motif}), 200

@app.route('/api/pre_analyze_batch', methods=['POST'])
@responsable_required
def pre_analyze_batch():
    """Analyse IA automatique d'un lot de signalements (appelé au chargement de la page)."""
    data = request.json
    signal_ids = data.get('signal_ids', [])
    results = {}
    
    for sid in signal_ids:
        # Check cache first
        if sid in ia_cache:
            results[str(sid)] = ia_cache[sid]
            continue
        
        # Check if already analyzed in DB
        existing = IAAnalysis.query.filter_by(signalement_id=sid).order_by(IAAnalysis.id.desc()).first()
        if existing:
            ia_cache[sid] = existing.to_dict()
            results[str(sid)] = existing.to_dict()
            continue
        
        # Run fresh analysis
        catastrophe = db.session.get(Catastrophe, sid)
        if not catastrophe:
            continue
        
        start_time = time.time()
        heure_actuelle = datetime.utcnow().hour
        
        user_decl = db.session.get(User, catastrophe.declare_par_id)
        hist_user = module_ia.scorer_fiabilite_citoyen(
            user_decl.nb_signalements_valides if user_decl else 0,
            user_decl.nb_signalements_rejetes if user_decl else 0
        )
        signalements_recents = Catastrophe.query.filter_by(zone=catastrophe.zone).filter(
            Catastrophe.date_declaration >= datetime.utcnow() - timedelta(hours=2)
        ).all()
        resultat_fake = module_ia.detecter_fake_news(catastrophe.description, hist_user, [s.to_dict() for s in signalements_recents])
        confidence_score = int(resultat_fake.get('score_fiabilite', 50))
        
        pop_zone = 3000000 if catastrophe.zone == "NORD" else 1000000 if catastrophe.zone in ["EST", "OUEST", "CENTRE"] else 500000
        res_risque = module_ia.predire_niveau_risque(
            type_catastrophe=catastrophe.type, zone=catastrophe.zone,
            mois=datetime.utcnow().month, population_zone=pop_zone,
            superficie_affectee=5.0, heure=heure_actuelle
        )
        
        res_impact = module_ia.estimer_impact(
            type_catastrophe=catastrophe.type, zone=catastrophe.zone,
            niveau_risque=res_risque['niveau_risque'], population_zone=pop_zone,
            superficie_affectee=5.0, duree_estimee_heures=24
        )
        
        meteo = module_ia.obtenir_meteo(catastrophe.localisation)
        evacuation = module_ia.suggerer_evacuation(catastrophe.localisation)
        consignes_ia = module_ia._generer_recommandations_risque(res_risque['niveau_risque'])
        duration = round(time.time() - start_time, 2)
        
        # Priorisation
        zones_touches = [
            {"nom": f"Zone Critique {catastrophe.localisation}", "victimes": res_impact['victimes_estimees']['valeur'], "blesses_critiques": max(0, res_impact['blesses_estimes']['valeur'] // 3), "blesses_graves": res_impact['blesses_estimes']['valeur'], "accessible": True, "temps_heures": 1, "population": pop_zone},
            {"nom": "Quartier adjacent Nord", "victimes": max(0, res_impact['victimes_estimees']['valeur'] // 4), "blesses_critiques": 2, "blesses_graves": 5, "accessible": True, "temps_heures": 2, "population": pop_zone // 5},
        ]
        priorisation = module_ia.prioriser_interventions(zones_touches, {})
        
        # Allocation
        ressources_dispo = {"equipes_secours": 20, "ambulances": 15, "lits_hopitaux": 100, "vehicules_logistique": 10, "tentes": 50, "kits_nourriture": 500, "kits_eau": 800}
        zones_besoins = [{"nom": z["zone"], "priorite": z["score_priorite"], **{f"{k}_necessaires": v for k, v in res_impact['ressources_requises'].items()}} for z in priorisation.get('zones_prioritaires', [])]
        allocation = module_ia.optimiser_allocation_ressources(ressources_dispo, zones_besoins)
        
        analysis_details = {
            "fake_news": {"motifs": resultat_fake.get('motifs', []), "confiance": confidence_score, "score_camembert": resultat_fake.get('score_camembert', 0), "score_indicateurs": resultat_fake.get('score_indicateurs', 0), "probabilite_ml": resultat_fake.get('probabilite_ml', 0), "statut": resultat_fake.get('statut', ''), "est_fake": resultat_fake.get('est_fake', False)},
            "risque_details": {"niveau": res_risque['niveau_risque'], "facteurs": res_risque.get('facteurs_principaux', []), "probabilites": res_risque.get('probabilites', {}), "shap_raisons": res_risque.get('shap_raisons', []), "facteur_nuit": res_risque.get('facteur_nuit', False), "heure": heure_actuelle, "recommandations": res_risque.get('recommandations', []), "analyse_historique": res_risque.get('analyse_historique', [])},
            "impact": {"victimes_estimees": res_impact['victimes_estimees'], "blesses_estimes": res_impact['blesses_estimes'], "sinistres_estimes": res_impact['sinistres_estimes'], "degats_materiaux_estimes": res_impact['degats_materiaux_estimes'], "gravite_globale": res_impact['gravite_globale'], "ressources_requises": res_impact['ressources_requises'], "impact_evacuation": res_impact.get('impact_evacuation', '')},
            "priorisation": {"zones_prioritaires": [{"zone": z['zone'], "score_priorite": z['score_priorite'], "niveau_urgence": z['niveau_urgence'], "justifications": z.get('justifications', [])} for z in priorisation.get('zones_prioritaires', [])], "ordre_intervention": priorisation.get('ordre_intervention', []), "alertes_speciales": priorisation.get('alertes_speciales', []), "recommandation_globale": priorisation.get('recommandation_globale', '')},
            "allocation": {"allocation_par_zone": allocation.get('allocation_par_zone', []), "taux_couverture": allocation.get('taux_couverture', 0), "alertes_deficit": allocation.get('alertes_deficit', []), "recommandation": allocation.get('recommandation', '')},
            "consignes": consignes_ia, "meteo": meteo, "evacuation": evacuation,
            "version_modele": "CATASTROPI IA v2.0 (TF-IDF + RF + OSRM)", "duree_analyse": duration
        }
        
        resp_user = get_current_user()
        pz_list = [z['zone'] for z in priorisation.get('zones_prioritaires', [])]
        
        analysis = IAAnalysis(
            signalement_id=sid, confidence_score=confidence_score, risk_level=res_risque['niveau_risque'],
            estimated_victims_min=max(0, res_impact['victimes_estimees']['valeur'] - 50),
            estimated_victims_max=res_impact['victimes_estimees']['valeur'] + 150,
            priority_zones=json.dumps(pz_list), recommended_resources=json.dumps(res_impact['ressources_requises']),
            suggested_agents=json.dumps(["Équipe Alpha", "Équipe Beta"]),
            instructions=json.dumps(analysis_details), status="pending",
            analyzed_at=datetime.utcnow(), analysis_duration=duration,
            responsable_id=resp_user.id if resp_user else None
        )
        db.session.add(analysis)
        db.session.commit()
        
        ia_cache[sid] = analysis.to_dict()
        results[str(sid)] = analysis.to_dict()
    
    return jsonify(results), 200

@app.route('/api/ia_analysis', methods=['GET'])
@responsable_required
def get_ia_analyses():
    analyses = IAAnalysis.query.order_by(IAAnalysis.id.desc()).all()
    return jsonify([a.to_dict() for a in analyses]), 200

@app.route('/api/get_analyses', methods=['GET'])
@responsable_required
def get_analyses_filtered():
    q = IAAnalysis.query
    status_filter = request.args.get('status')
    if status_filter and status_filter != 'all':
        q = q.filter_by(status=status_filter)
    analyses = q.order_by(IAAnalysis.id.desc()).all()
    return jsonify([a.to_dict() for a in analyses]), 200

@app.route('/api/catastrophes/<int:catastrophe_id>/publier', methods=['POST'])
@responsable_required
def publier_catastrophe(catastrophe_id):
    catastrophe = db.session.get(Catastrophe, catastrophe_id)
    catastrophe.statut = "PUBLIEE"
    catastrophe.date_publication = datetime.utcnow()
    catastrophe.publie_par_id = get_jwt_identity()
    
    citoyen = db.session.get(User, catastrophe.declare_par_id)
    if citoyen:
        citoyen.nb_signalements_valides += 1
        citoyen.fiabilite_score = module_ia.scorer_fiabilite_citoyen(citoyen.nb_signalements_valides, citoyen.nb_signalements_rejetes)['score']
        
    db.session.commit()
    log_action(get_jwt_identity(), f"A publié la catastrophe #{catastrophe_id}")
    
    citoyens = User.query.filter_by(role='CITOYEN').all()
    for c in citoyens:
        create_notification(c.id, f"ALERTE : Catastrophe '{catastrophe.type}' déclarée proche de {catastrophe.localisation}. Suivez les consignes de sécurité.", catastrophe.id)
    
    socketio.emit('nouvelle_alerte', {'message': f"ALERTE : Catastrophe '{catastrophe.type}'", 'catastrophe': catastrophe.to_dict()})
    
    return jsonify(catastrophe.to_dict()), 200

@app.route('/api/catastrophes/<int:catastrophe_id>', methods=['PUT', 'GET'])
@jwt_required()
def modifier_catastrophe(catastrophe_id):
    c = db.session.get(Catastrophe, catastrophe_id)
    if not c: return jsonify({'message': 'Not found'}), 404
    if request.method == 'GET':
        return jsonify(c.to_dict()), 200

    data = request.json
    if 'type' in data: c.type = data['type']
    if 'zone' in data: c.zone = data['zone']
    if 'description' in data: c.description = data['description']
    if 'statut' in data: c.statut = data['statut']
    db.session.commit()
    log_action(get_jwt_identity(), f"Modification de la catastrophe #{catastrophe_id}")
    return jsonify(c.to_dict()), 200

@app.route('/api/agents_disponibles', methods=['GET'])
@responsable_required
def get_agents_disponibles():
    # Return ALL agents with availability info
    agents = User.query.filter_by(role='AGENT', actif=True).all()
    sous_requete_missions_actives = db.session.query(Mission.agent_id).filter(Mission.statut != 'TERMINEE').subquery()
    
    result = []
    for a in agents:
        missions_actives = Mission.query.filter(Mission.agent_id == a.id, Mission.statut != 'TERMINEE').count()
        result.append({
            'id': a.id,
            'nom': a.nom,
            'competence': a.competence or 'Secouriste polyvalent',
            'localisation_actuelle': a.localisation_actuelle or 'Base centrale',
            'missions_actives': missions_actives,
            'disponible': missions_actives == 0
        })
    return jsonify(result), 200

@app.route('/api/catastrophes/<int:catastrophe_id>/agents', methods=['POST'])
@responsable_required
def assigner_agent(catastrophe_id):
    data = request.json
    agent_ids = data.get('agent_ids', [])
    if not agent_ids and data.get('agent_id'):
        agent_ids = [data.get('agent_id')]
        
    m_list = []
    for agent_id in agent_ids:
        m = Mission(
            agent_id=agent_id,
            catastrophe_id=catastrophe_id,
            instructions=data.get('instructions', "Rendez-vous sur les lieux pour évaluation initiale.")
        )
        db.session.add(m)
        m_list.append(m)
        
    db.session.commit()
    log_action(get_jwt_identity(), f"A assigné les agents {agent_ids} à la catastrophe #{catastrophe_id}")
    for agent_id in agent_ids:
        create_notification(agent_id, f"Nouvelle mission assignée. Catastrophe #{catastrophe_id}: {data.get('instructions', 'Évaluation initiale')}", catastrophe_id)
    
    return jsonify([m.to_dict() for m in m_list]), 201

@app.route('/api/catastrophes/<int:catastrophe_id>/ressources', methods=['POST'])
@responsable_required
def assigner_ressource(catastrophe_id):
    data = request.json
    r = Ressource(
        type=data['type'],
        nom=data['nom'],
        quantite=data.get('quantite', 1),
        catastrophe_id=catastrophe_id
    )
    db.session.add(r)
    db.session.commit()
    log_action(get_jwt_identity(), f"Ressource {r.nom} assignée à {catastrophe_id}")
    return jsonify(r.to_dict()), 201

# --- 5. RAPPORTS EXPORT ---
@app.route('/api/reports/catastrophe/<int:catastrophe_id>', methods=['GET'])
@jwt_required()
def report_catastrophe(catastrophe_id):
    try:
        from reportlab.pdfgen import canvas
        c = db.session.get(Catastrophe, catastrophe_id)
        if not c: return "Not Found", 404
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 750, f"Rapport de Catastrophe #{c.id}: {c.type}")
        p.setFont("Helvetica", 12)
        p.drawString(100, 720, f"Statut: {c.statut}")
        p.drawString(100, 700, f"Zone: {c.zone} - {c.localisation}")
        p.drawString(100, 680, f"Date Déclaration: {c.date_declaration}")
        p.drawString(100, 660, f"Niveau de Risque: {c.niveau_risque}")
        p.drawString(100, 640, f"Victimes: {c.nb_victimes} (estimé: {c.estimation_victimes})")
        
        p.drawString(100, 600, "Description:")
        p.setFont("Helvetica", 10)
        p.drawString(100, 580, c.description[:80] + "...")
        
        p.showPage()
        p.save()
        buffer.seek(0)
        
        log_action(get_jwt_identity(), f"Edition du rapport PDF catastrophe #{catastrophe_id}")
        return send_file(buffer, as_attachment=True, download_name=f'rapport_catastrophe_{c.id}.pdf', mimetype='application/pdf')
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Export victimes CSV
@app.route('/api/catastrophes/<int:catastrophe_id>/victimes/export/csv', methods=['GET'])
@jwt_required()
def export_victimes_csv(catastrophe_id):
    victimes = Victime.query.filter_by(catastrophe_id=catastrophe_id).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Nom', 'Age', 'Etat', 'Localisation', 'Date'])
    for v in victimes:
        writer.writerow([v.id, v.nom, v.age, v.etat, v.localisation, v.date_enregistrement])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": f"attachment;filename=victimes_catastrophe_{catastrophe_id}.csv"}
    )

# Export victimes PDF
@app.route('/api/catastrophes/<int:catastrophe_id>/victimes/export/pdf', methods=['GET'])
@jwt_required()
def export_victimes_pdf(catastrophe_id):
    victimes = Victime.query.filter_by(catastrophe_id=catastrophe_id).all()
    cat = db.session.get(Catastrophe, catastrophe_id)
    
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, f"Liste des Victimes - {cat.type if cat else ''} #{catastrophe_id}")
    p.setFont("Helvetica", 10)
    y = 720
    for v in victimes:
        p.drawString(100, y, f"#{v.id} - {v.nom} ({v.age} ans) - {v.etat} - {v.localisation}")
        y -= 18
        if y < 50:
            p.showPage()
            y = 750
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f'victimes_{catastrophe_id}.pdf', mimetype='application/pdf')

# --- 6. AGENT DE SECOURS ---

@app.route('/api/agent/missions', methods=['GET'])
@agent_required
def get_agent_missions():
    user_id = get_jwt_identity()
    missions = Mission.query.filter_by(agent_id=user_id).order_by(Mission.id.desc()).all()
    results = [m.to_dict() for m in missions]
    for r in results:
        cat = db.session.get(Catastrophe, r['catastrophe_id'])
        r['ai_conseils'] = module_ia.recommander_consignes(cat.type, cat.niveau_risque) if cat else {}
    return jsonify(results), 200

@app.route('/api/agent/missions/<int:mission_id>/status', methods=['PUT'])
@agent_required
def update_mission_status(mission_id):
    m = db.session.get(Mission, mission_id)
    if str(m.agent_id) != str(get_jwt_identity()): return jsonify({"message": "Non autorisé"}), 403
    new_statut = request.json.get('statut')
    m.statut = new_statut
    if new_statut == 'TERMINEE':
        m.date_fin = datetime.utcnow()
    db.session.commit()
    log_action(get_jwt_identity(), f"Mission #{mission_id} statuts mis à jour: {m.statut}")
    return jsonify(m.to_dict()), 200

@app.route('/api/agent/victimes', methods=['POST'])
@agent_required
def enregistrer_victime():
    data = request.json
    v = Victime(
        nom=data.get('nom', 'Inconnu'),
        age=data.get('age'),
        etat=data['etat'],
        localisation=data['localisation'],
        catastrophe_id=data['catastrophe_id']
    )
    db.session.add(v)
    
    c = db.session.get(Catastrophe, data['catastrophe_id'])
    if c:
        c.nb_victimes += 1
        
    db.session.commit()
    
    socketio.emit('update_victimes', {'catastrophe_id': data['catastrophe_id'], 'nb_victimes': c.nb_victimes if c else 0, 'victime': v.to_dict()})
    log_action(get_jwt_identity(), f"Enregistrement victime pour catastrophe {data['catastrophe_id']}")
    return jsonify(v.to_dict()), 201

@app.route('/api/catastrophes/<int:catastrophe_id>/victimes', methods=['GET'])
@jwt_required()
def get_victimes_catastrophe(catastrophe_id):
    victimes = Victime.query.filter_by(catastrophe_id=catastrophe_id).order_by(Victime.id.desc()).all()
    return jsonify([v.to_dict() for v in victimes]), 200

# Update victime état
@app.route('/api/victimes/<int:victime_id>', methods=['PUT'])
@agent_required
def update_victime(victime_id):
    v = db.session.get(Victime, victime_id)
    if not v: return jsonify({'message': 'Not found'}), 404
    data = request.json
    if 'etat' in data: v.etat = data['etat']
    if 'nom' in data: v.nom = data['nom']
    if 'age' in data: v.age = data['age']
    if 'localisation' in data: v.localisation = data['localisation']
    db.session.commit()
    return jsonify(v.to_dict()), 200

@app.route('/api/agent/besoins', methods=['POST'])
@agent_required
def declarer_besoin():
    data = request.json
    b = Besoin(
        mission_id=data['mission_id'],
        catastrophe_id=data['catastrophe_id'],
        type_besoin=data['type_besoin'],
        quantite=str(data.get('quantite', '1')),
        urgence=data.get('urgence', 'Haut'),
        declare_par_id=get_jwt_identity()
    )
    db.session.add(b)
    db.session.commit()
    
    log_action(get_jwt_identity(), f"A déclaré un besoin urgent: {b.type_besoin}")
    resps = User.query.filter_by(role='RESPONSABLE').all()
    for r in resps:
        create_notification(r.id, f"Nouveau Besoin URGENT ({b.type_besoin}) demandé pour mission #{b.mission_id}", b.catastrophe_id)
        
    return jsonify(b.to_dict()), 201

# List besoins for responsable
@app.route('/api/besoins', methods=['GET'])
@responsable_required
def get_all_besoins():
    besoins = Besoin.query.filter_by(statut='en_attente').order_by(Besoin.id.desc()).all()
    return jsonify([b.to_dict() for b in besoins]), 200

@app.route('/api/besoins/<int:besoin_id>/approuver', methods=['POST'])
@responsable_required
def approuver_besoin(besoin_id):
    b = db.session.get(Besoin, besoin_id)
    if not b: return jsonify({'message': 'Not found'}), 404
    b.resolu = True
    b.statut = 'approuve'
    db.session.commit()
    create_notification(b.declare_par_id, f"✅ Votre demande de besoin ({b.type_besoin}) a été APPROUVÉE. Ressources en route.", b.catastrophe_id)
    return jsonify({"message": "Approuvé"}), 200
    
@app.route('/api/besoins/<int:besoin_id>/refuser', methods=['POST'])
@responsable_required
def refuser_besoin(besoin_id):
    b = db.session.get(Besoin, besoin_id)
    if not b: return jsonify({'message': 'Not found'}), 404
    motif = request.json.get('motif', 'Ressources non disponibles')
    b.statut = 'refuse'
    b.motif_refus = motif
    db.session.commit()
    create_notification(b.declare_par_id, f"❌ Votre demande de besoin ({b.type_besoin}) a été REFUSÉE: {motif}.", b.catastrophe_id)
    return jsonify({"message": "Refusé"}), 200

# --- CHATBOT ---
@app.route('/api/chatbot', methods=['POST'])
@jwt_required()
def chatbot_query():
    query = request.json.get('query', '').lower().strip()
    
    # Wilayas algériennes pour matching
    wilayas = [
        "adrar","chlef","laghouat","oum el bouaghi","batna","béjaïa","biskra","béchar",
        "blida","bouira","tamanrasset","tébessa","tlemcen","tiaret","tizi ouzou","alger",
        "djelfa","jijel","sétif","saïda","skikda","sidi bel abbès","annaba","guelma",
        "constantine","médéa","mostaganem","m'sila","mascara","ouargla","oran","el bayadh",
        "illizi","bordj bou arréridj","boumerdès","el tarf","tindouf","tissemsilt",
        "el oued","khenchela","souk ahras","tipaza","mila","aïn defla","naâma",
        "aïn témouchent","ghardaïa","relizane","el m'ghair","el meniaa","ouled djellal",
        "bordj badji mokhtar","béni abbès","timimoun","touggourt","djanet","in salah","in guezzam"
    ]
    
    # Types de catastrophes
    types_cat = ["séisme","tremblement","incendie","feu","inondation","crue",
                 "tempête","accident","explosion","glissement","tsunami","canicule","neige"]
    
    # 1. Consignes de sécurité
    if 'consigne' in query or 'instruction' in query or 'securite' in query or 'sécurité' in query:
        consignes_map = {
            'séisme': "📝 **Consignes Séisme**:\n1. Mettez-vous sous un meuble solide.\n2. Éloignez-vous des fenêtres et miroirs.\n3. Ne sortez qu'après la fin des secousses.\n4. En extérieur, éloignez-vous des bâtiments.\n5. Après les secousses, coupez gaz et électricité.",
            'tremblement': "📝 **Consignes Séisme**:\n1. Mettez-vous sous un meuble solide.\n2. Éloignez-vous des fenêtres et miroirs.\n3. Ne sortez qu'après la fin des secousses.",
            'incendie': "📝 **Consignes Incendie**:\n1. Évacuez immédiatement par les escaliers.\n2. Protégez votre visage avec un linge mouillé.\n3. Baissez-vous s'il y a de la fumée.\n4. Ne jamais utiliser les ascenseurs.\n5. Fermez les portes derrière vous.",
            'feu': "📝 **Consignes Incendie**:\n1. Évacuez immédiatement.\n2. Appelez le 14 (Protection Civile).\n3. Ne combattez le feu que s'il est petit et maîtrisable.",
            'inondation': "📝 **Consignes Inondation**:\n1. Montez aux étages supérieurs.\n2. Coupez électricité et gaz.\n3. Ne marchez/conduisez pas dans l'eau en mouvement.\n4. Préparez un kit d'urgence (eau, nourriture, médicaments).",
            'crue': "📝 **Consignes Crue**:\n1. Éloignez-vous des oueds et rivières.\n2. Rejoignez les points hauts.\n3. N'essayez pas de traverser une crue.",
            'tempête': "📝 **Consignes Tempête**:\n1. Restez à l'intérieur.\n2. Fermez volets et fenêtres.\n3. Éloignez-vous des arbres et lignes électriques.\n4. Rangez les objets susceptibles d'être emportés.",
            'canicule': "📝 **Consignes Canicule**:\n1. Buvez beaucoup d'eau.\n2. Restez au frais.\n3. Évitez les sorties entre 11h et 16h.\n4. Surveillez les personnes vulnérables."
        }
        for key, msg in consignes_map.items():
            if key in query:
                return jsonify({"response": msg})
        return jsonify({"response": "📝 **Consignes Générales**:\n1. Restez calme.\n2. Écoutez les autorités.\n3. Appelez le 14 (Protection Civile) ou le 15 (SAMU).\n4. Préparez un kit d'urgence."})
    
    # 2. Traitement / Premiers secours
    if 'traitement' in query or 'premiers secours' in query or 'soigner' in query:
        traitements = {
            'brûlure': "🆘 **Premiers Secours - Brûlure**:\n1. Refroidir sous l'eau tiède 15 min.\n2. Ne JAMAIS percer les cloques.\n3. Couvrir avec un linge propre.\n4. Appeler les urgences si grave.",
            'fracture': "🆘 **Premiers Secours - Fracture**:\n1. Immobiliser le membre.\n2. Ne PAS tenter de replacer l'os.\n3. Appliquer de la glace autour.\n4. Appeler les secours.",
            'hémorragie': "🆘 **Premiers Secours - Hémorragie**:\n1. Appuyer fermement avec un linge propre.\n2. Surélever le membre si possible.\n3. Ne pas retirer le linge.\n4. Appeler les urgences.",
            'étouffement': "🆘 **Premiers Secours - Étouffement**:\n1. 5 tapes dans le dos.\n2. Si inefficace: manoeuvre de Heimlich.\n3. Appeler le 15 si la victime perd connaissance.",
            'noyade': "🆘 **Premiers Secours - Noyade**:\n1. Sortir la victime de l'eau.\n2. Vérifier la respiration.\n3. Position latérale de sécurité.\n4. Pratiquer le bouche-à-bouche si nécessaire."
        }
        for key, msg in traitements.items():
            if key in query:
                return jsonify({"response": msg})
        return jsonify({"response": "🆘 **Premiers Secours Généraux**:\n1. Vérifier la conscience.\n2. Appeler le 15 (SAMU).\n3. Position latérale de sécurité si inconscient.\n4. Ne déplacez pas un blessé grave."})
    
    # 3. Recherche victime par nom
    if 'victime' in query:
        nom_cherche = query.replace('victime', '').strip()
        if nom_cherche:
            v_match = Victime.query.filter(Victime.nom.ilike(f'%{nom_cherche}%')).first()
            if v_match:
                cat = db.session.get(Catastrophe, v_match.catastrophe_id)
                cat_name = cat.type if cat else "Inconnue"
                return jsonify({"response": f"👥 **Victime trouvée**:\n- **Nom**: {v_match.nom}\n- **Âge**: {v_match.age}\n- **État**: {v_match.etat}\n- **Catastrophe**: {cat_name}\n- **Localisation**: {v_match.localisation}"})
            return jsonify({"response": "Je ne trouve aucune victime correspondant à ce nom."})
    
    # 4. Recherche wilaya + catastrophe
    wilaya_trouvee = None
    for w in wilayas:
        if w in query:
            wilaya_trouvee = w
            break
    
    if wilaya_trouvee and ('catastrophe' in query or 'alerte' in query or any(t in query for t in types_cat)):
        cats = Catastrophe.query.filter(
            Catastrophe.localisation.ilike(f'%{wilaya_trouvee}%') | Catastrophe.zone.ilike(f'%{wilaya_trouvee}%')
        ).filter(Catastrophe.statut.in_(['PUBLIEE', 'Validé par IA'])).order_by(Catastrophe.id.desc()).limit(5).all()
        
        if cats:
            response = f"📍 **Catastrophes actives à {wilaya_trouvee.capitalize()}**:\n"
            for c in cats:
                response += f"\n• **{c.type}** ({c.localisation}) - Risque: {c.niveau_risque} - Victimes: {c.nb_victimes}"
            return jsonify({"response": response})
        return jsonify({"response": f"Aucune catastrophe active trouvée pour {wilaya_trouvee.capitalize()}."})
    
    # 5. Recherche par type de catastrophe
    cat_match = Catastrophe.query.filter(
        (Catastrophe.type.ilike(f'%{query}%')) | (Catastrophe.localisation.ilike(f'%{query}%'))
    ).order_by(Catastrophe.id.desc()).first()
    
    if cat_match:
        return jsonify({"response": f"📍 **Dernière Catastrophe correspondante**:\n- **Type**: {cat_match.type} à {cat_match.localisation}\n- **Statut**: {cat_match.statut}\n- **Risque**: {cat_match.niveau_risque}\n- **Victimes**: {cat_match.nb_victimes} (estimation: {cat_match.estimation_victimes})"})
    
    # 6. Aide générale
    if 'aide' in query or 'help' in query or 'bonjour' in query:
        return jsonify({"response": "🤖 **Assistant IA CATASTROPI**\n\nJe peux vous aider avec:\n• **Consignes de sécurité**: tapez 'consignes séisme'\n• **Premiers secours**: tapez 'traitement brûlure'\n• **État d'une victime**: tapez 'victime ahmed'\n• **Catastrophes par zone**: tapez 'alger catastrophe'\n• **Infos catastrophe**: tapez le type (ex: 'incendie')"})
        
    return jsonify({"response": "Je n'ai pas compris votre requête. Essayez:\n• 'consignes séisme' pour les instructions\n• 'victime [nom]' pour chercher une victime\n• '[wilaya] catastrophe' pour les alertes d'une zone\n• 'traitement brûlure' pour les premiers secours\n• 'aide' pour la liste complète"})

@app.route('/api/notifications', methods=['GET'])
@jwt_required()
def get_notifications():
    user_id = get_jwt_identity()
    notifs = Notification.query.filter((Notification.user_id == user_id) | (Notification.user_id == None)).order_by(Notification.timestamp.desc()).limit(20).all()
    return jsonify([n.to_dict() for n in notifs]), 200

@app.route('/api/notifications/read', methods=['POST'])
@jwt_required()
def mark_notifications_read():
    user_id = get_jwt_identity()
    Notification.query.filter_by(user_id=user_id, read=False).update({"read": True})
    db.session.commit()
    return jsonify({"message": "OK"}), 200

# --- 8. ROUTES CITOYEN ---
@app.route('/api/citoyen/signalements', methods=['GET'])
@jwt_required()
def get_citoyen_signalements():
    user_id = get_jwt_identity()
    cats = Catastrophe.query.filter_by(declare_par_id=user_id).order_by(Catastrophe.date_declaration.desc()).all()
    return jsonify([c.to_dict() for c in cats]), 200

@app.route('/api/citoyen/score', methods=['GET'])
@jwt_required()
def get_citoyen_score():
    u = get_current_user()
    return jsonify({'fiabilite_score': u.fiabilite_score, 'valides': u.nb_signalements_valides, 'rejetes': u.nb_signalements_rejetes})


# ==========================================
# SEEDING DE LA BASE DE DONNÉES
# ==========================================

def seed_db():
    if User.query.count() > 0:
        return
    
    print("Initialisation base de données + Zones...")
    
    admin = User(nom="Admin Master", email="admin@catastropi.dz", password=generate_password_hash("admin123"), role="ADMIN")
    resp = User(nom="Responsable Zone", email="responsable@catastropi.dz", password=generate_password_hash("resp123"), role="RESPONSABLE")
    agent = User(nom="Agent Terrain", email="agent@catastropi.dz", password=generate_password_hash("agent123"), role="AGENT", competence="Secourisme avancé", localisation_actuelle="Alger Centre")
    agent2 = User(nom="Agent Karim", email="agent2@catastropi.dz", password=generate_password_hash("agent123"), role="AGENT", competence="Pompier spécialisé", localisation_actuelle="Boumerdès")
    agent3 = User(nom="Agent Sarah", email="agent3@catastropi.dz", password=generate_password_hash("agent123"), role="AGENT", competence="Médecin urgentiste", localisation_actuelle="Blida")
    citoyen = User(nom="Citoyen Actif", email="citoyen@catastropi.dz", password=generate_password_hash("citoyen123"), role="CITOYEN")
    
    db.session.add_all([admin, resp, agent, agent2, agent3, citoyen])
    db.session.commit()

    log_action(admin.id, "Initialisation système et premier seed de DB")

    urgences = [
        ("Séisme", "NORD", "Alger Centre", "CRITIQUE", "Effondrement signalé sur plusieurs bâtiments résidentiels au niveau de la rue Didouche.", 145),
        ("Inondation", "NORD", "Boumerdès", "MOYEN", "Routes principales submergées suite à l'orage exceptionnel sur la région.", 5),
        ("Tempête Côtière", "NORD", "Béjaïa Manche", "MOYEN", "Vents côtiers très puissants. Risque de vagues submersives.", 2),
        ("Effondrement Pont", "EST", "Constantine", "CRITIQUE", "Rupture de voie sur un pont périphérique secondaire.", 10),
        ("Incendie Forêt", "EST", "Annaba (Edough)", "ELEVE", "Départ de feu important sur le massif de l'Edough.", 0),
        ("Inondation Urbaine", "OUEST", "Oran", "MOYEN", "Bouchage massif des canalisations suite aux forts orages.", 8),
        ("Incendie", "OUEST", "Tlemcen", "ELEVE", "Incendie dans le parc naturel.", 0),
        ("Tempête de sable", "SUD", "Ouargla", "ELEVE", "Visibilité nulle sur les axes routiers principaux. Plusieurs carambolages.", 3),
        ("Alerte Canicule", "SUD", "Ghardaïa", "CRITIQUE", "Température dépassant 50 degrés, pannes électriques réseau signalées.", 20),
        ("Feu de Forêt Massif", "CENTRE", "Chréa, Blida", "CRITIQUE", "Les flammes menacent les chalets du parc naturel national.", 0),
    ]
    
    for i, urg in enumerate(urgences):
        cnf = Catastrophe(
            type=urg[0], zone=urg[1], localisation=urg[2], 
            description=urg[4], statut="PUBLIEE", niveau_risque=urg[3], 
            image_url="", declare_par_id=citoyen.id, publie_par_id=resp.id, 
            estimation_victimes=urg[5], nb_victimes=urg[5],
            date_publication=datetime.utcnow() - timedelta(minutes=random.randint(5, 600))
        )
        db.session.add(cnf)
        
    db.session.commit()
    
    last_cat = Catastrophe.query.first()
    if last_cat:
        m = Mission(agent_id=agent.id, catastrophe_id=last_cat.id, statut="EN_ROUTE", instructions="Evaluer urgence incendie.")
        db.session.add(m)
        db.session.commit()
    
    print("Données de démonstration chargées avec succès !")

# ==========================================
# SIMULATEUR IOT (Tâche en arrière-plan)
# ==========================================
def simulateur_iot_background():
    while True:
        socketio.sleep(30)
        with app.app_context():
            cat = Catastrophe.query.filter_by(statut="PUBLIEE").first()
            if cat:
                event_msgs = [
                    f"⚠️ [Capteur IOT] Aggravation détectée zone {cat.localisation}. Risque d'effondrement.",
                    f"🔴 [Drone IA] Propagation thermique rapide identifiée près de {cat.localisation}.",
                    f"🌊 [Sonde Niveau] Alerte: Montée critique du niveau de l'eau (+15cm).",
                    f"🌬️ [Station Météo] Vitesse du vent > 90km/h. Évacuation recommandée."
                ]
                msg = random.choice(event_msgs)
                print(f"--- IOT SIMULATOR: {msg}")
                socketio.emit('notification', {'user_id': None, 'message': msg, 'catastrophe_id': cat.id})
                socketio.emit('dashboard_update', {'type': 'iot_alert', 'catastrophe_id': cat.id, 'message': msg})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_db()
    
    socketio.start_background_task(simulateur_iot_background)
    socketio.run(app, debug=True, port=5000, use_reloader=False, allow_unsafe_werkzeug=True)
