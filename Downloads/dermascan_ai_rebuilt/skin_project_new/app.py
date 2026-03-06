from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import numpy as np
from PIL import Image
import os
from datetime import datetime, timedelta
import json
from werkzeug.utils import secure_filename
from functools import wraps
import random
import string

app = Flask(__name__)
app.secret_key = 'skincare-ai-ultra-secret-2025'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Try to load TensorFlow model
try:
    import tensorflow as tf
    model = tf.keras.models.load_model("model/skin_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Demo mode active: {e}")

# ─────────────────────────────────────────────
# DOCTOR DATABASE (with login credentials & disease specializations)
# ─────────────────────────────────────────────
# Doctor profile images (served from static/doctors/)
DOCTOR_IMAGES = {
    1: 'static/doctors/doctor_1.png',
    2: 'static/doctors/doctor_2.png',
    3: 'static/doctors/doctor_3.png',
    4: 'static/doctors/doctor_4.png',
    5: 'static/doctors/doctor_5.png',
}

DOCTORS = [
    {
        "id": 1,
        "name": "Dr. Priya Sharma",
        "degree": "MBBS, MD (Dermatology), FAAD",
        "specialization": "Dermatologist & Skin Allergy Specialist",
        "type": "allopathic",
        "experience": "15 Years",
        "email": "dr.priya.sharma@skincare.in",
        "password": "Priya@2024",
        "phone": "+91-9876543210",
        "hospital": "Apollo Skin & Hair Clinic",
        "address": "2nd Floor, Apollo Hospital Complex, Banjara Hills, Hyderabad – 500034",
        "timings": "Mon–Sat: 10:00 AM – 6:00 PM",
        "languages": "English, Hindi, Telugu",
        "rating": 4.9,
        "reviews": 324,
        "diseases": ["Acne", "Eczema", "Melanoma", "Psoriasis", "Ringworm", "Vitiligo", "Normal"]
    },
    {
        "id": 2,
        "name": "Dr. Rajesh Kumar",
        "degree": "BAMS, MD (Ayurveda)",
        "specialization": "Ayurvedic Skin & Lifestyle Physician",
        "type": "ayurvedic",
        "experience": "12 Years",
        "email": "dr.rajesh.kumar@ayurskin.in",
        "password": "Rajesh@2024",
        "phone": "+91-9823456710",
        "hospital": "Ayur Wellness Center",
        "address": "Shop No. 45, Green Park Colony, Saket, New Delhi – 110017",
        "timings": "Mon–Sun: 9:00 AM – 5:00 PM",
        "languages": "English, Hindi, Punjabi",
        "rating": 4.8,
        "reviews": 218,
        "diseases": ["Acne", "Eczema", "Psoriasis", "Vitiligo", "Ringworm", "Normal"]
    },
    {
        "id": 3,
        "name": "Dr. Meena Patel",
        "degree": "BHMS, MD (Homeopathy)",
        "specialization": "Homeopathic Skin & Allergy Specialist",
        "type": "homeopathic",
        "experience": "10 Years",
        "email": "dr.meena.patel@homeocare.in",
        "password": "Meena@2024",
        "phone": "+91-9812345678",
        "hospital": "Dr. Patel's Homeo Clinic",
        "address": "Plot 7, Navrangpura, Near Gujarat High Court, Ahmedabad – 380009",
        "timings": "Tue–Sun: 9:30 AM – 7:00 PM",
        "languages": "English, Gujarati, Hindi",
        "rating": 4.7,
        "reviews": 189,
        "diseases": ["Acne", "Eczema", "Psoriasis", "Vitiligo", "Ringworm", "Normal"]
    },
    {
        "id": 4,
        "name": "Dr. Anand Menon",
        "degree": "MBBS, DNB (Dermatology & Venereology)",
        "specialization": "Pediatric & Adult Dermatologist",
        "type": "allopathic",
        "experience": "18 Years",
        "email": "dr.anand.menon@dermacure.in",
        "password": "Anand@2024",
        "phone": "+91-9900112233",
        "hospital": "DermaCure Skin Institute",
        "address": "3rd Floor, Prestige Tower, MG Road, Bangalore – 560001",
        "timings": "Mon–Fri: 11:00 AM – 7:00 PM, Sat: 10:00 AM – 2:00 PM",
        "languages": "English, Kannada, Malayalam",
        "rating": 4.9,
        "reviews": 412,
        "diseases": ["Acne", "Eczema", "Melanoma", "Psoriasis", "Ringworm", "Vitiligo", "Normal"]
    },
    {
        "id": 5,
        "name": "Dr. Sunita Joshi",
        "degree": "BAMS, PhD (Panchakarma)",
        "specialization": "Ayurvedic Panchakarma & Skin Detox",
        "type": "ayurvedic",
        "experience": "20 Years",
        "email": "dr.sunita.joshi@panchakarma.in",
        "password": "Sunita@2024",
        "phone": "+91-9765432100",
        "hospital": "Vaidyaratnam Ayurveda Hospital",
        "address": "Opp. Bus Stand, Thrissur, Kerala – 680001",
        "timings": "Mon–Sat: 8:00 AM – 3:00 PM",
        "languages": "English, Malayalam, Hindi",
        "rating": 4.8,
        "reviews": 276,
        "diseases": ["Acne", "Eczema", "Psoriasis", "Vitiligo", "Ringworm", "Normal"]
    }
]

# Inject image path and whatsapp into each doctor dict
for _d in DOCTORS:
    _d.setdefault('image', DOCTOR_IMAGES.get(_d['id'], ''))
    _d.setdefault('avatar', '')  # keep for backward compat but unused
    _d.setdefault('whatsapp', _d.get('phone', ''))

# Doctor email → password map for quick lookup
DOCTOR_CREDENTIALS = {d["email"]: d["password"] for d in DOCTORS}

# ─────────────────────────────────────────────
# DISEASE INFO (no hardcoded medicines – fetched via Claude API in frontend)
# ─────────────────────────────────────────────
DISEASE_INFO = {
    "Acne": {
        "name": "Acne Vulgaris (Pimples)",
        "description": "Acne occurs when hair follicles become clogged with oil and dead skin cells, leading to whiteheads, blackheads, and pimples.",
        "severity": "Low to Moderate",
        "severity_class": "warning",
        "recovery_time": "4–8 weeks with treatment",
        "causes": ["Excess sebum production", "Bacterial infection (C. acnes)", "Hormonal changes", "Unhealthy diet", "Stress & poor sleep"],
        "symptoms": ["Whiteheads & blackheads", "Inflamed red pimples", "Pustules with pus", "Oily skin", "Possible scarring"],
        "treatment_steps": [
            {"phase": "Week 1–2", "step": "Cleanse twice daily with gentle face wash. Start medication gently.", "tip": "Do NOT pick pimples"},
            {"phase": "Week 3–4", "step": "Add non-comedogenic moisturizer if skin feels dry.", "tip": "Change pillow cover every 2 days"},
            {"phase": "Week 5–6", "step": "Visible improvement expected. Reduce if irritation occurs.", "tip": "Drink 3 litres of water daily"},
            {"phase": "Week 7–8", "step": "Maintenance phase to prevent relapse.", "tip": "Eat less sugar and dairy"}
        ],
        "prevention_tips": ["Wash face twice daily", "Never sleep with makeup", "Use oil-free skincare", "Manage stress", "Change pillow covers regularly"],
        "diet_advice": "Avoid sugar, dairy, fried food. Eat vegetables, fruits, green tea, zinc-rich foods.",
        "icon": "fa-circle-nodes",
        "color": "#F59E0B"
    },
    "Eczema": {
        "name": "Atopic Dermatitis (Eczema)",
        "description": "Eczema is a chronic inflammatory skin condition characterized by dry, itchy, and inflamed skin linked to genetics, immune dysfunction, and environmental triggers.",
        "severity": "Moderate",
        "severity_class": "warning",
        "recovery_time": "2–6 weeks per flare-up",
        "causes": ["Genetic predisposition", "Overactive immune response", "Dry skin", "Allergens", "Irritants (soap, detergent)", "Stress"],
        "symptoms": ["Dry scaly patches", "Intense itching (worse at night)", "Red or brownish skin", "Thickened cracked skin", "Oozing bumps"],
        "treatment_steps": [
            {"phase": "Day 1–3", "step": "Apply moisturizer every 4 hours. Keep clean and cool.", "tip": "Use cotton clothing only"},
            {"phase": "Day 4–7", "step": "Start cream. Take antihistamine at night.", "tip": "Avoid all identified triggers"},
            {"phase": "Week 2–4", "step": "Use fragrance-free soap and detergent.", "tip": "Bathe in lukewarm water"},
            {"phase": "Week 5–6", "step": "Taper medicine. Continue moisturizing daily.", "tip": "Manage stress"}
        ],
        "prevention_tips": ["Use fragrance-free moisturizer after bath", "Wear soft cotton clothes", "Avoid hot water baths", "Keep nails short", "Use a humidifier"],
        "diet_advice": "Avoid eggs, cow's milk, soy, wheat, peanuts. Eat turmeric, fish, probiotics, leafy greens.",
        "icon": "fa-droplet",
        "color": "#3B82F6"
    },
    "Melanoma": {
        "name": "Melanoma (Skin Cancer – URGENT)",
        "description": "Melanoma is the most dangerous form of skin cancer. Early detection is life-saving. Seek immediate medical attention.",
        "severity": "CRITICAL",
        "severity_class": "danger",
        "recovery_time": "Depends on stage – consult oncologist immediately",
        "causes": ["Excessive UV radiation", "Fair skin", "Family history", "Multiple moles", "Weakened immunity", "Previous sunburns"],
        "symptoms": ["Changing mole size/color/shape", "Asymmetric spot", "Irregular ragged border", "Multiple colors in one mole", "Bleeding or oozing mole"],
        "treatment_steps": [
            {"phase": "IMMEDIATE", "step": "Book appointment with dermatologist or oncologist TODAY.", "tip": "Early action saves life"},
            {"phase": "Day 1–7", "step": "Avoid all sun exposure. Document daily with photos.", "tip": "Biopsy needed for diagnosis"},
            {"phase": "After Diagnosis", "step": "Follow oncologist's plan: surgery, immunotherapy.", "tip": "Join patient support group"},
            {"phase": "Long-term", "step": "Monthly skin self-exams. Annual checkups. Lifelong SPF.", "tip": "Never use tanning beds"}
        ],
        "prevention_tips": ["SEE A DOCTOR TODAY", "Apply SPF 50+ daily", "Never use tanning beds", "Wear UV-protective clothing", "Annual skin cancer screenings"],
        "diet_advice": "Eat antioxidant-rich foods – berries, green tea, broccoli, tomatoes, omega-3. Avoid alcohol.",
        "icon": "fa-triangle-exclamation",
        "color": "#EF4444"
    },
    "Normal": {
        "name": "Healthy Skin (No Disease Detected)",
        "description": "Your skin appears healthy. No significant skin condition detected. Continue your current skincare routine.",
        "severity": "None",
        "severity_class": "success",
        "recovery_time": "N/A – Maintain healthy routine",
        "causes": ["No skin disease detected"],
        "symptoms": ["Clear even skin tone", "No visible lesions", "Normal skin texture"],
        "treatment_steps": [
            {"phase": "Morning", "step": "Face wash → Toner → Moisturizer → SPF 50", "tip": "Never skip sunscreen"},
            {"phase": "Evening", "step": "Makeup removal → Cleanser → Serum → Night cream", "tip": "Sleep on clean pillow"},
            {"phase": "Weekly", "step": "Gentle exfoliation to remove dead skin cells", "tip": "Don't over-exfoliate"},
            {"phase": "Monthly", "step": "Self-check for new moles using ABCDE method", "tip": "Photograph and track changes"}
        ],
        "prevention_tips": ["Drink 3 litres of water daily", "Sleep 7–9 hours nightly", "Eat Vitamin C, E, Zinc rich foods", "Daily SPF", "Avoid smoking"],
        "diet_advice": "Eat amla, berries, papaya, almonds, salmon, spinach. Drink green tea, coconut water.",
        "icon": "fa-circle-check",
        "color": "#10B981"
    },
    "Psoriasis": {
        "name": "Psoriasis (Chronic Autoimmune)",
        "description": "Psoriasis causes rapid buildup of skin cells resulting in silvery-white scaling on red, inflamed skin. Chronic condition requiring ongoing management.",
        "severity": "Moderate to High",
        "severity_class": "danger",
        "recovery_time": "Chronic – 4–12 weeks per flare-up",
        "causes": ["Overactive immune system", "Genetic predisposition", "Stress triggers", "Infections", "Alcohol and smoking"],
        "symptoms": ["Red patches with thick silvery scales", "Dry cracked bleeding skin", "Itching burning soreness", "Thickened pitted nails", "Joint swelling (psoriatic arthritis)"],
        "treatment_steps": [
            {"phase": "Week 1–2", "step": "Begin topical treatment. Take brief lukewarm baths. Moisturize immediately.", "tip": "Hot water WORSENS psoriasis"},
            {"phase": "Week 3–4", "step": "Add medicated shampoo if scalp affected.", "tip": "Avoid scratching plaques"},
            {"phase": "Week 5–8", "step": "Reduce steroid gradually if improving.", "tip": "Moderate morning sunlight helps"},
            {"phase": "Ongoing", "step": "Manage triggers: reduce stress, quit alcohol.", "tip": "Track flare-up patterns"}
        ],
        "prevention_tips": ["Moisturize with thick creams after every bath", "Avoid stress", "Quit smoking and alcohol", "Get morning sunlight", "Avoid skin injuries"],
        "diet_advice": "Avoid alcohol, red meat, dairy, gluten. Eat fatty fish, turmeric, ginger, broccoli, berries.",
        "icon": "fa-shield-virus",
        "color": "#8B5CF6"
    },
    "Ringworm": {
        "name": "Tinea (Ringworm / Fungal Infection)",
        "description": "Ringworm is a highly contagious fungal skin infection appearing as circular ring-shaped red scaly patches. Common in warm humid climates.",
        "severity": "Low to Moderate",
        "severity_class": "warning",
        "recovery_time": "2–4 weeks with antifungal treatment",
        "causes": ["Dermatophyte fungi", "Direct skin-to-skin contact", "Sharing towels or clothing", "Contact with infected animals", "Warm moist environment"],
        "symptoms": ["Ring-shaped red scaly patches", "Itching and burning", "Clear center with raised border", "Expanding ring", "Hair loss in scalp ringworm"],
        "treatment_steps": [
            {"phase": "Day 1–3", "step": "Start antifungal cream. Keep area clean and dry.", "tip": "Do NOT cover with tight bandage"},
            {"phase": "Day 4–14", "step": "Continue even if symptoms improve. Don't stop early.", "tip": "Wash clothes in hot water"},
            {"phase": "Week 3–4", "step": "Continue full course.", "tip": "Avoid sharing towels and combs"},
            {"phase": "After Healing", "step": "Use antifungal powder in skin folds as prevention.", "tip": "Keep skin dry"}
        ],
        "prevention_tips": ["Keep skin clean and dry", "Never share towels", "Wear loose cotton clothing", "Change underwear daily", "Treat pets if infected"],
        "diet_advice": "Avoid sugar and refined carbs. Eat probiotic foods, garlic, coconut oil, turmeric, ginger.",
        "icon": "fa-bacterium",
        "color": "#F97316"
    },
    "Vitiligo": {
        "name": "Vitiligo (White Patches / Shwitra)",
        "description": "Vitiligo causes pale white patches due to destruction of melanocytes. Non-contagious and non-infectious. Chronic autoimmune condition.",
        "severity": "Moderate (Cosmetic + Autoimmune)",
        "severity_class": "warning",
        "recovery_time": "Chronic – Repigmentation takes months to years",
        "causes": ["Autoimmune melanocyte destruction", "Genetic predisposition", "Oxidative stress", "Thyroid disorders"],
        "symptoms": ["Flat white patches on skin", "Premature hair whitening", "Color loss inside mouth", "Sunburn sensitivity in patches"],
        "treatment_steps": [
            {"phase": "Month 1–3", "step": "Start topical treatment. 15 min morning sun exposure daily.", "tip": "Be patient – results take months"},
            {"phase": "Month 3–6", "step": "Continue. Add vitamin B12, D, Copper supplements.", "tip": "Stress worsens vitiligo"},
            {"phase": "Month 6–12", "step": "NBUVB Phototherapy if prescribed.", "tip": "Track patch size monthly"},
            {"phase": "Ongoing", "step": "Screen for thyroid and autoimmune conditions annually.", "tip": "Use SPF 50+ on patches"}
        ],
        "prevention_tips": ["Apply SPF 50+ on white patches", "Manage stress actively", "Check thyroid every 6 months", "Eat copper-rich foods", "Avoid tattoos over patches"],
        "diet_advice": "Eat copper-rich foods (sesame, dark chocolate), Vitamin B12, green leafy vegetables. Avoid excess citrus.",
        "icon": "fa-layer-group",
        "color": "#6366F1"
    }
}

# ─────────────────────────────────────────────
# DISEASE → DOCTOR MAPPING (filter by treatment type + specialization)
# ─────────────────────────────────────────────
def get_doctors_for_disease(disease, medicine_type='allopathic'):
    matching = []
    for doc in DOCTORS:
        if disease in doc.get("diseases", []):
            # For allopathic filter, prefer allopathic doctors first
            if medicine_type == 'allopathic' and doc['type'] == 'allopathic':
                matching.insert(0, doc)
            elif medicine_type == 'ayurvedic' and doc['type'] == 'ayurvedic':
                matching.insert(0, doc)
            elif medicine_type == 'homeopathic' and doc['type'] == 'homeopathic':
                matching.insert(0, doc)
            else:
                matching.append(doc)
    # Ensure at least 2 doctors
    if len(matching) < 2:
        for doc in DOCTORS:
            if doc not in matching and disease in doc.get("diseases", []):
                matching.append(doc)
    return matching[:3]  # max 3 allocated doctors

# ─────────────────────────────────────────────
# IN-MEMORY DATABASE
# ─────────────────────────────────────────────
users_db = {}  # email → {password, role, name, phone, created_at}
scans_db = []
appointments_db = []  # stores all booked appointments
feedbacks_db    = []  # stores patient → doctor ratings
reset_tokens = {}  # token → email
google_users = {}  # For Google OAuth simulation

def init_demo_users():
    """Initialize demo users"""
    users_db["patient@demo.com"] = {
        "password": "demo123", "role": "client", "name": "Demo Patient",
        "phone": "+91-9999000000", "created_at": "2025-01-01"
    }
    # Add doctors to users_db for login
    for doc in DOCTORS:
        users_db[doc["email"]] = {
            "password": doc["password"], "role": "doctor", "name": doc["name"],
            "phone": doc["phone"], "created_at": "2024-01-01",
            "doctor_id": doc["id"]
        }

init_demo_users()

# ─────────────────────────────────────────────
# AUTH DECORATORS
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def doctor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('user_role') != 'doctor':
            flash('Doctor access required.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# IMAGE PREDICTION
# ─────────────────────────────────────────────
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_disease(filepath):
    class_names = list(DISEASE_INFO.keys())
    if model:
        try:
            img = Image.open(filepath).convert('RGB')
            prediction = model.predict(preprocess_image(img))
            idx = np.argmax(prediction)
            return class_names[min(idx, len(class_names)-1)], float(prediction[0][idx])*100
        except Exception as e:
            print(f"Model error: {e}")
    demo = ["Acne", "Eczema", "Psoriasis", "Ringworm", "Vitiligo", "Normal"]
    if not hasattr(predict_disease, '_idx'):
        predict_disease._idx = 0
    disease = demo[predict_disease._idx % len(demo)]
    predict_disease._idx += 1
    return disease, round(random.uniform(78.0, 96.0), 1)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_email'):
        return redirect(url_for('client_dashboard') if session.get('user_role') == 'client' else url_for('doctor_dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if email in users_db and users_db[email]['password'] == password:
            u = users_db[email]
            session['user_email'] = email
            session['user_role'] = u['role']
            session['user_name'] = u['name']
            if u['role'] == 'doctor':
                session['doctor_id'] = u.get('doctor_id')
            flash(f"Welcome back, {u['name'].split()[0]}!", 'success')
            return redirect(url_for('doctor_dashboard') if u['role'] == 'doctor' else url_for('client_dashboard'))
        flash('Invalid email or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/google-login')
def google_login():
    # Simulated Google OAuth – in production use Flask-Dance or Authlib
    flash('Google OAuth requires configuration. Use email/password for demo.', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        name = request.form.get('name', '').strip()
        phone = request.form.get('phone', '').strip()
        if email in users_db:
            flash('Email already registered. Please login.', 'warning')
            return redirect(url_for('login'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('register.html')
        users_db[email] = {
            'password': password, 'role': 'client', 'name': name,
            'phone': phone, 'created_at': datetime.now().strftime('%Y-%m-%d')
        }
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        if email in users_db:
            token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
            reset_tokens[token] = {'email': email, 'expires': datetime.now() + timedelta(hours=1)}
            flash(f'Password reset link sent! (Demo token: {token[:8]}...)', 'success')
        else:
            flash('If that email exists, a reset link has been sent.', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    token_data = reset_tokens.get(token)
    if not token_data or datetime.now() > token_data['expires']:
        flash('Invalid or expired reset link.', 'danger')
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        new_password = request.form.get('password', '')
        if len(new_password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('reset_password.html', token=token)
        email = token_data['email']
        users_db[email]['password'] = new_password
        del reset_tokens[token]
        flash('Password updated successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', token=token)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/client/dashboard')
@login_required
def client_dashboard():
    user_scans = [s for s in scans_db if s.get('user_email') == session['user_email']]
    user_appts = [a for a in appointments_db if a.get('user_email') == session['user_email']]
    # Which doctor_ids the patient has already rated
    rated_doctor_ids = {f['doctor_id'] for f in feedbacks_db if f['user_email'] == session['user_email']}
    return render_template('client_dashboard.html',
                           scans=user_scans,
                           appointments=user_appts,
                           rated_doctor_ids=rated_doctor_ids)

@app.route('/doctor/dashboard')
@login_required
@doctor_required
def doctor_dashboard():
    doctor_id = session.get('doctor_id')
    # Show only scans allocated to this doctor
    my_scans = [s for s in scans_db if doctor_id in s.get('allocated_doctor_ids', [])]
    disease_counts = {}
    for scan in my_scans:
        d = scan.get('disease', 'Unknown')
        disease_counts[d] = disease_counts.get(d, 0) + 1
    # Get current doctor info
    current_doctor = next((d for d in DOCTORS if d['id'] == doctor_id), None)
    # Get appointments booked with this doctor
    my_appointments = [a for a in appointments_db if a.get('doctor_id') == doctor_id]
    # Count new (unread) appointments — ones booked in last 24 hrs
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=24)
    new_appts = [
        a for a in my_appointments
        if datetime.strptime(a['booked_at'], '%Y-%m-%d %H:%M:%S') > cutoff
    ]
    # Feedback from patients for this doctor
    feedbacks = [f for f in feedbacks_db if f.get('doctor_id') == doctor_id]
    if feedbacks:
        avg_rating = round(sum(f['rating'] for f in feedbacks) / len(feedbacks), 1)
    else:
        avg_rating = current_doctor['rating'] if current_doctor else 0
    return render_template('doctor_dashboard.html',
                           scans=my_scans,
                           total_scans=len(my_scans),
                           disease_counts=disease_counts,
                           current_doctor=current_doctor,
                           users_db=users_db,
                           appointments=my_appointments,
                           new_appts_count=len(new_appts),
                           feedbacks=feedbacks,
                           avg_rating=avg_rating)

@app.route('/health-form', methods=['GET', 'POST'])
@login_required
def health_form():
    if request.method == 'POST':
        # Validate all required MCQ fields
        required = ['gender', 'age_group', 'skin_type', 'problem_duration',
                    'skin_location', 'itching', 'spreading', 'previous_treatment', 'medicine_type']
        missing = [f for f in required if not request.form.get(f)]
        if missing:
            flash('Please answer all required questions before proceeding.', 'danger')
            return render_template('health_form.html')

        health_data = {
            'gender': request.form.get('gender'),
            'age_group': request.form.get('age_group'),
            'skin_type': request.form.get('skin_type'),
            'problem_duration': request.form.get('problem_duration'),
            'skin_location': request.form.get('skin_location'),
            'itching': request.form.get('itching'),
            'burning': request.form.get('burning'),
            'spreading': request.form.get('spreading'),
            'diabetes': request.form.get('diabetes'),
            'thyroid': request.form.get('thyroid'),
            'allergies': request.form.get('allergies'),
            'pregnant': request.form.get('pregnant'),
            'previous_treatment': request.form.get('previous_treatment'),
            'treatment_used': request.form.get('treatment_used', ''),
            'medicine_type': request.form.get('medicine_type'),
            'skin_description': request.form.get('skin_description', '')
        }
        session['health_data'] = health_data
        session['health_form_completed'] = True
        return redirect(url_for('scan'))
    return render_template('health_form.html')

@app.route('/scan', methods=['GET', 'POST'])
@login_required
def scan():
    # Enforce MCQ first
    if not session.get('health_form_completed'):
        flash('Please complete the health assessment form first.', 'warning')
        return redirect(url_for('health_form'))

    health_data = session.get('health_data', {})

    if request.method == 'POST':
        medicine_type = health_data.get('medicine_type', 'allopathic')
        disease = "Normal"
        confidence = 85.0
        image_path = ''

        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = filepath
            disease, confidence = predict_disease(filepath)

        disease_data = DISEASE_INFO.get(disease, DISEASE_INFO['Normal'])
        allocated_doctors = get_doctors_for_disease(disease, medicine_type)
        allocated_ids = [d['id'] for d in allocated_doctors]

        scan_data = {
            'id': len(scans_db) + 1,
            'user_email': session['user_email'],
            'user_name': session['user_name'],
            'user_phone': users_db.get(session['user_email'], {}).get('phone', 'N/A'),
            'image_path': image_path,
            'disease': disease,
            'confidence': confidence,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'disease_info': disease_data,
            'medicine_type': medicine_type,
            'health_data': health_data,
            'allocated_doctor_ids': allocated_ids,
            'allocated_doctors': allocated_doctors
        }
        scans_db.append(scan_data)
        # Clear health form session so next scan requires fresh form
        session.pop('health_form_completed', None)

        return render_template('result.html',
                               scan=scan_data,
                               disease_info=disease_data,
                               confidence=confidence,
                               medicine_type=medicine_type,
                               allocated_doctors=allocated_doctors)

    return render_template('scan.html', health_data=health_data)

@app.route('/scan/<int:scan_id>')
@login_required
def view_scan(scan_id):
    scan = next((s for s in scans_db if s['id'] == scan_id), None)
    if not scan:
        flash('Scan not found.', 'danger')
        return redirect(url_for('client_dashboard'))
    if session.get('user_role') != 'doctor' and scan['user_email'] != session['user_email']:
        flash('Access denied.', 'danger')
        return redirect(url_for('client_dashboard'))
    return render_template('result.html',
                           scan=scan,
                           disease_info=scan.get('disease_info', {}),
                           confidence=scan.get('confidence', 0),
                           medicine_type=scan.get('medicine_type', 'allopathic'),
                           allocated_doctors=scan.get('allocated_doctors', []))

@app.route('/doctors')
@login_required
def doctors():
    # Build per-doctor live rating stats from feedbacks_db
    doctor_stats = {}
    for f in feedbacks_db:
        did = f['doctor_id']
        if did not in doctor_stats:
            doctor_stats[did] = {'total': 0, 'count': 0}
        doctor_stats[did]['total'] += f['rating']
        doctor_stats[did]['count'] += 1
    # Attach live stats to a copy of DOCTORS
    doctors_with_stats = []
    for d in DOCTORS:
        dc = dict(d)
        stats = doctor_stats.get(d['id'])
        if stats and stats['count'] > 0:
            dc['live_rating']  = round(stats['total'] / stats['count'], 1)
            dc['live_reviews'] = stats['count']
        else:
            dc['live_rating']  = d['rating']
            dc['live_reviews'] = d['reviews']
        doctors_with_stats.append(dc)
    return render_template('doctors.html', doctors=doctors_with_stats)

@app.route('/book-appointment/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def book_appointment(doctor_id):
    doctor = next((d for d in DOCTORS if d['id'] == doctor_id), None)
    if not doctor:
        flash('Doctor not found.', 'danger')
        return redirect(url_for('doctors'))

    min_date = datetime.now().strftime('%Y-%m-%d')

    if request.method == 'POST':
        date      = request.form.get('date', '')
        time      = request.form.get('time', '')
        reason    = request.form.get('reason', '')
        notes     = request.form.get('notes', '')
        visit_type = request.form.get('visit_type', 'In-Person')

        appt = {
            'id': len(appointments_db) + 1,
            'user_email': session['user_email'],
            'user_name': session['user_name'],
            'doctor_id': doctor_id,
            'doctor_name': doctor['name'],
            'doctor_spec': doctor['specialization'],
            'doctor_hospital': doctor['hospital'],
            'doctor_image': doctor.get('image', ''),
            'date': date,
            'time': time,
            'visit_type': visit_type,
            'reason': reason,
            'notes': notes,
            'booked_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'Pending'   # Awaiting doctor acceptance
        }
        appointments_db.append(appt)

        # ── Simulate email to PATIENT (Pending notice) ──
        print(f"[EMAIL → PATIENT] To: {session['user_email']}")
        print(f"[EMAIL → PATIENT] Subject: ⏳ Appointment Request Sent – {doctor['name']}")
        print(f"[EMAIL → PATIENT] Dear {session['user_name']}, your {visit_type} appointment request")
        print(f"[EMAIL → PATIENT]   Doctor : {doctor['name']}")
        print(f"[EMAIL → PATIENT]   Date   : {date} at {time}")
        print(f"[EMAIL → PATIENT]   Status : PENDING – waiting for doctor confirmation")

        # ── Simulate email to DOCTOR (new request) ──
        print(f"[EMAIL → DOCTOR]  To: {doctor['email']}")
        print(f"[EMAIL → DOCTOR]  Subject: 📅 New Appointment Request from {session['user_name']}")
        print(f"[EMAIL → DOCTOR]  Please log in to your dashboard to Accept or Decline.")
        print(f"[EMAIL → DOCTOR]    Patient : {session['user_name']} ({session['user_email']})")
        print(f"[EMAIL → DOCTOR]    Date    : {date} at {time}  |  Type: {visit_type}")
        print(f"[EMAIL → DOCTOR]    Reason  : {reason}")
        if notes:
            print(f"[EMAIL → DOCTOR]    Notes   : {notes}")

        flash(
            f'\u23f3 Appointment request sent to {doctor["name"]} for {date} at {time}. '
            f'Status is <strong>Pending</strong> – you will be notified when the doctor confirms.',
            'info'
        )
        return redirect(url_for('client_dashboard'))

    return render_template('book_appointment.html', doctor=doctor, min_date=min_date)

@app.route('/education')
def education():
    return render_template('education.html', disease_info=DISEASE_INFO)

@app.route('/about')
def about():
    return render_template('about.html')

# ─────────────────────────────────────────────
# APPOINTMENT ACCEPT / REJECT (Doctor only)
# ─────────────────────────────────────────────
@app.route('/appointment/<int:appt_id>/accept', methods=['POST'])
@login_required
@doctor_required
def accept_appointment(appt_id):
    appt = next((a for a in appointments_db if a['id'] == appt_id), None)
    if not appt or appt['doctor_id'] != session.get('doctor_id'):
        flash('Appointment not found or access denied.', 'danger')
        return redirect(url_for('doctor_dashboard'))

    appt['status'] = 'Confirmed'
    appt['accepted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Email to PATIENT – Confirmed ──
    print(f"[EMAIL → PATIENT] To: {appt['user_email']}")
    print(f"[EMAIL → PATIENT] Subject: ✅ Appointment CONFIRMED – {appt['doctor_name']}")
    print(f"[EMAIL → PATIENT] Dear {appt['user_name']}, your {appt['visit_type']} appointment")
    print(f"[EMAIL → PATIENT]   Doctor  : {appt['doctor_name']}")
    print(f"[EMAIL → PATIENT]   Date    : {appt['date']} at {appt['time']}")
    print(f"[EMAIL → PATIENT]   Status  : ✅ CONFIRMED")

    flash(
        f"✅ Appointment for <strong>{appt['user_name']}</strong> on {appt['date']} at "
        f"{appt['time']} has been <strong>Confirmed</strong>. Patient notified by email.",
        'success'
    )
    return redirect(url_for('doctor_dashboard'))


@app.route('/appointment/<int:appt_id>/reject', methods=['POST'])
@login_required
@doctor_required
def reject_appointment(appt_id):
    appt = next((a for a in appointments_db if a['id'] == appt_id), None)
    if not appt or appt['doctor_id'] != session.get('doctor_id'):
        flash('Appointment not found or access denied.', 'danger')
        return redirect(url_for('doctor_dashboard'))

    appt['status'] = 'Rejected'
    appt['rejected_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Email to PATIENT – Rejected ──
    print(f"[EMAIL → PATIENT] To: {appt['user_email']}")
    print(f"[EMAIL → PATIENT] Subject: ❌ Appointment Request Declined – {appt['doctor_name']}")
    print(f"[EMAIL → PATIENT] Dear {appt['user_name']}, unfortunately your appointment request")
    print(f"[EMAIL → PATIENT]   Doctor : {appt['doctor_name']}  Date: {appt['date']} at {appt['time']}")
    print(f"[EMAIL → PATIENT]   Status : ❌ DECLINED – please rebook or contact another doctor")

    flash(
        f"Appointment for <strong>{appt['user_name']}</strong> on {appt['date']} has been "
        f"<strong>Declined</strong>. Patient notified by email.",
        'warning'
    )
    return redirect(url_for('doctor_dashboard'))


# ─────────────────────────────────────────────
# PATIENT FEEDBACK / RATING
# ─────────────────────────────────────────────
@app.route('/feedback/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def submit_feedback(doctor_id):
    doctor = next((d for d in DOCTORS if d['id'] == doctor_id), None)
    if not doctor:
        flash('Doctor not found.', 'danger')
        return redirect(url_for('client_dashboard'))

    # Only patients with a Confirmed appointment can rate
    user_appts = [
        a for a in appointments_db
        if a['user_email'] == session['user_email']
        and a['doctor_id'] == doctor_id
        and a['status'] == 'Confirmed'
    ]
    if not user_appts:
        flash('You can only rate a doctor after a confirmed appointment.', 'warning')
        return redirect(url_for('client_dashboard'))

    # Check existing feedback
    existing = next(
        (f for f in feedbacks_db
         if f['user_email'] == session['user_email'] and f['doctor_id'] == doctor_id),
        None
    )

    if request.method == 'POST':
        try:
            rating = max(1, min(5, int(request.form.get('rating', 5))))
        except (ValueError, TypeError):
            rating = 5
        comment = request.form.get('comment', '').strip()[:500]

        if existing:
            existing['rating']     = rating
            existing['comment']    = comment
            existing['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            flash(f'Your feedback for {doctor["name"]} has been updated! ⭐ {rating}/5', 'success')
        else:
            feedbacks_db.append({
                'id':         len(feedbacks_db) + 1,
                'user_email': session['user_email'],
                'user_name':  session['user_name'],
                'doctor_id':  doctor_id,
                'doctor_name': doctor['name'],
                'rating':     rating,
                'comment':    comment,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            flash(f'Thank you for rating {doctor["name"]}! ⭐ {rating}/5 — your feedback helps others.', 'success')

        return redirect(url_for('client_dashboard'))

    return render_template('feedback.html', doctor=doctor, existing=existing)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)

