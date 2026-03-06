# DermaScan AI – Rebuilt Application

## What's New (All 11 Requirements Implemented)

1. **Modern UI** – Navy/Teal color scheme, DM Serif + Outfit fonts, animations, glassmorphism
2. **Google Login** – Google OAuth button (requires Flask-Dance setup for production)
3. **Forgot Password** – Full forgot/reset password flow with secure tokens
4. **Phone Number on Register** – Required phone field in registration form
5. **MCQ Before Scan** – 9 mandatory health questions must be answered before scan is allowed
6. **Disease-Based Doctor Filter** – Doctors auto-allocated based on detected disease + medicine preference
7. **Doctor Login Credentials** – Each doctor has email + password; only sees their allocated patients
8. **Patient Contact Info** – Doctor dashboard shows patient email, phone for easy contact
9. **Login Required** – Scan and doctor contact pages require authentication
10. **Icons Instead of Emojis** – Font Awesome 6.5 icons used throughout
11. **AI Medicines** – Claude API generates medicine recommendations dynamically (not hardcoded)

## Doctor Login Credentials
| Doctor | Email | Password |
|--------|-------|----------|
| Dr. Priya Sharma | dr.priya.sharma@skincare.in | Priya@2024 |
| Dr. Rajesh Kumar | dr.rajesh.kumar@ayurskin.in | Rajesh@2024 |
| Dr. Meena Patel | dr.meena.patel@homeocare.in | Meena@2024 |
| Dr. Anand Menon | dr.anand.menon@dermacure.in | Anand@2024 |
| Dr. Sunita Joshi | dr.sunita.joshi@panchakarma.in | Sunita@2024 |

## Demo Patient
- Email: patient@demo.com
- Password: demo123

## Running the App
```bash
pip install -r requirements.txt
python app.py
```
Visit: http://localhost:5000
