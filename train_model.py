# ✅ FILE: train_model.py — Train and save ML model for maternal topic & mood classification

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# === 🔸 Training Data ===
X_train = [
    "I'm bleeding during my second trimester",
    "I feel dizzy and have a fever",
    "Can I take folic acid and iron tablets?",
    "I'm experiencing morning sickness",
    "My back hurts a lot at night",
    "How often are ultrasounds safe?",
    "What should I eat during pregnancy?",
    "I'm having mood swings often",
    "I'm craving spicy food lately",
    "Baby kicks have reduced today",
    "My legs are swollen",
    "I feel tired and can't sleep well",
    "I feel burning in my chest after meals",
    "Will I get stretch marks?",
    "What are signs of labor?",
    "Can I take paracetamol while pregnant?",
    "I forgot my prenatal vitamin today",
    "Is walking daily good during pregnancy?",
    "What causes gestational diabetes?",
    "Is too much water harmful?",
    "How much weight should I gain during pregnancy?",
    "Can I drink coffee while pregnant?",
    "Is it normal to have cramps in early pregnancy?",
    "Can I travel by flight in my third trimester?",
    "How often should I visit the doctor?",
    "Is sex safe during pregnancy?",
    "Can I dye my hair while pregnant?",
    "What exercises should I avoid during pregnancy?",
    "What should I eat in pregnancy?",
    "What vaccines do I need during pregnancy?",
    "How to prevent preterm labor?",
    "What is a high-risk pregnancy?",
    "How to manage work stress while pregnant?",
    "How can I improve my baby’s brain development?",
    "What are Braxton Hicks contractions?",
    "How can I boost my immunity during pregnancy?",
    "Can I sleep on my stomach in early pregnancy?",
    "How to relieve constipation during pregnancy?",
    "Is it normal to feel hot during pregnancy?",
    "I felt my baby's first kick!",
    "We just found out it's a baby girl!",
    "I feel deeply connected with my baby",
    "My partner is super supportive, I’m grateful",
    "Why does no one tell how hard pregnancy really is?",
    "My doctor keeps ignoring my concerns!",
    "I have a lot of worries about how the delivery will go",
    "What if something happens to my baby?",
    "I'm anxious before every checkup",
    "What if I’m not ready to be a mom?",
    "I keep thinking about the worst outcomes"
]

y_train = [
    "bleeding", "fever", "folic acid", "morning sickness", "back pain",
    "ultrasound", "diet", "mood swings", "cravings", "baby kicks",
    "swelling", "sleep", "heartburn", "stretch marks", "labor signs",
    "medications", "prenatal vitamins", "exercise", "gestational diabetes", "hydration",
    "weight gain", "caffeine", "cramps", "travel", "doctor visits",
    "sex", "hair dye", "exercise", "diet",
    "vaccines", "preterm labor", "high-risk pregnancy", "work stress", "baby development",
    "braxton hicks", "immunity", "sleep", "constipation", "body temperature",
    "first kick", "baby girl", "connected", "partner", "hard pregnancy",
    "doctor", "delivery", "happens", "checkup", "ready", "outcomes"
]

# ✅ Validate data length
assert len(X_train) == len(y_train), f"Mismatch: {len(X_train)} texts and {len(y_train)} labels."

# === 🔠 Vectorize Text
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X_train)

# === 🤖 Train the Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_vectors, y_train)

# === 📎 Save the Model and Vectorizer
joblib.dump(model, "topic_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model training complete! Files saved: topic_model.pkl and vectorizer.pkl")
