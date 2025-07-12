# ✅ FILE: app.py — Maternal Chatbot with Auto Mood Detection

from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# === Load the Topic Model and Vectorizer ===
topic_model = joblib.load("topic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# === Knowledge Base (Topic → Answer) ===
knowledge_base = {
    "bleeding": "Bleeding during pregnancy can be normal in early stages, but always consult a doctor if it occurs.",
    "fever": "High fever during pregnancy should be checked by a doctor to avoid any risk to the baby.",
    "folic acid": "Folic acid helps prevent neural tube defects and is essential during early pregnancy.",
    "morning sickness": "Morning sickness is common. Small meals and hydration can help ease it.",
    "back pain": "Back pain is common due to changes in posture and weight. Light stretching may help.",
    "ultrasound": "Ultrasounds help track baby’s growth and detect any issues. They are generally safe.",
    "diet": "A balanced diet with fruits, vegetables, and proteins supports healthy pregnancy.",
    "mood swings": "Mood swings are common. Relaxation techniques and emotional support can help.",
    "cravings": "Cravings are typical. Try to maintain balanced nutrition along with satisfying them safely.",
    "baby kicks": "Feeling baby kicks is reassuring. Report any noticeable decrease to your doctor.",
    "swelling": "Mild swelling is normal, but sudden or severe swelling may need medical attention.",
    "sleep": "Sleep challenges are common. Use pillows for support and maintain a relaxing bedtime routine.",
    "heartburn": "Avoid large meals, spicy foods, and lie with your head elevated to ease heartburn.",
    "stretch marks": "Moisturizing and staying hydrated may help reduce stretch marks.",
    "labor signs": "Signs of labor include contractions, water breaking, and lower back pressure.",
    "medications": "Only take medications approved by your doctor during pregnancy.",
    "prenatal vitamins": "They provide essential nutrients like folic acid and iron for baby’s growth.",
    "exercise": "Moderate activity like walking or prenatal yoga is usually safe unless advised otherwise.",
    "gestational diabetes": "It can be managed with diet, exercise, and sometimes insulin.",
    "hydration": "Drink enough fluids to stay hydrated and support increased blood volume.",
    "weight gain": "Expected weight gain varies. Your doctor can provide personalized guidance.",
    "caffeine": "Limit caffeine to less than 200mg/day, about one regular cup of coffee.",
    "cramps": "Mild cramps can be normal. Severe or persistent cramps should be reported.",
    "travel": "Travel is usually safe until late in pregnancy. Check with your doctor beforehand.",
    "doctor visits": "Regular visits are essential to monitor your and your baby’s health.",
    "sex": "Sex is generally safe during pregnancy unless your doctor advises against it.",
    "hair dye": "Use dye in a well-ventilated space and avoid harsh chemicals during the first trimester.",
    "vaccines": "Vaccines like Tdap and flu are recommended for maternal and fetal protection.",
    "preterm labor": "Watch for early signs and consult your doctor if you suspect anything unusual.",
    "high-risk pregnancy": "High-risk pregnancies need closer monitoring and specialized care.",
    "work stress": "Take breaks, delegate tasks, and speak openly with your employer when needed.",
    "baby development": "Your baby’s brain and body are developing rapidly. Good nutrition and bonding help.",
    "braxton hicks": "These are practice contractions and usually go away with rest or hydration.",
    "immunity": "Healthy food, hygiene, and enough sleep help boost immunity during pregnancy.",
    "constipation": "Fiber-rich foods, fluids, and gentle movement help relieve constipation.",
    "body temperature": "Feeling warm is normal, but excessive heat or fever should be reported.",
    "first kick": "Feeling your baby's first kick means healthy development!",
    "baby girl": "Knowing your baby’s gender is an exciting part of your journey. Congratulations!",
    "connected": "Feeling connected with your baby is a beautiful part of pregnancy. Keep nurturing that bond.",
    "partner": "Having a supportive partner can make pregnancy smoother and more joyful.",
    "hard pregnancy": "Pregnancy is a complex journey. It’s okay to find it difficult — you're doing great.",
    "doctor": "If you're feeling unheard, it's okay to ask more questions or seek a second opinion.",
    "delivery": "Worrying about delivery is normal. Prepare by creating a birth plan with your doctor.",
    "happens": "Pregnancy fears are valid. But most pregnancies go smoothly with regular care.",
    "checkup": "It's natural to feel anxious before checkups — try deep breathing or bring someone with you.",
    "ready": "Many moms feel unsure if they’re ready. Doubts show how much you care already.",
    "outcomes": "Try not to fixate on the worst outcomes. Focus on healthy habits and support systems."
}

# === Custom Mood Tone by Topic ===
topic_tone_prefix = {
    "bleeding": "😟 I understand this can be alarming.",
    "fever": "😟 Please take care, fever can be serious.",
    "folic acid": "😊 Great question on staying healthy!",
    "morning sickness": "😕 Many face this — let’s ease it.",
    "back pain": "😖 That sounds uncomfortable.",
    "ultrasound": "🙂 Let’s talk about scans.",
    "diet": "🥗 Healthy choices matter!",
    "mood swings": "😵 It’s okay to feel mixed emotions.",
    "cravings": "😋 Cravings are totally normal.",
    "baby kicks": "👶 How exciting!",
    "swelling": "😟 Let’s check on that.",
    "sleep": "😴 Rest is important.",
    "heartburn": "🔥 Let’s calm that burn.",
    "stretch marks": "🙂 Many moms ask this.",
    "labor signs": "🕓 Let’s prepare for the big moment.",
    "medications": "💊 Safety first!",
    "prenatal vitamins": "👍 Good job staying on track.",
    "exercise": "🏃 Let’s keep it safe.",
    "gestational diabetes": "📊 Let’s manage it together.",
    "hydration": "💧 Stay hydrated!",
    "weight gain": "📈 Let’s check healthy ranges.",
    "caffeine": "☕ Let’s sip smartly.",
    "cramps": "😣 Let’s make sure it’s normal.",
    "travel": "✈️ Let’s keep things safe.",
    "doctor visits": "🩺 Staying updated is great!",
    "sex": "❤️ Many moms ask this.",
    "hair dye": "🎨 Safety comes first.",
    "vaccines": "💉 Let’s protect you and baby.",
    "preterm labor": "⚠️ Be alert to early signs.",
    "high-risk pregnancy": "🚨 Extra care matters.",
    "work stress": "🧠 Let’s manage that load.",
    "baby development": "🧠 Growing smart & strong!",
    "braxton hicks": "🔁 Just practice contractions.",
    "immunity": "🛡️ Let’s boost your defense.",
    "constipation": "💩 Let’s fix that discomfort.",
    "body temperature": "🌡️ Feeling warm is okay, but be careful.",
    "first kick": "🎉 How precious!",
    "baby girl": "👧 Congratulations!",
    "connected": "❤️ Bonding is beautiful.",
    "partner": "🤝 Support is golden.",
    "hard pregnancy": "💔 You’re stronger than you know.",
    "doctor": "👩‍⚕️ Let’s talk advocacy.",
    "delivery": "📦 Getting close!",
    "happens": "🌀 It’s okay to worry.",
    "checkup": "😬 Many feel anxious too.",
    "ready": "🍼 Many wonder this.",
    "outcomes": "📉 Let’s stay hopeful."
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please enter a message."})

    # Detect topic
    vector = vectorizer.transform([user_input])
    predicted_topic = topic_model.predict(vector)[0]

    # Get response from knowledge base
    response = knowledge_base.get(predicted_topic, "I'm here to help. Could you share more details?")

    # Get custom tone
    mood_prefix = topic_tone_prefix.get(predicted_topic, "🙂 Here's what I can share:")

    # Final combined response
    final = f"{mood_prefix} {response}"
    return jsonify({"reply": final})

if __name__ == '__main__':
    app.run(debug=True)
