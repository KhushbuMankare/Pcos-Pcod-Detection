import streamlit as st
import numpy as np
import pickle

# Load trained model
pickled_model = pickle.load(open("modelfinal1.pkl", "rb"))

st.title("🔬 HerHelth- PCOS Detection")
st.subheader("Enter your medical details below:")

# --- Define the 20 features your model was trained on ---
features = [
    "Age (yrs)", "Weight (Kg)", "Height (Cm)", "BMI",
    "Waist (inch)", "Hip (inch)", "Waist-Hip Ratio",
    "Cycle length (days)", "Cycle (R/I)", "Pulse rate (bpm)",
    "RR (breaths/min)", "Hb (g/dl)", "FSH (mIU/mL)", "LH (mIU/mL)",
    "FSH/LH ratio", "TSH (mIU/L)", "AMH (ng/mL)", "PRL (ng/mL)",
    "Vit D3 (ng/mL)", "RBS (mg/dl)"
]

# --- Input section ---
inputs = {}
for feature in features:
    value = st.number_input(f"{feature}", value=0.0, step=0.1, key=feature)
    inputs[feature] = value

# Convert to numpy array for prediction
inputs_arr = np.array(list(inputs.values())).reshape(1, -1)

# --- Prediction ---
if st.button("🔍 Diagnose"):
    prediction = pickled_model.predict(inputs_arr)[0]
    prob = pickled_model.predict_proba(inputs_arr)[0][1] * 100  # Risk score in %

    if prediction == 1:
        st.error(f"⚠️ High chance of PCOS detected. (Risk Score: {prob:.2f}%)")
    else:
        st.success(f"✅ No significant signs of PCOS detected. (Risk Score: {prob:.2f}%)")

    # --- Personalized Recommendations ---
    st.subheader("📝 Personalized Recommendations")

    recs = []
    if inputs["BMI"] > 25:
        recs.append("Maintain a healthy weight with balanced diet & exercise.")
    if inputs["Cycle length (days)"] > 35 or inputs["Cycle (R/I)"] == 0:
        recs.append("Irregular cycles detected. Track periods & consult a gynecologist.")
    if inputs["AMH (ng/mL)"] > 5:
        recs.append("High AMH levels observed. May indicate ovarian dysfunction.")
    if inputs["Vit D3 (ng/mL)"] < 20:
        recs.append("Low Vitamin D3 detected. Consider supplements or sunlight exposure.")
    if inputs["RBS (mg/dl)"] > 120:
        recs.append("Elevated blood sugar. Reduce sugar intake & get glucose testing.")

    if recs:
        for r in recs:
            st.write(f"- {r}")
    else:
        st.write("✅ Your values look within normal range. Keep maintaining a healthy lifestyle!")

    # --- Next Steps ---
    st.subheader("🚀 Next Steps")
    if prediction == 1:
        st.write("- 📌 Schedule a gynecologist consultation.")
        st.write("- 📌 Get an ultrasound to check ovarian cysts.")
        st.write("- 📌 Follow lifestyle modifications (diet, exercise).")
    else:
        st.write("- ✅ Maintain your healthy lifestyle.")
        st.write("- ✅ Track menstrual cycles and routine health checkups.")
