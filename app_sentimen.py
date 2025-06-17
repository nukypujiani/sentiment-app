import streamlit as st
import joblib

# Load pipeline
preprocessor = joblib.load('preprocessor_model.pkl')
tfidf_pipeline = joblib.load('tfidf_pipeline.pkl')
feature_selector = joblib.load('selected_features.pkl')
bwelm_model = joblib.load('bwelm_model_fs.pkl')
hybrid_model = joblib.load('fr_optuna_model_fs.pkl')

st.title("📊 Sentiment Analysis App")
st.markdown("Masukkan ulasan aplikasi dan dapatkan prediksi sentimen.")

user_input = st.text_area("Tulis ulasan Anda di sini:")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
    else:
        processed_text = preprocessor.transform([user_input])
        tfidf_features = tfidf_pipeline.transform(processed_text)
        selected_features = feature_selector.transform(tfidf_features)
        
        bwelm_pred = bwelm_model.predict(selected_features)[0]
        bwelm_proba = bwelm_model.predict_proba(selected_features)[0]
        
        hybrid_pred = hybrid_model.predict(selected_features)[0]
        hybrid_proba = hybrid_model.predict_proba(selected_features)[0]
        
        label_map = {-1: "Negatif", 1: "Positif"}
        
        st.subheader("Hasil Prediksi")
        st.write(f"**BWELM**: {label_map.get(bwelm_pred)} (Neg: {bwelm_proba[0]:.4f}, Pos: {bwelm_proba[1]:.4f})")
        st.write(f"**Hybrid**: {label_map.get(hybrid_pred)} (Neg: {hybrid_proba[0]:.4f}, Pos: {hybrid_proba[1]:.4f})")
