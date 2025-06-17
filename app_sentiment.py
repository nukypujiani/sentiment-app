import streamlit as st
import joblib
import re
import emoji
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indoNLP.preprocessing import replace_slang, replace_word_elongation, emoji_to_words

# ====== Preprocessor Class ======
class Preprocessor:
    def __init__(self):
        stop_factory = StopWordRemoverFactory()
        self.stop_remover = stop_factory.create_stop_word_remover()
        stem_factory = StemmerFactory()
        self.stemmer = stem_factory.create_stemmer()

    def cleansing(self, text):
        text = text.lower()
        text = emoji.demojize(text, delimiters=("", ""))
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalisasi(self, text):
        text = replace_slang(text)
        text = replace_word_elongation(text)
        text = emoji_to_words(text)
        return text

    def remove_repeated_emoji(self, text):
        emoji_pattern = re.compile("[" 
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        emojis = emoji_pattern.findall(text)
        for e in set(emojis):
            text = re.sub(f"{e}+", e, text)
        return text

    def process(self, text):
        text = self.cleansing(text)
        text = self.normalisasi(text)
        text = self.remove_repeated_emoji(text)
        tokens = text.split()
        tokens = [word for word in tokens if self.stop_remover.remove(word) != '']
        text = ' '.join(tokens)
        text = self.stemmer.stem(text)
        return text

# ====== Load pipeline model ======
tfidf_pipeline = joblib.load('tfidf_pipeline.pkl')
feature_selector = joblib.load('selected_features.pkl')
bwelm_model = joblib.load('bwelm_model_fs.pkl')
hybrid_model = joblib.load('fr_optuna_model_fs.pkl')

# ====== Streamlit App ======
st.title("📊 Sentiment Analysis App")
st.markdown("Masukkan ulasan aplikasi dan dapatkan prediksi sentimen.")

user_input = st.text_area("Tulis ulasan Anda di sini:")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
    else:
        preprocessor = Preprocessor()
        cleaned = preprocessor.process(user_input)

        tfidf_features = tfidf_pipeline.transform([cleaned])
        selected_features_data = tfidf_features[:, [int(f.split('_')[1])-1 for f in feature_selector]]

        # BWELM prediction
        bwelm_pred = bwelm_model.predict(selected_features_data)[0]
        bwelm_proba = bwelm_model.predict_proba(selected_features_data)[0]

        # Hybrid prediction
        hybrid_pred = hybrid_model.predict(bwelm_proba.reshape(1, -1))[0]
        hybrid_proba = hybrid_model.predict_proba(bwelm_proba.reshape(1, -1))[0]

        label_map = {-1: "Negatif", 1: "Positif"}

        st.subheader("Hasil Prediksi")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**BWELM Model**")
            st.write(f"Sentimen: {label_map.get(bwelm_pred, 'Tidak diketahui')}")
            st.write(f"Probabilitas Negatif: {bwelm_proba[0]:.4f}")
            st.write(f"Probabilitas Positif: {bwelm_proba[1]:.4f}")

        with col2:
            st.markdown("**Hybrid Model**")
            st.write(f"Sentimen: {label_map.get(hybrid_pred, 'Tidak diketahui')}")
            st.write(f"Probabilitas Negatif: {hybrid_proba[0]:.4f}")
            st.write(f"Probabilitas Positif: {hybrid_proba[1]:.4f}")

        st.success("Prediksi berhasil dilakukan!")
