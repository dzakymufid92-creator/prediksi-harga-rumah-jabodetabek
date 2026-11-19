import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Rumah Jabodetabek",
    page_icon="🏠",
    layout="wide"
)

# Muat model dan metadata sekali saja (disimpan di cache)
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_xgb.joblib")
    with open("meta.json", "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()

# Format angka ke Rupiah
def format_rupiah(angka):
    return f"Rp {angka:,.0f}".replace(",", ".")

# Label ramah pengguna untuk formulir
LABELS = {
    "city": "Kota / Kabupaten",
    "district": "Kecamatan / Area",
    "certificate": "Jenis Sertifikat",
    "land_size_m2": "Luas Tanah (m²)",
    "building_size_m2": "Luas Bangunan (m²)",
    "bedrooms": "Jumlah Kamar Tidur",
    "bathrooms": "Jumlah Kamar Mandi",
    "floors": "Jumlah Lantai",
    "carports": "Jumlah Carport"
}

# Menu Samping (Sidebar)
st.sidebar.title("Menu")
halaman = st.sidebar.radio(
    "Pilih Halaman:",
    ["Prediksi", "Performa Model", "Tentang Data"]
)

# ==========================================
# HALAMAN 1: PREDIKSI
# ==========================================
if halaman == "Prediksi":
    st.title("🏠 Prediksi Harga Rumah Tinggal Jabodetabek")
    st.write("Isi spesifikasi rumah di bawah ini untuk mendapatkan estimasi harga penawaran.")

    col_kiri, col_kanan = st.columns([1.2, 1], gap="large")

    with col_kiri:
        st.subheader("Isi Data Properti")
        with st.form("form_prediksi"):
            input_data = {}

            # Input Kategorikal (Dropdown)
            for col in meta["cat_cols"]:
                options = meta["cat_options"][col]
                input_data[col] = st.selectbox(LABELS.get(col, col), options)

            # Input Numerik (2 kolom)
            c1, c2 = st.columns(2)
            for idx, col in enumerate(meta["num_cols"]):
                r = meta["num_ranges"][col]
                target_col_ui = c1 if idx % 2 == 0 else c2
                with target_col_ui:
                    input_data[col] = st.number_input(
                        LABELS.get(col, col),
                        min_value=float(r["min"]),
                        max_value=float(r["max"]),
                        value=float(r["median"]),
                        step=1.0
                    )

            submitted = st.form_submit_button("Prediksi Harga", use_container_width=True)

    with col_kanan:
        st.subheader("Hasil Prediksi")
        if submitted:
            kolom_urutan = meta["num_cols"] + meta["cat_cols"]
            df_input = pd.DataFrame([input_data])[kolom_urutan]

            # Prediksi skala log lalu kembalikan ke Rupiah
            pred_log = model.predict(df_input)[0]
            pred_rp = float(np.expm1(pred_log))

            # Hitung rentang estimasi berdasarkan MAPE
            metrics_xgb = meta["metrics"].get("XGBoost (Hasil Tuning)") or list(meta["metrics"].values())[-1]
            mape_persen = metrics_xgb["MAPE (%)"] / 100.0
            batas_bawah = max(0, pred_rp * (1 - mape_persen))
            batas_atas = pred_rp * (1 + mape_persen)

            st.success(f"### Estimasi Harga:\n# {format_rupiah(pred_rp)}")
            st.info(
                f"**Perkiraan Rentang (±{metrics_xgb['MAPE (%)']:.1f}% MAPE):**\n\n"
                f"{format_rupiah(batas_bawah)} s.d. {format_rupiah(batas_atas)}"
            )
            st.caption(
                "*Catatan: Hasil ini merupakan estimasi harga penawaran (asking price) "
                "berdasarkan data listing tahun 2022, bukan penilaian resmi (appraisal).*"
            )
        else:
            st.info("Silakan lengkapi data properti di sebelah kiri, lalu tekan tombol **Prediksi Harga**.")

# ==========================================
# HALAMAN 2: PERFORMA MODEL
# ==========================================
elif halaman == "Performa Model":
    st.title("📊 Performa Model")

    metrics_dict = meta["metrics"]
    xgb_key = "XGBoost (Hasil Tuning)" if "XGBoost (Hasil Tuning)" in metrics_dict else list(metrics_dict.keys())[-1]
    m_xgb = metrics_dict[xgb_key]

    st.subheader(f"Evaluasi Model Utama: {xgb_key}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE", format_rupiah(m_xgb["MAE (Rp)"]))
    k2.metric("RMSE", format_rupiah(m_xgb["RMSE (Rp)"]))
    k3.metric("R²", f"{m_xgb['R2']:.4f}")
    k4.metric("MAPE", f"{m_xgb['MAPE (%)']:.2f}%")

    st.divider()

    col_grafik1, col_grafik2 = st.columns(2)

    with col_grafik1:
        st.subheader("Perbandingan Model (R²)")
        df_metrics = pd.DataFrame(metrics_dict).T
        st.bar_chart(df_metrics["R2"])
        st.dataframe(
            df_metrics.style.format({
                "MAE (Rp)": "Rp {:,.0f}",
                "RMSE (Rp)": "Rp {:,.0f}",
                "R2": "{:.4f}",
                "MAPE (%)": "{:.2f}%"
            }),
            use_container_width=True
        )

    with col_grafik2:
        st.subheader("10 Fitur Paling Berpengaruh")
        try:
            xgb_model = model.named_steps["model"]
            prep = model.named_steps["prep"]
            feature_names = prep.get_feature_names_out()
            importances = xgb_model.feature_importances_

            df_imp = pd.DataFrame({
                "Fitur": [f.replace("num__", "").replace("cat__", "") for f in feature_names],
                "Kepentingan": importances
            }).sort_values("Kepentingan", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(df_imp["Fitur"][::-1], df_imp["Kepentingan"][::-1], color="#2E86C1")
            ax.set_xlabel("Tingkat Pengaruh (Feature Importance)")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Grafik fitur penting tidak dapat ditampilkan: {e}")

# ==========================================
# HALAMAN 3: TENTANG DATA
# ==========================================
elif halaman == "Tentang Data":
    st.title("ℹ️ Tentang Data & Batasan Sistem")
    st.markdown("""
    ### Sumber Data
    * **Dataset:** *Daftar Harga Rumah Jabodetabek* dari Kaggle (bersumber dari situs listing Rumah123, diakses akhir tahun 2022).
    * **Cakupan Wilayah:** Jakarta, Bogor, Depok, Tangerang, dan Bekasi (Jabodetabek).
    * **Objek:** Rumah tinggal (properti non-rumah telah disaring saat pembersihan data).

    ### Metode Pemodelan
    * Algoritma utama yang digunakan adalah **Extreme Gradient Boosting (XGBoost)** yang dioptimalkan menggunakan *RandomizedSearchCV* (5-fold cross validation).
    * Data dibagi menjadi **80% data latih** dan **20% data uji**, dengan penyaringan outlier harga pada persentil ke-1 hingga ke-99 serta transformasi logaritma (`log1p`) pada variabel harga.

    ### Batasan Penggunaan
    * Harga yang diprediksi adalah **harga penawaran (*asking price*)** pada iklan tahun 2022, bukan harga transaksi aktual.
    * Hasil estimasi hanya berfungsi sebagai **acuan awal** dan bukan pengganti penilaian resmi (*appraisal*).
    """)