import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- KONFIGURASI DASAR ---
st.set_page_config(page_title="Radar Proyeksi H+7", layout="wide", initial_sidebar_state="collapsed")

# --- INJEKSI CSS KUSTOM (CLEAN DARK MODE) ---
st.markdown("""
    <style>
    .stApp { background-color: #0B101E; color: #FFFFFF; }
    div[data-testid="metric-container"] {
        background-color: #151C2C;
        border: 1px solid #25324B;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    div[data-testid="stMetricLabel"] { color: #8B9BB4 !important; font-weight: 600; font-size: 16px; justify-content: center; }
    div[data-testid="stMetricValue"] { color: #22D3EE !important; font-size: 36px; font-weight: bold; }
    div[data-testid="stMetricDelta"] { font-size: 18px; justify-content: center; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 RADAR PROYEKSI AI H+7")
st.markdown("<p style='color: #8B9BB4; font-size: 18px;'>Fokus murni pada probabilitas arah harga satu minggu ke depan.</p>", unsafe_allow_html=True)

# --- INPUT TUNGGAL ---
col_input, col_btn = st.columns([4, 1])
with col_input:
    kode_input = st.text_input("Masukkan Kode Target (Misal: BBCA, NISP):", "BBCA, NISP")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    tombol_pindai = st.button("🚀 BIDIK TARGET", use_container_width=True)

st.markdown("---")

# --- MAIN LOOP (FOKUS AI) ---
if tombol_pindai:
    daftar_saham = [s.strip().upper() for s in kode_input.split(',') if s.strip()]
    daftar_saham_fixed = [s if s.endswith('.JK') else f"{s}.JK" for s in daftar_saham]
    
    for kode_saham in daftar_saham_fixed:
        st.markdown(f"<h3 style='color: #E2E8F0; text-align: center; margin-top: 30px;'>KODE OPERASI: {kode_saham}</h3>", unsafe_allow_html=True)
        
        try:
            ticker = yf.Ticker(kode_saham)
            df = ticker.history(period='90d') # Cukup 90 hari untuk AI
            
            if df.empty:
                st.error(f"Data {kode_saham} tidak ditemukan.")
                continue

            prices = df['Close'].values[-60:]
            days = np.arange(len(prices)).reshape(-1, 1)
            weights = np.exp(np.linspace(-2., 0., len(prices)))
            
            poly = PolynomialFeatures(degree=2)
            days_poly = poly.fit_transform(days)
            
            model = LinearRegression()
            model.fit(days_poly, prices, sample_weight=weights)
            
            r2_score = model.score(days_poly, prices, sample_weight=weights)
            
            hari_prediksi = 7
            future_days = np.arange(len(prices), len(prices) + hari_prediksi).reshape(-1, 1)
            future_poly = poly.transform(future_days)
            prediction = model.predict(future_poly)
            
            residuals = prices - model.predict(days_poly)
            std_dev = np.std(residuals)
            upper_band = prediction + (1.5 * std_dev)
            lower_band = prediction - (1.5 * std_dev)

            harga_skrg = prices[-1]
            target_h7 = prediction[-1]
            persentase = ((target_h7 - harga_skrg) / harga_skrg) * 100

            # --- PANEL METRIK RAKSASA ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Harga Saat Ini", f"Rp {harga_skrg:,.0f}")
            c2.metric("Proyeksi AI (H+7)", f"Rp {target_h7:,.0f}", f"{persentase:+.2f}%")
            
            # Menerjemahkan R2 Score menjadi bahasa manusia yang sederhana
            if r2_score > 0.75:
                status_ai = "Sangat Kuat & Valid"
            elif r2_score > 0.50:
                status_ai = "Biasa (Normal)"
            else:
                status_ai = "Acak (Hati-hati)"
                
            c3.metric("Tingkat Keyakinan AI", status_ai)

            # --- GRAFIK BERSIH ---
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Harga masa lalu
            ax.plot(df.index[-60:], prices, label='Jejak Harga 60 Hari', color='#FACC15', linewidth=2.5)
            
            # Garis Prediksi H+7
            tgl_pred = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=hari_prediksi)
            ax.plot(tgl_pred, prediction, color='#22D3EE', linestyle='--', marker='o', markersize=8, label='Jalur Proyeksi H+7', linewidth=2)
            
            # Awan Probabilitas
            ax.fill_between(tgl_pred, lower_band, upper_band, color='#22D3EE', alpha=0.15, label='Batas Toleransi (Aman)')
            
            # Kosmetik minimalis
            ax.legend(facecolor='#151C2C', edgecolor='#25324B', labelcolor='#E2E8F0', fontsize=12)
            ax.set_facecolor('#0B101E')
            ax.tick_params(colors='#8B9BB4', labelsize=10)
            ax.grid(color='#25324B', linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#25324B')
            ax.spines['bottom'].set_color('#25324B')
            
            fig.patch.set_facecolor('#0B101E')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Gagal memproses {kode_saham}: {e}")

    st.success("🏁 EKSEKUSI SELESAI")
