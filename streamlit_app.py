import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- KONFIGURASI DASAR ---
st.set_page_config(page_title="The Strategic Commander", layout="wide", initial_sidebar_state="expanded")

# --- INJEKSI CSS KUSTOM (ULTIMATE DARK MODE) ---
st.markdown("""
    <style>
    .stApp { background-color: #0B101E; color: #FFFFFF; }
    div[data-testid="metric-container"] {
        background-color: #151C2C;
        border: 1px solid #25324B;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] { color: #8B9BB4 !important; font-weight: 600; font-size: 13px; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 24px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🎖️ THE STRATEGIC COMMANDER V6.0")
st.markdown("<p style='color: #8B9BB4;'>Radar Intelijen Terpadu: AI Prediction, Volume Flow & Fundamental Snapshot</p>", unsafe_allow_html=True)

# --- SIDEBAR CONTROL ---
st.sidebar.markdown("### 🕹️ COMMAND CENTER")
kode_input = st.sidebar.text_input("🎯 Target Saham (Gunakan Koma):", "BBCA, NISP, TINS")
hari_prediksi = st.sidebar.slider("📅 Rentang Proyeksi AI (Hari)", 1, 14, 7)
tombol_pindai = st.sidebar.button("🚀 EKSEKUSI PEMINDAIAN")

# --- FUNGSI INTELIJEN ---
def hitung_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- MAIN LOOP ---
if tombol_pindai:
    daftar_saham = [s.strip().upper() for s in kode_input.split(',') if s.strip()]
    daftar_saham_fixed = [s if s.endswith('.JK') else f"{s}.JK" for s in daftar_saham]
    
    for kode_saham in daftar_saham_fixed:
        st.markdown("<hr style='border: 1px solid #25324B;'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: #E2E8F0;'>🏢 ANALISIS STRATEGIS: {kode_saham}</h2>", unsafe_allow_html=True)
        
        try:
            ticker = yf.Ticker(kode_saham)
            df = ticker.history(period='120d')
            info = ticker.info # Penarikan data fundamental
            
            if df.empty:
                st.error(f"Data {kode_saham} tidak ditemukan.")
                continue

            # 1. MODUL FUNDAMENTAL (Value Snapshot)
            per = info.get('trailingPE', 0)
            pbv = info.get('priceToBook', 0)
            roe = info.get('returnOnEquity', 0) * 100
            
            # 2. MODUL AI & CONFIDENCE (R-Squared)
            prices = df['Close'].values[-60:]
            days = np.arange(len(prices)).reshape(-1, 1)
            weights = np.exp(np.linspace(-2., 0., len(prices)))
            poly = PolynomialFeatures(degree=2)
            days_poly = poly.fit_transform(days)
            
            model = LinearRegression()
            model.fit(days_poly, prices, sample_weight=weights)
            
            # Skor Keyakinan (R2)
            r2_score = model.score(days_poly, prices, sample_weight=weights)
            
            future_days = np.arange(len(prices), len(prices) + hari_prediksi).reshape(-1, 1)
            future_poly = poly.transform(future_days)
            prediction = model.predict(future_poly)
            
            # Zona Volatilitas
            residuals = prices - model.predict(days_poly)
            std_dev = np.std(residuals)
            upper_band = prediction + (1.5 * std_dev)
            lower_band = prediction - (1.5 * std_dev)

            # 3. MODUL VOLUME SPIKE
            last_vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].tail(20).mean()
            vol_spike = (last_vol / avg_vol)
            
            # --- DISPLAY DASHBOARD METRIK ---
            st.markdown("##### 📊 DATA FUNDAMENTAL & AI")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("PER (Price/Earnings)", f"{per:.2f}x" if per > 0 else "N/A")
            c2.metric("PBV (Price/Book)", f"{pbv:.2f}x" if pbv > 0 else "N/A")
            c3.metric("AI Confidence (R²)", f"{r2_score:.2f}", "Strong Pattern" if r2_score > 0.7 else "Weak Pattern")
            c4.metric("Volume Spike", f"{vol_spike:.2f}x", "Accumulation" if vol_spike > 1.5 else "Normal")
            c5.metric(f"Probabilitas H+{hari_prediksi}", f"{((prediction[-1]-prices[-1])/prices[-1]*100):+.2f}%")

            # --- DISPLAY GRAFIK (PRICE + VOLUME) ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            
            # Plot 1: Harga & Proyeksi
            ax1.plot(df.index[-60:], prices, label='Harga Aktual', color='#FACC15', linewidth=2)
            tgl_pred = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=hari_prediksi)
            ax1.plot(tgl_pred, prediction, color='#22D3EE', linestyle='--', marker='o', label='Proyeksi AI')
            ax1.fill_between(tgl_pred, lower_band, upper_band, color='#22D3EE', alpha=0.1, label='Zona Volatilitas')
            
            ax1.set_title(f"Price Action & AI Guidance: {kode_saham}", color='#E2E8F0', fontsize=14)
            ax1.legend(facecolor='#151C2C', edgecolor='#25324B', labelcolor='#E2E8F0')
            ax1.set_facecolor('#0B101E')
            ax1.tick_params(colors='#8B9BB4')
            ax1.grid(color='#25324B', linestyle='--', alpha=0.3)

            # Plot 2: Volume Histogram
            colors = ['#10B981' if df['Close'].iloc[i] > df['Open'].iloc[i] else '#EF4444' for i in range(len(df)-60, len(df))]
            ax2.bar(df.index[-60:], df['Volume'].tail(60), color=colors, alpha=0.8, label='Volume')
            ax2.axhline(avg_vol, color='white', linestyle=':', alpha=0.5, label='Avg Vol (20)')
            
            ax2.set_facecolor('#0B101E')
            ax2.tick_params(colors='#8B9BB4')
            ax2.grid(color='#25324B', linestyle='--', alpha=0.3)
            ax2.legend(facecolor='#151C2C', edgecolor='#25324B', labelcolor='#E2E8F0')
            
            fig.patch.set_facecolor('#0B101E')
            plt.subplots_adjust(hspace=0.05)
            st.pyplot(fig)

            # --- NEWS & SENTIMENT ---
            news = ticker.news[:3]
            if news:
                st.markdown("##### 📰 INTELIJEN BERITA TERBARU")
                for item in news:
                    with st.expander(f"📌 {item['title']}"):
                        st.write(f"Penerbit: {item['publisher']} | [Link Berita]({item['link']})")

        except Exception as e:
            st.error(f"Gagal memproses {kode_saham}: {e}")

    st.success("🏁 OPERASI PEMINDAIAN V6.0 SELESAI")
