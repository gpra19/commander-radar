import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- KONFIGURASI DASAR ---
st.set_page_config(page_title="The Strategic Commander", layout="wide")

# CSS Kustom untuk tampilan lebih profesional (Modern Dark Mode)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True) # ✅ INI YANG BENAR

st.title("🎖️ THE STRATEGIC COMMANDER V4.0")
st.subheader("Integrated Tactical Dashboard: AI Prediction + Technicals + Sentiment")

# --- SIDEBAR INPUT ---
st.sidebar.header("🕹️ CONTROL PANEL")
kode_input = st.sidebar.text_input("🎯 Masukkan Kode Saham (Pisahkan dengan koma):", "BBCA, NISP, TINS")
hari_prediksi = st.sidebar.slider("📅 Rentang Proyeksi (Hari)", 1, 14, 7)
tombol_pindai = st.sidebar.button("🚀 EKSEKUSI PEMINDAIAN")

# --- FUNGSI INTELIJEN ---
def hitung_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def dapatkan_sentimen(news_list):
    # Logika sederhana pemindai kata kunci sentimen
    pos_words = ['profit', 'up', 'bull', 'buy', 'gain', 'growth', 'laba', 'naik', 'beli', 'positif']
    neg_words = ['loss', 'down', 'bear', 'sell', 'drop', 'cut', 'rugi', 'turun', 'jual', 'perang', 'negatif']
    
    score = 0
    for news in news_list:
        text = news['title'].lower()
        for w in pos_words: 
            if w in text: score += 1
        for w in neg_words: 
            if w in text: score -= 1
    
    if score > 0: return "🟢 BULLISH SENTIMENT", "Sentimen berita cenderung positif."
    elif score < 0: return "🔴 BEARISH SENTIMENT", "Waspada sentimen negatif di media."
    return "⚪ NEUTRAL SENTIMENT", "Tidak ada berita mencolok hari ini."

# --- MAIN LOOP ---
if tombol_pindai:
    daftar_saham = [s.strip().upper() for s in kode_input.split(',') if s.strip()]
    daftar_saham_fixed = [s if s.endswith('.JK') else f"{s}.JK" for s in daftar_saham]
    
    for kode_saham in daftar_saham_fixed:
        st.markdown(f"## 🏢 ANALISIS STRATEGIS: {kode_saham}")
        
        try:
            # 1. Tarik Data & Indikator Klasik
            ticker = yf.Ticker(kode_saham)
            df = ticker.history(period='120d') # Ambil lebih panjang untuk MA
            
            if df.empty:
                st.error(f"Data {kode_saham} tidak ditemukan.")
                continue

            # Hitung MA dan RSI
            df['MA20'] = df['Close'].rolling(20).mean()
            df['RSI'] = hitung_rsi(df['Close'])
            rsi_now = df['RSI'].iloc[-1]
            
            # 2. Reaktor AI (Polinomial V3)
            prices = df['Close'].values[-60:] # Gunakan 60 hari terakhir untuk prediksi
            days = np.arange(len(prices)).reshape(-1, 1)
            weights = np.exp(np.linspace(-2., 0., len(prices)))
            poly = PolynomialFeatures(degree=2)
            days_poly = poly.fit_transform(days)
            
            model = LinearRegression()
            model.fit(days_poly, prices, sample_weight=weights)
            
            # Prediksi H+N
            future_days = np.arange(len(prices), len(prices) + hari_prediksi).reshape(-1, 1)
            future_poly = poly.transform(future_days)
            prediction = model.predict(future_poly)
            
            # Zona Volatilitas
            residuals = prices - model.predict(days_poly)
            std_dev = np.std(residuals)
            upper_band = prediction + (1.5 * std_dev)
            lower_band = prediction - (1.5 * std_dev)

            # --- DISPLAY DASHBOARD ---
            # Baris 1: Metrik AI & Teknikal
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Harga Terakhir", f"Rp {prices[-1]:,.0f}")
            c2.metric("RSI (14)", f"{rsi_now:.2f}", "Overbought" if rsi_now > 70 else "Oversold" if rsi_now < 30 else "Normal")
            c3.metric("Target Median", f"Rp {prediction[-1]:,.0f}")
            c4.metric("Probabilitas H+{0}".format(hari_prediksi), f"{((prediction[-1]-prices[-1])/prices[-1]*100):+.2f}%")

            # Baris 2: TRADING PLAN (Otomatis)
            st.markdown("### 📝 RENCANA OPERASI (TRADING PLAN)")
            entry_price = prices[-1]
            tp_price = upper_band[-1]
            sl_price = lower_band[-1]
            risk = entry_price - sl_price
            reward = tp_price - entry_price
            rr_ratio = reward / risk if risk > 0 else 0
            
            t1, t2, t3, t4 = st.columns(4)
            t1.info(f"**ENTRY AREA**\n\nRp {entry_price:,.0f}")
            t2.success(f"**TAKE PROFIT**\n\nRp {tp_price:,.0f}")
            t3.error(f"**STOP LOSS**\n\nRp {sl_price:,.0f}")
            t4.warning(f"**RISK/REWARD**\n\nRatio: {rr_ratio:.2f}")

            # Baris 3: Visualisasi
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Subplot 1: Price & Prediction
            ax1.plot(df.index[-60:], prices, label='Harga Aktual', color='gold', linewidth=2)
            ax1.plot(df.index[-60:], df['MA20'].tail(60), label='MA20 Trend', color='orange', alpha=0.6)
            
            tgl_pred = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=hari_prediksi)
            ax1.plot(tgl_pred, prediction, color='cyan', linestyle='--', marker='o', label='Proyeksi AI')
            ax1.fill_between(tgl_pred, lower_band, upper_band, color='cyan', alpha=0.1, label='Zona Volatilitas')
            
            ax1.set_title(f"Analisis Teknikal & AI: {kode_saham}", color='white')
            ax1.legend()
            ax1.set_facecolor('#0e1117')
            ax1.grid(alpha=0.1)

            # Subplot 2: RSI
            ax2.plot(df.index[-60:], df['RSI'].tail(60), color='magenta', label='RSI')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax2.set_ylim(0, 100)
            ax2.set_facecolor('#0e1117')
            ax2.legend()
            
            fig.patch.set_facecolor('#0e1117')
            st.pyplot(fig)

            # Baris 4: SENTIMEN BERITA
            st.markdown("### 📰 RADAR SENTIMEN & BERITA")
            news = ticker.news[:5]
            if news:
                s_label, s_desc = dapatkan_sentimen(news)
                st.subheader(s_label)
                st.caption(s_desc)
                for item in news:
                    with st.expander(f"📌 {item['title']}"):
                        st.write(f"Penerbit: {item['publisher']}")
                        st.write(f"[Baca Berita Lengkap]({item['link']})")
            else:
                st.write("Tidak ada berita terbaru untuk kode ini.")

        except Exception as e:
            st.error(f"Gagal memproses {kode_saham}: {e}")

    st.success("🏁 OPERASI PEMINDAIAN SELESAI")
