import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- KONFIGURASI DASAR ---
st.set_page_config(page_title="The Strategic Commander", layout="wide", initial_sidebar_state="expanded")

# --- INJEKSI CSS KUSTOM (UI/UX UPGRADE) ---
st.markdown("""
    <style>
    /* Latar belakang utama aplikasi */
    .stApp { background-color: #0B101E; color: #FFFFFF; }
    
    /* Mempercantik kotak metrik bawaan */
    div[data-testid="metric-container"] {
        background-color: #151C2C;
        border: 1px solid #25324B;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Warna teks label metrik (abu-abu kebiruan) */
    div[data-testid="stMetricLabel"] { color: #8B9BB4 !important; font-weight: 600; font-size: 14px; }
    
    /* Warna teks nilai metrik (putih terang) */
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 28px; }
    
    /* Menyembunyikan tombol menu bawaan Streamlit agar lebih bersih */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Fungsi untuk membuat Kartu HTML Kustom
def buat_kartu(title, value, border_color):
    return f"""
    <div style="background-color: #151C2C; border-left: 5px solid {border_color}; padding: 15px 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); margin-bottom: 20px;">
        <p style="margin: 0; font-size: 13px; color: #8B9BB4; font-weight: 700; text-transform: uppercase;">{title}</p>
        <h3 style="margin: 5px 0 0 0; color: #FFFFFF; font-size: 24px;">{value}</h3>
    </div>
    """

st.title("🎖️ THE STRATEGIC COMMANDER V4.1")
st.markdown("<p style='color: #8B9BB4;'>Integrated Tactical Dashboard: AI Prediction + Technicals + Sentiment</p>", unsafe_allow_html=True)

# --- SIDEBAR INPUT ---
st.sidebar.markdown("### 🕹️ CONTROL PANEL")
kode_input = st.sidebar.text_input("🎯 Target Saham (Gunakan Koma):", "BBCA, INTP, TINS")
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
    pos_words = ['profit', 'up', 'bull', 'buy', 'gain', 'growth', 'laba', 'naik', 'beli', 'positif']
    neg_words = ['loss', 'down', 'bear', 'sell', 'drop', 'cut', 'rugi', 'turun', 'jual', 'perang', 'negatif']
    score = 0
    for news in news_list:
        text = news['title'].lower()
        for w in pos_words: score += 1 if w in text else 0
        for w in neg_words: score -= 1 if w in text else 0
    
    if score > 0: return "🟢 BULLISH SENTIMENT", "Sentimen berita cenderung positif."
    elif score < 0: return "🔴 BEARISH SENTIMENT", "Waspada sentimen negatif di media."
    return "⚪ NEUTRAL SENTIMENT", "Tidak ada berita mencolok hari ini."

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
            
            if df.empty:
                st.error(f"Data {kode_saham} tidak ditemukan.")
                continue

            # Kalkulasi Indikator
            df['MA20'] = df['Close'].rolling(20).mean()
            df['RSI'] = hitung_rsi(df['Close'])
            rsi_now = df['RSI'].iloc[-1]
            
            # Kalkulasi AI
            prices = df['Close'].values[-60:]
            days = np.arange(len(prices)).reshape(-1, 1)
            weights = np.exp(np.linspace(-2., 0., len(prices)))
            poly = PolynomialFeatures(degree=2)
            days_poly = poly.fit_transform(days)
            
            model = LinearRegression()
            model.fit(days_poly, prices, sample_weight=weights)
            
            future_days = np.arange(len(prices), len(prices) + hari_prediksi).reshape(-1, 1)
            future_poly = poly.transform(future_days)
            prediction = model.predict(future_poly)
            
            residuals = prices - model.predict(days_poly)
            std_dev = np.std(residuals)
            upper_band = prediction + (1.5 * std_dev)
            lower_band = prediction - (1.5 * std_dev)

            # --- DISPLAY BARIS 1: METRIK ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Harga Terakhir", f"Rp {prices[-1]:,.0f}")
            c2.metric("RSI (14)", f"{rsi_now:.2f}", "Overbought" if rsi_now > 70 else "Oversold" if rsi_now < 30 else "Normal")
            c3.metric("Target Median", f"Rp {prediction[-1]:,.0f}")
            c4.metric(f"Probabilitas H+{hari_prediksi}", f"{((prediction[-1]-prices[-1])/prices[-1]*100):+.2f}%")

            # --- DISPLAY BARIS 2: TRADING PLAN ---
            st.markdown("<h4 style='color: #8B9BB4; margin-top: 20px;'>📝 RENCANA OPERASI (TRADING PLAN)</h4>", unsafe_allow_html=True)
            
            entry_price = prices[-1]
            tp_price = upper_band[-1]
            sl_price = lower_band[-1]
            risk = entry_price - sl_price
            reward = tp_price - entry_price
            rr_ratio = reward / risk if risk > 0 else 0
            
            t1, t2, t3, t4 = st.columns(4)
            with t1: st.markdown(buat_kartu("🎯 AREA ENTRY", f"Rp {entry_price:,.0f}", "#3B82F6"), unsafe_allow_html=True)
            with t2: st.markdown(buat_kartu("✅ TAKE PROFIT", f"Rp {tp_price:,.0f}", "#10B981"), unsafe_allow_html=True)
            with t3: st.markdown(buat_kartu("🛑 STOP LOSS", f"Rp {sl_price:,.0f}", "#EF4444"), unsafe_allow_html=True)
            with t4: st.markdown(buat_kartu("⚖️ RISK / REWARD", f"{rr_ratio:.2f}x", "#F59E0B"), unsafe_allow_html=True)

            # --- DISPLAY BARIS 3: GRAFIK ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot 1: Harga
            ax1.plot(df.index[-60:], prices, label='Harga Aktual', color='#FACC15', linewidth=2)
            ax1.plot(df.index[-60:], df['MA20'].tail(60), label='MA20 Trend', color='#FB923C', alpha=0.6)
            
            tgl_pred = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=hari_prediksi)
            ax1.plot(tgl_pred, prediction, color='#22D3EE', linestyle='--', marker='o', label='Proyeksi AI')
            ax1.fill_between(tgl_pred, lower_band, upper_band, color='#22D3EE', alpha=0.1, label='Zona Volatilitas')
            
            # Kosmetik Plot
            ax1.set_title(f"Analisis Teknikal & AI: {kode_saham}", color='#E2E8F0', fontsize=14)
            ax1.legend(facecolor='#151C2C', edgecolor='#25324B', labelcolor='#E2E8F0')
            ax1.set_facecolor('#0B101E')
            ax1.tick_params(colors='#8B9BB4')
            ax1.grid(color='#25324B', linestyle='--', alpha=0.5)

            # Plot 2: RSI
            ax2.plot(df.index[-60:], df['RSI'].tail(60), color='#D946EF', linewidth=2, label='RSI')
            ax2.axhline(70, color='#EF4444', linestyle='--', alpha=0.5)
            ax2.axhline(30, color='#10B981', linestyle='--', alpha=0.5)
            ax2.set_ylim(0, 100)
            ax2.set_facecolor('#0B101E')
            ax2.tick_params(colors='#8B9BB4')
            ax2.grid(color='#25324B', linestyle='--', alpha=0.5)
            ax2.legend(facecolor='#151C2C', edgecolor='#25324B', labelcolor='#E2E8F0')
            
            fig.patch.set_facecolor('#0B101E')
            st.pyplot(fig)

            # --- DISPLAY BARIS 4: SENTIMEN ---
            st.markdown("<h4 style='color: #8B9BB4;'>📰 RADAR SENTIMEN & BERITA</h4>", unsafe_allow_html=True)
            news = ticker.news[:5]
            if news:
                s_label, s_desc = dapatkan_sentimen(news)
                st.write(f"**{s_label}** - {s_desc}")
                for item in news:
                    with st.expander(f"📌 {item['title']}"):
                        st.write(f"Penerbit: {item['publisher']}")
                        st.write(f"[Baca Berita Lengkap]({item['link']})")
            else:
                st.write("Tidak ada pantauan berita terbaru.")

        except Exception as e:
            st.error(f"Gagal memproses {kode_saham}: {e}")

    st.success("🏁 OPERASI PEMINDAIAN SELESAI")
