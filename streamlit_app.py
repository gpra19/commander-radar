import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas_ta as ta # Amunisi baru untuk indikator teknikal masif

# --- KONFIGURASI DASAR ---
st.set_page_config(page_title="The Institutional Commander", layout="wide", initial_sidebar_state="expanded")

# --- INJEKSI CSS KUSTOM (ULTIMATE DARK MODE) ---
st.markdown("""
    <style>
    .stApp { background-color: #0B101E; color: #FFFFFF; }
    div[data-testid="metric-container"] { background-color: #151C2C; border: 1px solid #25324B; padding: 10px 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    div[data-testid="stMetricLabel"] { color: #8B9BB4 !important; font-weight: 600; font-size: 13px; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 20px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .laci-judul { color: #22D3EE; font-size: 18px; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #25324B; padding-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎖️ THE STRATEGIC COMMANDER V7.0")
st.markdown("<p style='color: #8B9BB4;'>Terminal Institusi: Prediksi AI, Analisis Teknikal Lanjut & Jejak Bandar</p>", unsafe_allow_html=True)

# --- SIDEBAR CONTROL ---
st.sidebar.markdown("### 🕹️ COMMAND CENTER")
kode_input = st.sidebar.text_input("🎯 Target Saham (Gunakan Koma):", "BBCA, NISP")
hari_prediksi = st.sidebar.slider("📅 Rentang Proyeksi AI (Hari)", 1, 14, 7)
tombol_pindai = st.sidebar.button("🚀 EKSEKUSI PEMINDAIAN")

# --- FUNGSI BANTUAN FUNDAMENTAL ---
def format_angka(angka, format_tipe="biasa"):
    if angka is None or pd.isna(angka): return "N/A"
    try:
        if format_tipe == "persen": return f"{angka * 100:.2f}%"
        elif format_tipe == "desimal": return f"{angka:.2f}"
        elif format_tipe == "miliar": return f"Rp {angka / 1e9:,.2f} M"
        elif format_tipe == "triliun": return f"Rp {angka / 1e12:,.2f} T"
        else: return f"{angka:,.0f}"
    except: return "N/A"

# --- MAIN LOOP ---
if tombol_pindai:
    daftar_saham = [s.strip().upper() for s in kode_input.split(',') if s.strip()]
    daftar_saham_fixed = [s if s.endswith('.JK') else f"{s}.JK" for s in daftar_saham]
    
    for kode_saham in daftar_saham_fixed:
        st.markdown("<hr style='border: 1px solid #25324B;'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: #E2E8F0;'>🏢 KODE OPERASI: {kode_saham}</h2>", unsafe_allow_html=True)
        
        try:
            ticker = yf.Ticker(kode_saham)
            # Menarik data 1 tahun agar MA200 bisa dihitung
            df = ticker.history(period='1y')
            info = ticker.info 
            
            if df.empty or len(df) < 200:
                st.warning(f"Data {kode_saham} tidak cukup (butuh min 200 hari tempur). Coba emiten lain.")
                continue

            # --- INJEKSI PANDAS-TA (MASSIVE CALCULATION) ---
            # 1. Kategori Tren & Presisi
            df.ta.macd(append=True) # Menghasilkan MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            df.ta.rsi(length=14, append=True)
            df.ta.adx(length=14, append=True) # Menghasilkan ADX_14, DMP_14, DMN_14
            df.ta.atr(length=14, append=True)
            df.ta.willr(length=14, append=True)
            df.ta.stoch(append=True) # Menghasilkan STOCHk_14_3_3, STOCHd_14_3_3
            df.ta.bbands(length=20, append=True) # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
            df.ta.ema(length=10, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)
            df.ta.vwap(append=True)
            
            # Pivot Points (Standard)
            high, low, close = df['High'].iloc[-2], df['Low'].iloc[-2], df['Close'].iloc[-2]
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            
            # 2. Kategori Volume & Institusi
            df.ta.cmf(length=20, append=True)
            df.ta.obv(append=True)
            df.ta.mfi(length=14, append=True)
            
            # Custom Volume Strength & Avg Price 90D (Bandar Modal Proxy VWMA 90)
            avg_vol_20 = df['Volume'].rolling(20).mean()
            vol_strength = df['Volume'].iloc[-1] / avg_vol_20.iloc[-1] if avg_vol_20.iloc[-1] > 0 else 0
            df.ta.vwma(length=90, append=True) # VWMA_90 sebagai estimasi modal bandar 90 hari
            
            # Mengambil baris terakhir untuk di-display
            last = df.iloc[-1]
            harga_skrg = last['Close']

            # --- MODUL AI (REGRESI POLINOMIAL V6) ---
            prices_ai = df['Close'].values[-60:] # AI tetap fokus di 60 hari terakhir
            days_ai = np.arange(len(prices_ai)).reshape(-1, 1)
            weights = np.exp(np.linspace(-2., 0., len(prices_ai)))
            poly = PolynomialFeatures(degree=2)
            days_poly = poly.fit_transform(days_ai)
            
            model = LinearRegression()
            model.fit(days_poly, prices_ai, sample_weight=weights)
            r2_score = model.score(days_poly, prices_ai, sample_weight=weights)
            
            future_days = np.arange(len(prices_ai), len(prices_ai) + hari_prediksi).reshape(-1, 1)
            future_poly = poly.transform(future_days)
            prediction = model.predict(future_poly)
            
            residuals = prices_ai - model.predict(days_poly)
            std_dev_ai = np.std(residuals)
            upper_band = prediction + (1.5 * std_dev_ai)
            lower_band = prediction - (1.5 * std_dev_ai)

            # --- PANEL UTAMA (AI & GRAFIK) ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Harga Terakhir", f"Rp {harga_skrg:,.0f}")
            c2.metric("Trend Besar (MA200)", f"Rp {last.get('SMA_200', 0):,.0f}", "Uptrend" if harga_skrg > last.get('SMA_200', 0) else "Downtrend")
            c3.metric(f"Target Median H+{hari_prediksi}", f"Rp {prediction[-1]:,.0f}")
            c4.metric("AI Confidence (R²)", f"{r2_score:.2f}", "Akurat" if r2_score > 0.7 else "Acak")

            # Grafik Price & Volume
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            ax1.plot(df.index[-60:], prices_ai, label='Harga Aktual', color='#FACC15', linewidth=2)
            
            tgl_pred = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=hari_prediksi)
            ax1.plot(tgl_pred, prediction, color='#22D3EE', linestyle='--', marker='o', label='Proyeksi AI')
            ax1.fill_between(tgl_pred, lower_band, upper_band, color='#22D3EE', alpha=0.1, label='Zona Pendaratan AI')
            
            ax1.set_title(f"Visualisasi Taktis (60 Hari): {kode_saham}", color='#E2E8F0', fontsize=12)
            ax1.legend(facecolor='#151C2C', edgecolor='#25324B', labelcolor='#E2E8F0')
            ax1.set_facecolor('#0B101E')
            ax1.tick_params(colors='#8B9BB4')
            ax1.grid(color='#25324B', linestyle='--', alpha=0.3)

            colors = ['#10B981' if df['Close'].iloc[i] > df['Open'].iloc[i] else '#EF4444' for i in range(len(df)-60, len(df))]
            ax2.bar(df.index[-60:], df['Volume'].tail(60), color=colors, alpha=0.8, label='Volume')
            ax2.set_facecolor('#0B101E')
            ax2.tick_params(colors='#8B9BB4')
            ax2.grid(color='#25324B', linestyle='--', alpha=0.3)
            
            fig.patch.set_facecolor('#0B101E')
            plt.subplots_adjust(hspace=0.05)
            st.pyplot(fig)

            # --- LACI KOMPARTEMEN RAHASIA ---
            st.markdown("### 🗄️ PANEL INDIKATOR SPESIFIK")
            
            # LACI 1: TEKNIKAL
            with st.expander("📈 1. Presisi Teknikal & Arah Tren"):
                st.markdown("<div class='laci-judul'>Oscillator & Momentum</div>", unsafe_allow_html=True)
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("RSI (14)", format_angka(last.get('RSI_14'), "desimal"))
                t2.metric("Stochastic %K", format_angka(last.get('STOCHk_14_3_3'), "desimal"))
                t3.metric("Williams %R", format_angka(last.get('WILLR_14'), "desimal"))
                t4.metric("ADX (14)", format_angka(last.get('ADX_14'), "desimal"))
                
                st.markdown("<div class='laci-judul'>Moving Averages & Bands</div>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("EMA 10", f"Rp {last.get('EMA_10', 0):,.0f}")
                m2.metric("MA 20", f"Rp {last.get('SMA_20', 0):,.0f}")
                m3.metric("MA 50", f"Rp {last.get('SMA_50', 0):,.0f}")
                m4.metric("VWAP", f"Rp {last.get('VWAP_D', 0):,.0f}")
                
                st.markdown("<div class='laci-judul'>Volatilitas & Sinyal</div>", unsafe_allow_html=True)
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("ATR (14)", f"Rp {last.get('ATRr_14', 0):,.0f}")
                v2.metric("MACD Line", format_angka(last.get('MACD_12_26_9'), "desimal"))
                v3.metric("Signal Line", format_angka(last.get('MACDs_12_26_9'), "desimal"))
                v4.metric("Pivot (Std)", f"Rp {pivot:,.0f}")

            # LACI 2: VOLUME & INSTITUSI
            with st.expander("🐋 2. Kekuatan Volume & Arus Uang Institusi"):
                st.markdown("<div class='laci-judul'>Metrik Akumulasi Bandar</div>", unsafe_allow_html=True)
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Volume Strength (Vol.R)", f"{vol_strength:.2f}x")
                b2.metric("MFI (14) - Money Flow", format_angka(last.get('MFI_14'), "desimal"))
                b3.metric("CMF (20)", format_angka(last.get('CMF_20'), "desimal"))
                b4.metric("Est. Bandar Modal (VWMA 90)", f"Rp {last.get('VWMA_90', harga_skrg):,.0f}")

            # LACI 3: FUNDAMENTAL
            with st.expander("🏢 3. Laporan Fundamental & Aksi Korporasi"):
                st.markdown("<div class='laci-judul'>Valuasi & Profitabilitas</div>", unsafe_allow_html=True)
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Market Cap", format_angka(info.get('marketCap'), "triliun"))
                f2.metric("PER (Price to Earning)", format_angka(info.get('trailingPE'), "desimal") + "x")
                f3.metric("PBV (Price to Book)", format_angka(info.get('priceToBook'), "desimal") + "x")
                f4.metric("ROE", format_angka(info.get('returnOnEquity'), "persen"))
                
                st.markdown("<div class='laci-judul'>Dividen & Kesehatan Keuangan</div>", unsafe_allow_html=True)
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("EPS (Earning Per Share)", f"Rp {info.get('trailingEps', 0):,.0f}")
                d2.metric("Dividend Yield", format_angka(info.get('dividendYield'), "persen"))
                d3.metric("Payout Ratio", format_angka(info.get('payoutRatio'), "persen"))
                d4.metric("Debt to Equity", format_angka(info.get('debtToEquity'), "desimal") + "%")

        except Exception as e:
            st.error(f"⚠️ Terjadi gangguan transmisi pada {kode_saham}: {e}")

    st.success("🏁 OPERASI V7.0 SELESAI DENGAN PRESISI INSTITUSIONAL")
