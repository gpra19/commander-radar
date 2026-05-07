import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

# --- KONFIGURASI DASAR ---
st.set_page_config(page_title="The Strategic Commander", layout="wide", initial_sidebar_state="expanded")

# --- INJEKSI CSS KUSTOM ---
st.markdown("""
    <style>
    .stApp { background-color: #0B101E; color: #FFFFFF; }
    div[data-testid="metric-container"] { background-color: #151C2C; border: 1px solid #25324B; padding: 15px 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    div[data-testid="stMetricLabel"] { color: #8B9BB4 !important; font-weight: 600; font-size: 14px; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 28px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    /* Warna Tab/Radio Button */
    div.row-widget.stRadio > div{flex-direction:column;}
    </style>
    """, unsafe_allow_html=True)

# --- INIT DATABASE SEMENTARA (SESSION STATE) ---
if 'jurnal_df' not in st.session_state:
    st.session_state.jurnal_df = pd.DataFrame(columns=['Tanggal', 'Kode Saham', 'Strategi', 'Status', 'Profit/Loss (Rp)'])

# --- FUNGSI BANTUAN ---
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
    elif score < 0: return "🔴 BEARISH SENTIMENT", "Waspada sentimen negatif."
    return "⚪ NEUTRAL SENTIMENT", "Tidak ada berita mencolok."

# --- SISTEM NAVIGASI SIDEBAR ---
st.sidebar.image("https://img.icons8.com/color/96/000000/military-base.png", width=80)
st.sidebar.markdown("### 🖥️ PUSAT KENDALI")
menu_pilihan = st.sidebar.radio("PILIH DIVISI OPERASI:", ["📡 Radar Makro", "📒 Jurnal & Logistik"])
st.sidebar.markdown("---")

# ==========================================
# LAYAR 1: RADAR MAKRO (V4.2)
# ==========================================
if menu_pilihan == "📡 Radar Makro":
    st.title("📡 RADAR MAKRO TAKTIS")
    st.markdown("<p style='color: #8B9BB4;'>Pemindai AI, Momentum Harga & Intelijen Pasar</p>", unsafe_allow_html=True)
    
    kode_input = st.sidebar.text_input("🎯 Target Saham (Gunakan Koma):", "BBCA, NISP, TINS")
    hari_prediksi = st.sidebar.slider("📅 Rentang Proyeksi AI (Hari)", 1, 14, 7)
    tombol_pindai = st.sidebar.button("🚀 EKSEKUSI PEMINDAIAN")

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

                df['MA20'] = df['Close'].rolling(20).mean()
                df['RSI'] = hitung_rsi(df['Close'])
                rsi_now = df['RSI'].iloc[-1]
                
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

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Harga Terakhir", f"Rp {prices[-1]:,.0f}")
                c2.metric("Indikator RSI (14)", f"{rsi_now:.2f}", "Overbought" if rsi_now > 70 else "Oversold" if rsi_now < 30 else "Normal")
                c3.metric("Target Median", f"Rp {prediction[-1]:,.0f}")
                c4.metric(f"Probabilitas H+{hari_prediksi}", f"{((prediction[-1]-prices[-1])/prices[-1]*100):+.2f}%")

                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(df.index[-60:], prices, label='Harga Aktual', color='#FACC15', linewidth=2)
                ax1.plot(df.index[-60:], df['MA20'].tail(60), label='MA20 Trend', color='#FB923C', alpha=0.6)
                
                tgl_pred = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=hari_prediksi)
                ax1.plot(tgl_pred, prediction, color='#22D3EE', linestyle='--', marker='o', label='Proyeksi AI')
                ax1.fill_between(tgl_pred, lower_band, upper_band, color='#22D3EE', alpha=0.1, label='Zona Volatilitas')
                
                ax1.set_title(f"Visualisasi Historis & Proyeksi AI: {kode_saham}", color='#E2E8F0', fontsize=14)
                ax1.legend(facecolor='#151C2C', edgecolor='#25324B', labelcolor='#E2E8F0')
                ax1.set_facecolor('#0B101E')
                ax1.tick_params(colors='#8B9BB4')
                ax1.grid(color='#25324B', linestyle='--', alpha=0.5)
                fig.patch.set_facecolor('#0B101E')
                st.pyplot(fig)

                st.markdown("<h4 style='color: #8B9BB4; margin-top: 20px;'>📰 RADAR SENTIMEN & BERITA</h4>", unsafe_allow_html=True)
                news = ticker.news[:5]
                if news:
                    s_label, s_desc = dapatkan_sentimen(news)
                    st.write(f"**{s_label}** - {s_desc}")
                else:
                    st.write("Tidak ada pantauan berita terbaru.")

            except Exception as e:
                st.error(f"Gagal memproses {kode_saham}: {e}")

# ==========================================
# LAYAR 2: JURNAL & LOGISTIK (FITUR BARU)
# ==========================================
elif menu_pilihan == "📒 Jurnal & Logistik":
    st.title("📒 JURNAL TEMPUR & LOGISTIK")
    st.markdown("<p style='color: #8B9BB4;'>Pusat Pencatatan Kinerja Strategi dan Manajemen Amunisi RDN</p>", unsafe_allow_html=True)
    
    # Kalkulasi Metrik dari Session State
    df_jurnal = st.session_state.jurnal_df
    total_trade = len(df_jurnal)
    total_win = len(df_jurnal[df_jurnal['Profit/Loss (Rp)'] > 0])
    win_rate = (total_win / total_trade * 100) if total_trade > 0 else 0
    total_pnl = df_jurnal['Profit/Loss (Rp)'].sum()
    
    # Baris Metrik Kinerja Pasukan
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Operasi", f"{total_trade} Transaksi")
    m2.metric("Win Rate", f"{win_rate:.1f}%")
    m3.metric("Net Profit / Loss", f"Rp {total_pnl:,.0f}", f"{'Surplus' if total_pnl > 0 else 'Defisit' if total_pnl < 0 else ''}")
    m4.metric("Kesehatan Amunisi", "Aman", "Optimal")
    
    st.markdown("---")
    
    # Layout Kolom untuk Form Input dan Tabel
    col_kiri, col_kanan = st.columns([1, 2])
    
    with col_kiri:
        st.markdown("### 📝 Input Laporan Baru")
        with st.form("form_jurnal", clear_on_submit=True):
            tgl = st.date_input("Tanggal Operasi", datetime.today())
            kode = st.text_input("Kode Saham (Misal: BBCA)").upper()
            strategi = st.selectbox("Strategi Sandi", ["The Commander - Full Assault", "The Commander - Bom Waktu", "Scalping Kilat", "Buy On Weakness", "Lainnya"])
            status = st.radio("Status Posisi", ["Selesai (Closed)", "Masih Berjalan (Floating)"], horizontal=True)
            pnl = st.number_input("Hasil Profit/Loss (Rp) - Beri tanda minus (-) jika rugi", value=0, step=10000)
            
            submit = st.form_submit_button("💾 Simpan Catatan")
            
            if submit and kode:
                # Membuat data baru
                data_baru = pd.DataFrame({
                    'Tanggal': [tgl],
                    'Kode Saham': [kode],
                    'Strategi': [strategi],
                    'Status': [status],
                    'Profit/Loss (Rp)': [pnl]
                })
                # Menyimpan ke memori
                st.session_state.jurnal_df = pd.concat([st.session_state.jurnal_df, data_baru], ignore_index=True)
                st.success("Laporan berhasil diamankan di brankas Markas!")
                st.rerun() # Memuat ulang layar agar tabel & metrik ter-update
                
    with col_kanan:
        st.markdown("### 🗄️ Arsip Rekam Jejak (Buku Besar)")
        if df_jurnal.empty:
            st.info("Belum ada catatan tempur. Silakan laporkan operasi pertama Jenderal di panel kiri.")
        else:
            # Menampilkan tabel dengan gaya Streamlit
            st.dataframe(
                df_jurnal.style.format({'Profit/Loss (Rp)': "{:,.0f}"}),
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("🗑️ Hapus Semua Arsip Sementara"):
                st.session_state.jurnal_df = pd.DataFrame(columns=['Tanggal', 'Kode Saham', 'Strategi', 'Status', 'Profit/Loss (Rp)'])
                st.rerun()
