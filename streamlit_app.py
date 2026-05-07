import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Mengatur tampilan layar menjadi layar penuh (wide)
st.set_page_config(page_title="Radar The Commander", layout="wide")

st.title("🎖️ THE COMMANDER - Radar Prediksi H+7")
st.markdown("Masukkan kode target untuk memindai probabilitas pergerakan arah harga dalam 7 hari ke depan.")

# Layar Interaksi: Kolom input target sasaran
kode_input = st.text_input("🎯 Kode Target (Misal: BBCA, TINS, INDF):", "BBCA")

# Tombol Pelatuk
if st.button("🚀 Pindai Target"):
    # Membersihkan input dan otomatis menambahkan .JK jika belum ada
    kode_saham = kode_input.strip().upper()
    if not kode_saham.endswith('.JK'):
        kode_saham += '.JK'
        
    st.info(f"📡 Mengunci koordinat satelit untuk {kode_saham}...")
    
    try:
        # Menarik logistik data
        ticker = yf.Ticker(kode_saham)
        df = ticker.history(period='60d')
        
        if df.empty:
            st.error(f"❌ Gagal: Data riwayat untuk {kode_saham} tidak terdeteksi di medan tempur.")
        else:
            # Reaktor Kalkulasi: Regresi Linear
            prices = df['Close'].values
            days = np.arange(len(prices)).reshape(-1, 1)
            model = LinearRegression().fit(days, prices)
            r2 = model.score(days, prices)
            
            # Ekstrapolasi H+7
            future_days = np.arange(len(prices), len(prices) + 7).reshape(-1, 1)
            pred_7d = model.predict(future_days)
            
            harga_skrg = prices[-1]
            harga_nanti = pred_7d[-1]
            perubahan = ((harga_nanti - harga_skrg) / harga_skrg) * 100
            
            # Membangun Dashboard Metrik
            st.markdown("### 📊 Status Operasi")
            col1, col2, col3 = st.columns(3)
            col1.metric("Harga Terakhir", f"Rp {harga_skrg:,.0f}")
            col2.metric("Target H+7", f"Rp {harga_nanti:,.0f}", f"{perubahan:+.2f}%")
            col3.metric("Kekuatan Tren (R²)", f"{r2:.4f}")
            
            # Membangun Visualisasi Layar Radar
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, prices, label=f'Historis {kode_saham}', color='gold', linewidth=2)
            
            tgl_pred = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)
            ax.plot(tgl_pred, pred_7d, color='cyan', linestyle='--', marker='o', label='Proyeksi H+7')
            
            ax.set_title(f'Proyeksi Tren {kode_saham} (R²: {r2:.4f})', fontsize=14, color='white')
            ax.set_ylabel('Harga (Rp)', color='white')
            ax.tick_params(colors='white')
            fig.patch.set_facecolor('#0e1117') # Warna latar belakang gelap khas militer
            ax.set_facecolor('#0e1117')
            ax.legend()
            ax.grid(True, alpha=0.2)
            
            # Menampilkan grafik ke Streamlit
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"⚠️ Terjadi gangguan transmisi: {e}")
