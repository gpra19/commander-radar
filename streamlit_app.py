import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # Amunisi baru untuk kurva melengkung

# Mengatur tampilan layar menjadi layar penuh (wide)
st.set_page_config(page_title="Radar The Commander", layout="wide")

st.title("🎖️ THE COMMANDER - Radar Taktikal H+7 (Advanced AI)")
st.markdown("Memindai probabilitas harga dengan **Regresi Polinomial, Bobot Waktu, & Zona Volatilitas**.")

# Layar Interaksi Multi-Target
kode_input = st.text_input("🎯 Kode Target (Pisahkan dengan koma. Misal: BBCA, TINS, INDF):", "BBCA, NISP, TINS")

if st.button("🚀 Pindai Target"):
    daftar_saham = [s.strip().upper() for s in kode_input.split(',') if s.strip()]
    daftar_saham_fixed = [s if s.endswith('.JK') else f"{s}.JK" for s in daftar_saham]
    
    for kode_saham in daftar_saham_fixed:
        st.markdown("---") 
        st.info(f"📡 Mengunci koordinat satelit untuk {kode_saham}...")
        
        try:
            ticker = yf.Ticker(kode_saham)
            df = ticker.history(period='60d')
            
            if df.empty:
                st.error(f"❌ Gagal: Data riwayat untuk {kode_saham} tidak terdeteksi.")
                continue
                
            # --- MESIN ALGORITMA V3.0 ---
            prices = df['Close'].values
            days = np.arange(len(prices)).reshape(-1, 1)
            
            # 1. Pembobotan Waktu (Data terbaru bobotnya lebih besar secara eksponensial)
            weights = np.exp(np.linspace(-2., 0., len(prices))) 
            
            # 2. Regresi Polinomial (Derajat 2 agar garis bisa melengkung membaca momentum)
            poly = PolynomialFeatures(degree=2)
            days_poly = poly.fit_transform(days)
            
            # Melatih Model dengan bobot
            model = LinearRegression()
            model.fit(days_poly, prices, sample_weight=weights)
            
            # Prediksi Masa Lalu (untuk menghitung tingkat akurasi/error)
            pred_hist = model.predict(days_poly)
            
            # 3. Menghitung Zona Volatilitas (Standard Deviation dari Error Historis)
            residuals = prices - pred_hist
            std_dev = np.std(residuals)
            
            # Ekstrapolasi H+7
            future_days = np.arange(len(prices), len(prices) + 7).reshape(-1, 1)
            future_days_poly = poly.transform(future_days)
            pred_7d = model.predict(future_days_poly)
            
            # Kalkulasi Batas Atas (Bullish) & Bawah (Bearish) - Menggunakan 1.5 Standar Deviasi
            upper_bound = pred_7d + (1.5 * std_dev)
            lower_bound = pred_7d - (1.5 * std_dev)
            
            # --- VARIABEL LAPORAN ---
            harga_skrg = prices[-1]
            target_median = pred_7d[-1]
            target_bullish = upper_bound[-1]
            target_bearish = lower_bound[-1]
            
            perubahan_median = ((target_median - harga_skrg) / harga_skrg) * 100
            
            # --- DASHBOARD METRIK ---
            st.markdown(f"### 📊 Laporan Intelijen: **{kode_saham}**")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Harga Terakhir", f"Rp {harga_skrg:,.0f}")
            col2.metric("Skenario Bullish", f"Rp {target_bullish:,.0f}", "+ (Batas Atas)")
            col3.metric("Target Median H+7", f"Rp {target_median:,.0f}", f"{perubahan_median:+.2f}%")
            col4.metric("Skenario Bearish", f"Rp {target_bearish:,.0f}", "- (Batas Bawah)")
            
            # --- VISUALISASI RADAR ---
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot Historis
            ax.plot(df.index, prices, label=f'Harga Aktual', color='gold', linewidth=2)
            
            # Plot Kurva Fitting Historis
            ax.plot(df.index, pred_hist, color='white', linestyle=':', alpha=0.5, label='Garis Momentum')
            
            # Plot Ekstrapolasi Masa Depan
            tgl_pred = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)
            ax.plot(tgl_pred, pred_7d, color='cyan', linestyle='--', marker='o', label='Proyeksi Median H+7')
            
            # Plot Zona Pendaratan (Awan Probabilitas)
            ax.fill_between(tgl_pred, lower_bound, upper_bound, color='cyan', alpha=0.15, label='Zona Volatilitas (1.5 SD)')
            
            # Estetika Militer
            ax.set_title(f'Proyeksi Momentum Polinomial {kode_saham}', fontsize=14, color='white')
            ax.set_ylabel('Harga (Rp)', color='white')
            ax.tick_params(colors='white')
            fig.patch.set_facecolor('#0e1117') 
            ax.set_facecolor('#0e1117')
            ax.legend()
            ax.grid(True, alpha=0.2)
            
            st.pyplot(fig)
                
        except Exception as e:
            st.error(f"⚠️ Terjadi gangguan transmisi pada {kode_saham}: {e}")

    st.success("🏁 Seluruh target telah selesai dipindai!")
