import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="COMMANDER RADAR V8.1", layout="wide")
st.title("🎖️ THE COMMANDER - RADAR V8.1")
st.markdown("*Pusat Intelijen Taktis: Integrasi Prediksi Harga & Manajemen Risiko Berbasis ATR*")

# --- SIDEBAR INPUT ---
st.sidebar.header("📡 PUSAT KOMANDO")
ticker = st.sidebar.text_input("Kode Saham (Contoh: KLBF.JK)", value="KLBF.JK")
harga_beli = st.sidebar.number_input("Harga Beli Jenderal (Entry)", value=0.0)
pengali_atr = st.sidebar.slider("Multiplier Toleransi (ATR Multiplier)", 1.0, 3.0, 1.5, 0.5)

if st.sidebar.button("EKSEKUSI PEMINDAIAN"):
    # Membuka blok pelindung (try)
    try:
        with st.spinner('Memindai medan tempur...'):
            # 1. AMBIL DATA
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            
            if data.empty:
                st.error("Data tidak ditemukan. Pastikan ticker menggunakan akhiran .JK (Contoh: MAPI.JK)")
            else:
                # Perbaikan index kolom dari yfinance terbaru
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                # --- BAGIAN A: PERHITUNGAN ATR (VOLATILITAS ASLI) ---
                df = data.copy()
                df['H-L'] = df['High'] - df['Low']
                df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
                df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
                df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
                # Menghitung Rata-rata pergerakan ATR selama 14 hari
                df['ATR'] = df['TR'].rolling(window=14).mean()
                
                atr_terkini = df['ATR'].iloc[-1]
                jarak_toleransi = atr_terkini * pengali_atr
                rekomendasi_sl = harga_beli - jarak_toleransi if harga_beli > 0 else 0
                
                # --- BAGIAN B: PREDIKSI REGRESI LINIER (7 HARI KEDEPAN) ---
                df_pred = data.reset_index()
                df_pred['Date_Ordinal'] = df_pred['Date'].map(datetime.toordinal)
                X = df_pred[['Date_Ordinal']].values
                y = df_pred['Close'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                last_date = df_pred['Date'].max()
                tgl_pred = [last_date + timedelta(days=i) for i in range(1, 8)]
                future_ordinals = np.array([d.toordinal() for d in tgl_pred]).reshape(-1, 1)
                predictions = model.predict(future_ordinals)

                # --- BAGIAN C: VISUALISASI MEDAN TEMPUR (MATPLOTLIB) ---
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot Harga Historis (Ambil 30 hari terakhir agar grafik lebih zoom dan proporsional)
                hist_plot = df_pred.tail(30)
                ax.plot(hist_plot['Date'], hist_plot['Close'], label='Harga Historis (Close)', color='white', linewidth=2)
                
                # Plot Garis Prediksi Regresi
                ax.plot(tgl_pred, predictions, label='Proyeksi Regresi (H+7)', color='#F59E0B', linestyle='--', linewidth=2)
                
                # Plot Batas Toleransi ATR (Ini adalah baris yang Jenderal kirim sebelumnya)
                upper_band = predictions + jarak_toleransi
                lower_band = predictions - jarak_toleransi
                ax.fill_between(tgl_pred, lower_band, upper_band, color='#22D3EE', alpha=0.15, label='Batas Toleransi ATR (Aman)')
                
                # Menambahkan Garis Taktis jika Jenderal memasukkan Harga Beli
                if harga_beli > 0:
                    ax.axhline(y=harga_beli, color='#10B981', linestyle=':', linewidth=2, label='Titik Pendaratan (Harga Beli)')
                    ax.axhline(y=rekomendasi_sl, color='#EF4444', linestyle='-', linewidth=2, label='Tembok Evakuasi (Stop Loss Dinamis)')

                # Styling Grafik (Tema Gelap ala Dasbor Radar)
                ax.set_title(f"Peta Proyeksi Operasi: {ticker}", color='white', fontsize=14, fontweight='bold')
                ax.set_xlabel("Tanggal", color='gray')
                ax.set_ylabel("Harga (Rp)", color='gray')
                fig.patch.set_facecolor('#0E1117')
                ax.set_facecolor('#0E1117')
                ax.tick_params(colors='gray')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#4B5563')
                ax.legend(facecolor='#1F2937', edgecolor='#4B5563', labelcolor='white')
                
                # Tampilkan Grafik di Streamlit
                st.pyplot(fig)
                
                # --- BAGIAN D: DISPLAY HASIL METRIK ---
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🛡️ The Guardian (Analisis Risiko)")
                    st.metric("Amplitudo Harian (ATR 14)", f"Rp {atr_terkini:.2f}")
                    st.metric(f"Toleransi Guncangan ({pengali_atr}x)", f"Rp {jarak_toleransi:.2f}")
                    if harga_beli > 0:
                        st.error(f"🚨 Batas Evakuasi (Stop Loss): Rp {rekomendasi_sl:.0f}")
                        st.info(f"Persentase Risiko Jenderal: {((jarak_toleransi/harga_beli)*100):.2f}%")
                    else:
                        st.info("💡 Masukkan Harga Beli di sidebar untuk menghitung Stop Loss Dinamis.")

                with col2:
                    st.subheader("🔮 Vanguard AI (Prediksi H+7)")
                    pred_akhir = predictions[-1]
                    harga_terakhir = df['Close'].iloc[-1]
                    perubahan_pred = ((pred_akhir - harga_terakhir) / harga_terakhir) * 100
                    st.metric("Proyeksi Harga Hari Ke-7", f"Rp {pred_akhir:.2f}", f"{perubahan_pred:.2f}% dari harga saat ini")
                    
                    # Buat DataFrame untuk tabel prediksi
                    res_pred = pd.DataFrame({
                        'Tanggal': [d.strftime("%Y-%m-%d") for d in tgl_pred], 
                        'Prediksi Harga (Rp)': np.round(predictions, 2)
                    })
                    st.dataframe(res_pred, hide_index=True, use_container_width=True)

                st.success(f"Pemindaian Selesai untuk {ticker}! 🏰⚙️")

    # Ini adalah blok penutup wajib dari perintah 'try' di atas
    except Exception as e:
        st.error(f"Terjadi kesalahan sistem saat memproses data: {e}")
        st.warning("Laporkan kepada Teknisi: Periksa struktur data atau pastikan kode saham valid.")
