import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- KONFIGURASI HALAMAN (WIDE MODE) ---
st.set_page_config(page_title="Radar Analisis Saham", layout="wide")

st.title("Radar Volatilitas & Prediksi Harga")

# --- SIDEBAR INPUT ---
st.sidebar.header("Parameter Input")

# 1. Input Ticker Kosong & Tanpa .JK otomatis
ticker_input = st.sidebar.text_input("Kode Saham", value=None, placeholder="Contoh: BBCA")

# 2. Input Harga Kosong (Tidak ada 0.00)
harga_entry = st.sidebar.number_input("Harga Entry (Opsional)", value=None, placeholder="Masukkan harga beli")

pengali_atr = st.sidebar.slider("Pengali ATR (Toleransi)", 1.0, 3.0, 1.5, 0.5)

if st.sidebar.button("Proses Analisis"):
    if not ticker_input:
        st.warning("Silakan masukkan kode saham terlebih dahulu.")
    else:
        try:
            with st.spinner('Memproses data...'):
                # Penambahan .JK di belakang layar
                ticker_yf = ticker_input.strip().upper()
                if not ticker_yf.endswith(".JK"):
                    ticker_yf += ".JK"
                
                # Ambil Data
                data = yf.download(ticker_yf, period="3mo", interval="1d", progress=False)
                
                if data.empty:
                    st.error("Data tidak ditemukan. Pastikan kode saham benar.")
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    # --- PERHITUNGAN ATR ---
                    df = data.copy()
                    df['H-L'] = df['High'] - df['Low']
                    df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
                    df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
                    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
                    df['ATR'] = df['TR'].rolling(window=14).mean()
                    
                    atr_terkini = df['ATR'].iloc[-1]
                    jarak_toleransi = atr_terkini * pengali_atr
                    
                    # Konversi harga entry jika diisi
                    harga_beli = float(harga_entry) if harga_entry else 0.0
                    rekomendasi_sl = harga_beli - jarak_toleransi if harga_beli > 0 else 0
                    
                    # --- PREDIKSI REGRESI LINIER ---
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
                    
                    # --- LAYOUT HASIL (COLUMNS) ---
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Analisis Volatilitas (ATR 14)")
                        st.metric("Nilai ATR Terkini", f"Rp {atr_terkini:.2f}")
                        st.metric(f"Toleransi Guncangan ({pengali_atr}x)", f"Rp {jarak_toleransi:.2f}")
                        if harga_beli > 0:
                            st.error(f"Stop Loss Disarankan: Rp {rekomendasi_sl:.0f}")
                    
                    with col2:
                        st.subheader("Prediksi Harga (Regresi Linier)")
                        # 3. Harga 7 Hari cukup 1 nilai (Tanpa Tabel)
                        pred_akhir = predictions[-1]
                        harga_terakhir = df['Close'].iloc[-1]
                        perubahan_pred = ((pred_akhir - harga_terakhir) / harga_terakhir) * 100
                        st.metric("Proyeksi H+7", f"Rp {pred_akhir:.2f}", f"{perubahan_pred:.2f}%")

                    # --- VISUALISASI PLOTLY & TABS ---
                    st.markdown("<br>", unsafe_allow_html=True)
                    tab1, tab2 = st.tabs(["📊 Grafik Interaktif", "📝 Data Historis Mentah"])
                    
                    with tab1:
                        fig = go.Figure()
                        
                        # Data Historis
                        hist_plot = df_pred.tail(30)
                        fig.add_trace(go.Scatter(x=hist_plot['Date'], y=hist_plot['Close'], 
                                                 mode='lines', name='Harga Historis', 
                                                 line=dict(color='white', width=2)))
                        
                        # Proyeksi
                        fig.add_trace(go.Scatter(x=tgl_pred, y=predictions, 
                                                 mode='lines', name='Proyeksi (H+7)', 
                                                 line=dict(color='#F59E0B', width=2, dash='dash')))
                        
                        # Batas Atas & Bawah ATR (Shaded Area)
                        upper_band = predictions + jarak_toleransi
                        lower_band = predictions - jarak_toleransi
                        
                        fig.add_trace(go.Scatter(x=tgl_pred, y=upper_band, mode='lines', 
                                                 line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig.add_trace(go.Scatter(x=tgl_pred, y=lower_band, mode='lines', 
                                                 fill='tonexty', fillcolor='rgba(34, 211, 238, 0.15)', 
                                                 line=dict(width=0), name='Area Toleransi (ATR)'))
                        
                        # Garis Entry & SL
                        if harga_beli > 0:
                            fig.add_hline(y=harga_beli, line_dash="dot", line_color="#10B981", 
                                          annotation_text="Harga Entry", annotation_position="bottom right")
                            fig.add_hline(y=rekomendasi_sl, line_dash="solid", line_color="#EF4444", 
                                          annotation_text="Stop Loss", annotation_position="bottom right")

                        fig.update_layout(
                            template='plotly_dark',
                            hovermode='x unified',
                            margin=dict(l=20, r=20, t=30, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        # Render chart yang responsif
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10), use_container_width=True)
                        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses: {e}")
