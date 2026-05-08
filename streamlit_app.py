import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Radar Analisis Saham", layout="wide")

st.title("Radar Volatilitas & Prediksi Harga")

# --- SIDEBAR INPUT ---
st.sidebar.header("Parameter Input")

# Input Ticker & Harga (Default Kosong)
ticker_input = st.sidebar.text_input("Kode Saham", value=None, placeholder="Contoh: BBCA")
harga_entry = st.sidebar.number_input("Harga Entry (Opsional)", value=None, placeholder="Masukkan harga beli")
pengali_atr = st.sidebar.slider("Pengali ATR (Toleransi)", 1.0, 3.0, 1.5, 0.5)

if st.sidebar.button("Proses Analisis"):
    if not ticker_input:
        st.warning("Silakan masukkan kode saham terlebih dahulu.")
    else:
        try:
            with st.spinner('Sinkronisasi data...'):
                # Penambahan .JK otomatis
                ticker_yf = ticker_input.strip().upper()
                if not ticker_yf.endswith(".JK"):
                    ticker_yf += ".JK"
                
                # Pengambilan Data
                data = yf.download(ticker_yf, period="3mo", interval="1d", progress=False)
                
                if data.empty:
                    st.error("Data tidak ditemukan. Pastikan kode saham benar.")
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    # --- PERHITUNGAN TEKNIS ---
                    df = data.copy()
                    df['H-L'] = df['High'] - df['Low']
                    df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
                    df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
                    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
                    df['ATR'] = df['TR'].rolling(window=14).mean()
                    
                    atr_terkini = df['ATR'].iloc[-1]
                    jarak_toleransi = atr_terkini * pengali_atr
                    
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
                    
                    # --- PANEL HASIL ---
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Analisis Volatilitas")
                        st.metric("ATR Terkini", f"Rp {atr_terkini:,.2f}".replace(",", "."))
                        st.metric(f"Toleransi ({pengali_atr}x)", f"Rp {jarak_toleransi:,.2f}".replace(",", "."))
                        if harga_beli > 0:
                            st.error(f"Stop Loss: Rp {rekomendasi_sl:,.0f}".replace(",", "."))
                    
                    with col2:
                        st.subheader("Prediksi Harga")
                        pred_akhir = predictions[-1]
                        harga_terakhir = df['Close'].iloc[-1]
                        perubahan_pred = ((pred_akhir - harga_terakhir) / harga_terakhir) * 100
                        st.metric("Proyeksi H+7", f"Rp {pred_akhir:,.2f}".replace(",", "."), f"{perubahan_pred:.2f}%")

                    # --- VISUALISASI & TABEL ---
                    tab1, tab2 = st.tabs(["📊 Grafik Interaktif", "📝 Data Historis Mentah"])
                    
                    with tab1:
                        fig = go.Figure()
                        
                        # Historis
                        hist_data = df_pred.tail(30)
                        fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['Close'], 
                                                 mode='lines', name='Harga Close', 
                                                 line=dict(color='white', width=2)))
                        
                        # Prediksi
                        fig.add_trace(go.Scatter(x=tgl_pred, y=predictions, 
                                                 mode='lines', name='Prediksi H+7', 
                                                 line=dict(color='#F59E0B', width=2, dash='dash')))
                        
                        # Area Toleransi
                        upper_band = predictions + jarak_toleransi
                        lower_band = predictions - jarak_toleransi
                        fig.add_trace(go.Scatter(x=tgl_pred, y=upper_band, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig.add_trace(go.Scatter(x=tgl_pred, y=lower_band, mode='lines', fill='tonexty', 
                                                 fillcolor='rgba(34, 211, 238, 0.1)', line=dict(width=0), name='Zona Volatilitas'))
                        
                        if harga_beli > 0:
                            fig.add_hline(y=harga_beli, line_dash="dot", line_color="#10B981")
                            fig.add_hline(y=rekomendasi_sl, line_dash="solid", line_color="#EF4444")

                        fig.update_layout(
                            template='plotly_dark',
                            hovermode='x', # Lebih stabil, tidak mengikuti setiap pergerakan mouse
                            dragmode='pan',
                            margin=dict(l=20, r=20, t=50, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    with tab2:
                        # Format Tabel: Tanggal bersih & Ribuan pakai titik
                        df_raw = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15).copy()
                        df_raw.index = df_raw.index.strftime('%d-%m-%Y') # Format tanggal Indonesia
                        
                        # Formatting angka ribuan dengan titik
                        st.dataframe(df_raw.style.format({
                            'Open': '{:,.2f}',
                            'High': '{:,.2f}',
                            'Low': '{:,.2f}',
                            'Close': '{:,.2f}',
                            'Volume': '{:,.0f}'
                        }).format(lambda x: str(x).replace(",", "X").replace(".", ",").replace("X", "."), subset=['Open', 'High', 'Low', 'Close', 'Volume']),
                        use_container_width=True)
                        
        except Exception as e:
            st.error(f"Gagal memproses analisis: {e}")
                    
