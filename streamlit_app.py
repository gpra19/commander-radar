import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- FUNGSI FORMAT ANGKA RIBUAN (TITIK) ---
def format_rp(angka):
    if pd.isna(angka):
        return "-"
    # Format menjadi tanpa desimal, lalu ganti koma menjadi titik
    return f"{angka:,.0f}".replace(",", ".")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Radar Analisis Saham", layout="wide")

st.title("Radar Volatilitas & Prediksi Harga")

# --- SIDEBAR INPUT ---
st.sidebar.header("Parameter Input")

# Menggunakan text_input agar baris benar-benar kosong saat awal dimuat
ticker_input = st.sidebar.text_input("Kode Saham", value="", placeholder="Contoh: BBCA")
harga_entry_str = st.sidebar.text_input("Harga Entry (Opsional)", value="", placeholder="Masukkan harga beli")
pengali_atr = st.sidebar.slider("Pengali ATR (Toleransi)", 1.0, 3.0, 1.5, 0.5)

if st.sidebar.button("Proses Analisis"):
    if not ticker_input:
        st.warning("Silakan masukkan kode saham terlebih dahulu.")
    else:
        try:
            with st.spinner('Sinkronisasi data...'):
                ticker_yf = ticker_input.strip().upper()
                if not ticker_yf.endswith(".JK"):
                    ticker_yf += ".JK"
                
                data = yf.download(ticker_yf, period="3mo", interval="1d", progress=False)
                
                if data.empty:
                    st.error("Data tidak ditemukan. Pastikan kode saham benar.")
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    df = data.copy()
                    df['H-L'] = df['High'] - df['Low']
                    df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
                    df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
                    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
                    df['ATR'] = df['TR'].rolling(window=14).mean()
                    
                    atr_terkini = df['ATR'].iloc[-1]
                    jarak_toleransi = atr_terkini * pengali_atr
                    
                    # Konversi string ke angka
                    try:
                        harga_beli = float(harga_entry_str) if harga_entry_str.strip() != "" else 0.0
                    except ValueError:
                        harga_beli = 0.0
                        
                    rekomendasi_sl = harga_beli - jarak_toleransi if harga_beli > 0 else 0
                    
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
                        st.metric("ATR Terkini", f"Rp {format_rp(atr_terkini)}")
                        st.metric(f"Toleransi ({pengali_atr}x)", f"Rp {format_rp(jarak_toleransi)}")
                        if harga_beli > 0:
                            st.error(f"Stop Loss: Rp {format_rp(rekomendasi_sl)}")
                    
                    with col2:
                        st.subheader("Prediksi Harga")
                        pred_akhir = predictions[-1]
                        harga_terakhir = df['Close'].iloc[-1]
                        perubahan_pred = ((pred_akhir - harga_terakhir) / harga_terakhir) * 100
                        st.metric("Proyeksi H+7", f"Rp {format_rp(pred_akhir)}", f"{perubahan_pred:.2f}%")

                    # --- VISUALISASI & TABEL ---
                    tab1, tab2 = st.tabs(["📊 Grafik", "📝 Data Historis Mentah"])
                    
                    with tab1:
                        fig = go.Figure()
                        
                        hist_data = df_pred.tail(30)
                        
                        # Gabungkan semua tanggal (historis + prediksi) untuk panjang garis lurus
                        tanggal_min = hist_data['Date'].min()
                        tanggal_max = pd.Series(tgl_pred).max()
                        
                        fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['Close'], 
                                                 mode='lines', name='Harga Close', 
                                                 line=dict(color='white', width=2)))
                        
                        fig.add_trace(go.Scatter(x=tgl_pred, y=predictions, 
                                                 mode='lines', name='Prediksi H+7', 
                                                 line=dict(color='#F59E0B', width=2, dash='dash')))
                        
                        upper_band = predictions + jarak_toleransi
                        lower_band = predictions - jarak_toleransi
                        fig.add_trace(go.Scatter(x=tgl_pred, y=upper_band, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig.add_trace(go.Scatter(x=tgl_pred, y=lower_band, mode='lines', fill='tonexty', 
                                                 fillcolor='rgba(34, 211, 238, 0.1)', line=dict(width=0), name='Zona Volatilitas'))
                        
                        if harga_beli > 0:
                            # Garis Harga Entry (Hijau) - Ditambahkan ke legenda
                            fig.add_trace(go.Scatter(x=[tanggal_min, tanggal_max], y=[harga_beli, harga_beli],
                                                     mode='lines', name='Harga Entry', 
                                                     line=dict(color='#10B981', dash='dot', width=2)))
                            
                            # Garis Stop Loss (Merah) - Ditambahkan ke legenda
                            fig.add_trace(go.Scatter(x=[tanggal_min, tanggal_max], y=[rekomendasi_sl, rekomendasi_sl],
                                                     mode='lines', name='Stop Loss', 
                                                     line=dict(color='#EF4444', width=2)))

                        fig.update_layout(
                            template='plotly_dark',
                            hovermode='x', 
                            dragmode=False, # MENGUNCI GRAFIK AGAR STATIS
                            margin=dict(l=20, r=20, t=50, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        # Kunci sumbu X dan Y agar tidak bisa di-zoom atau di-scroll
                        fig.update_xaxes(fixedrange=True)
                        fig.update_yaxes(fixedrange=True)
                        
                        # Hilangkan tombol-tombol melayang di pojok kanan atas
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    with tab2:
                        df_raw = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15).copy()
                        df_raw.index = df_raw.index.strftime('%d-%m-%Y')
                        
                        # Terapkan format pemisah ribuan ke seluruh kolom di dalam tabel
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            df_raw[col] = df_raw[col].apply(lambda x: format_rp(x))
                        
                        st.dataframe(df_raw, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Gagal memproses analisis: {e}")
