import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import datetime, timedelta

# --- FUNGSI FORMAT ANGKA ---
def format_rp(angka):
    if pd.isna(angka): 
        return "-"
    return f"{angka:,.0f}".replace(",", ".")

# --- SISTEM MEMORI (CACHE) ---
@st.cache_data(ttl=300)
def tarik_data_radar(ticker):
    tkr = yf.Ticker(ticker)
    hist = tkr.history(period="6mo", interval="1d") # Diperpanjang 6 bulan untuk akurasi Support/Resisten
    return hist

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="The Commander - Radar Analisis", layout="wide")

st.title("🎯 Radar Trisula: Volatilitas, Regresi Melengkung & Monte Carlo")

# --- PARAMETER SIDEBAR ---
st.sidebar.header("Parameter Radar")
ticker_input = st.sidebar.text_input("Kode Saham", value="", placeholder="Contoh: BBCA")
harga_entry_str = st.sidebar.text_input("Harga Entry (Opsional)", value="", placeholder="Masukkan harga beli")
pengali_atr = st.sidebar.slider("Pengali ATR (Toleransi)", 1.0, 3.0, 1.5, 0.5)

if st.sidebar.button("Proses Analisis"):
    if not ticker_input:
        st.warning("Silakan masukkan kode saham terlebih dahulu.")
    else:
        try:
            with st.spinner('Menjalankan Simulasi Ratusan Skenario...'):
                ticker_yf = ticker_input.strip().upper()
                if not ticker_yf.endswith(".JK"):
                    ticker_yf += ".JK"
                
                data = tarik_data_radar(ticker_yf)
                
                if data.empty:
                    st.error("Data tidak ditemukan. Pastikan kode saham benar atau jaringan stabil.")
                else:
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                        
                    data = data.dropna(subset=['Close', 'High', 'Low'])

                    if data.empty:
                        st.warning("Data saham kosong (kemungkinan sedang disuspensi).")
                    else:
                        df = data.copy()
                        df['H-L'] = df['High'] - df['Low']
                        df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
                        df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
                        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
                        df['ATR'] = df['TR'].rolling(window=14).mean()
                        
                        atr_terkini = df['ATR'].dropna().iloc[-1] if not df['ATR'].dropna().empty else 0
                        jarak_toleransi = atr_terkini * pengali_atr
                        harga_terakhir = df['Close'].iloc[-1]
                        
                        try:
                            harga_beli = float(harga_entry_str) if harga_entry_str.strip() != "" else 0.0
                        except ValueError:
                            harga_beli = 0.0
                            
                        # UPGRADE 3: AUTO SUPPORT & RESISTANCE (Deteksi Puncak & Lembah 1 Bulan Terakhir)
                        res_kuat = df['High'].tail(20).max()
                        sup_kuat = df['Low'].tail(20).min()
                        
                        # Data Persiapan ML
                        df_pred = data.reset_index()
                        if 'Date' not in df_pred.columns:
                            df_pred = df_pred.rename(columns={'index': 'Date', 'Datetime': 'Date'})
                            
                        df_pred['Date_Ordinal'] = df_pred['Date'].map(datetime.toordinal)
                        
                        # Membatasi data ML hanya 45 hari terakhir agar tren lebih responsif (tidak kaku)
                        df_ml = df_pred.tail(45) 
                        X = df_ml[['Date_Ordinal']].values
                        y = df_ml['Close'].values
                        
                        # UPGRADE 1: REGRESI POLINOMIAL (DERAJAT 2 untuk prediksi melengkung)
                        model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
                        model_poly.fit(X, y)
                        
                        last_date = df_pred['Date'].max()
                        tgl_pred = [last_date + timedelta(days=i) for i in range(1, 8)]
                        X_pred = np.array([d.toordinal() for d in tgl_pred]).reshape(-1, 1)
                        predictions = model_poly.predict(X_pred)
                        
                        # UPGRADE 2: SIMULASI MONTE CARLO (1000 Skenario, 7 Hari)
                        returns = df['Close'].pct_change().dropna().tail(60) # Ambil volatilitas 60 hari terakhir
                        mu = returns.mean()
                        sigma = returns.std()
                        
                        simulasi_hari = 7
                        jumlah_simulasi = 1000
                        jalur_simulasi = np.zeros((simulasi_hari, jumlah_simulasi))
                        jalur_simulasi[0] = harga_terakhir
                        
                        for t in range(1, simulasi_hari):
                            rand_rets = np.random.normal(mu, sigma, jumlah_simulasi)
                            jalur_simulasi[t] = jalur_simulasi[t-1] * (1 + rand_rets)
                        
                        # Menghitung Pita Kepercayaan Monte Carlo (5% dan 95%)
                        mc_atas = np.percentile(jalur_simulasi, 95, axis=1)
                        mc_bawah = np.percentile(jalur_simulasi, 5, axis=1)
                        harga_akhir_simulasi = jalur_simulasi[-1]
                        
                        # Kalkulasi Probabilitas Taktis
                        if harga_beli > 0:
                            tp_target = harga_beli + jarak_toleransi
                            sl_target = harga_beli - jarak_toleransi
                            prob_hit_tp = (harga_akhir_simulasi >= tp_target).sum() / jumlah_simulasi * 100
                            prob_hit_sl = (harga_akhir_simulasi <= sl_target).sum() / jumlah_simulasi * 100
                        else:
                            prob_naik = (harga_akhir_simulasi > harga_terakhir).sum() / jumlah_simulasi * 100
                        
                        # TAMPILAN DASHBOARD
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("🛡️ Pertahanan (S&R)")
                            st.metric("Resisten Kuat (20D)", f"Rp {format_rp(res_kuat)}")
                            st.metric("Support Kuat (20D)", f"Rp {format_rp(sup_kuat)}")
                            st.metric("ATR Terkini", f"Rp {format_rp(atr_terkini)}")
                        
                        with col2:
                            st.subheader("📈 Prediksi Regresi")
                            pred_akhir = predictions[-1]
                            perubahan_pred = ((pred_akhir - harga_terakhir) / harga_terakhir) * 100
                            st.metric("Proyeksi H+7 (Polinomial)", f"Rp {format_rp(pred_akhir)}", f"{perubahan_pred:.2f}%")
                            st.metric(f"Toleransi Napas ({pengali_atr}x)", f"Rp {format_rp(jarak_toleransi)}")
                            
                        with col3:
                            st.subheader("🎲 Monte Carlo (1000x)")
                            if harga_beli > 0:
                                st.metric("Peluang Sentuh Target Profit", f"{prob_hit_tp:.1f}%")
                                st.metric("Risiko Sentuh Stop Loss", f"{prob_hit_sl:.1f}%")
                            else:
                                st.metric("Probabilitas Harga Naik", f"{prob_naik:.1f}%")
                                st.metric("Puncak Optimis H+7 (MC 95%)", f"Rp {format_rp(mc_atas[-1])}")

                        # VISUALISASI PETA TACTICAL
                        tab1, tab2 = st.tabs(["📊 Peta Pertempuran Trisula", "📝 Log Historis"])
                        with tab1:
                            fig = go.Figure()
                            hist_data = df_pred.tail(45)
                            tanggal_min = hist_data['Date'].min()
                            tanggal_max = pd.Series(tgl_pred).max()
                            
                            # Harga Asli
                            fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['Close'], mode='lines', name='Harga Aktual', line=dict(color='white', width=2)))
                            
                            # Auto Support & Resistance
                            fig.add_trace(go.Scatter(x=[tanggal_min, tanggal_max], y=[res_kuat, res_kuat], mode='lines', name='Resisten', line=dict(color='#EF4444', dash='dot', width=1)))
                            fig.add_trace(go.Scatter(x=[tanggal_min, tanggal_max], y=[sup_kuat, sup_kuat], mode='lines', name='Support', line=dict(color='#10B981', dash='dot', width=1)))
                            
                            # Regresi Polinomial Melengkung
                            fig.add_trace(go.Scatter(x=tgl_pred, y=predictions, mode='lines', name='Tren Polinomial', line=dict(color='#F59E0B', width=3)))
                            
                            # Zona Monte Carlo (Pita Probabilitas)
                            fig.add_trace(go.Scatter(x=tgl_pred, y=mc_atas, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                            fig.add_trace(go.Scatter(x=tgl_pred, y=mc_bawah, mode='lines', fill='tonexty', fillcolor='rgba(59, 130, 246, 0.15)', line=dict(width=0), name='Zona Probabilitas 90% (MC)'))
                            
                            if harga_beli > 0:
                                fig.add_trace(go.Scatter(x=[tanggal_min, tanggal_max], y=[harga_beli, harga_beli], mode='lines', name='Garis Entry', line=dict(color='cyan', dash='dash', width=2)))
                                sl_line = harga_beli - jarak_toleransi
                                fig.add_trace(go.Scatter(x=[tanggal_min, tanggal_max], y=[sl_line, sl_line], mode='lines', name='Batas Cut Loss', line=dict(color='red', width=2)))

                            fig.update_layout(template='plotly_dark', hovermode='x', dragmode=False, height=500, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            fig.update_xaxes(fixedrange=True)
                            fig.update_yaxes(fixedrange=True)
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                        with tab2:
                            df_raw = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15).copy()
                            df_raw.index = df_raw.index.strftime('%d-%m-%Y')
                            df_raw['Volume'] = df_raw['Volume'] / 100
                            df_raw = df_raw.rename(columns={'Open':'Open (Rp)', 'High':'High (Rp)', 'Low':'Low (Rp)', 'Close':'Close (Rp)', 'Volume':'Volume (Lot)'})
                            for col in df_raw.columns:
                                    df_raw[col] = df_raw[col].apply(format_rp)
                            st.dataframe(df_raw, use_container_width=True)

        except Exception as e:
            st.error(f"Sistem Gagal Memproses Operasi Trisula: {e}")
