import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- FUNGSI FORMAT ANGKA (TITIK) ---
def format_rp(angka):
    if pd.isna(angka): return "-"
    return f"{angka:,.0f}".replace(",", ".")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="The Commander Radar & Porto", layout="wide")

# --- NAVIGASI SIDEBAR ---
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio("Pilih Mode:", ["Radar Analisis", "Portofolio Live"])

# ==========================================
# MODE 1: RADAR ANALISIS (KODE LAMA ANDA)
# ==========================================
if menu == "Radar Analisis":
    st.title("Radar Volatilitas & Prediksi Harga")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Parameter Radar")
    ticker_input = st.sidebar.text_input("Kode Saham", value="", placeholder="Contoh: MAPI")
    harga_entry_str = st.sidebar.text_input("Harga Entry (Opsional)", value="", placeholder="Masukkan harga beli")
    pengali_atr = st.sidebar.slider("Pengali ATR", 1.0, 3.0, 1.5, 0.5)

    if st.sidebar.button("Proses Analisis"):
        if not ticker_input:
            st.warning("Masukkan kode saham.")
        else:
            try:
                ticker_yf = ticker_input.strip().upper()
                if not ticker_yf.endswith(".JK"): ticker_yf += ".JK"
                
                data = yf.download(ticker_yf, period="3mo", interval="1d", progress=False)
                
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    # Perhitungan ATR & Regresi (Logika sama seperti sebelumnya)
                    df = data.copy()
                    df['TR'] = pd.concat([df['High']-df['Low'], 
                                          abs(df['High']-df['Close'].shift(1)), 
                                          abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
                    df['ATR'] = df['TR'].rolling(window=14).mean()
                    atr_terkini = df['ATR'].iloc[-1]
                    jarak_toleransi = atr_terkini * pengali_atr
                    
                    try:
                        harga_beli = float(harga_entry_str) if harga_entry_str.strip() != "" else 0.0
                    except:
                        harga_beli = 0.0
                    rekomendasi_sl = harga_beli - jarak_toleransi if harga_beli > 0 else 0

                    # Linear Regression
                    df_pred = data.reset_index()
                    df_pred['Date_Ordinal'] = df_pred['Date'].map(datetime.toordinal)
                    X, y = df_pred[['Date_Ordinal']].values, df_pred['Close'].values
                    model = LinearRegression().fit(X, y)
                    tgl_pred = [df_pred['Date'].max() + timedelta(days=i) for i in range(1, 8)]
                    predictions = model.predict(np.array([d.toordinal() for d in tgl_pred]).reshape(-1, 1))

                    # Display Metrik
                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    c1.metric("ATR Terkini", f"Rp {format_rp(atr_terkini)}")
                    if harga_beli > 0:
                        c1.error(f"Stop Loss: Rp {format_rp(rekomendasi_sl)}")
                    
                    pred_akhir = predictions[-1]
                    perubahan = ((pred_akhir - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
                    c2.metric("Proyeksi H+7", f"Rp {format_rp(pred_akhir)}", f"{perubahan:.2f}%")

                    # Grafik Plotly
                    fig = go.Figure()
                    hist_data = df_pred.tail(30)
                    fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['Close'], name='Close', line=dict(color='white')))
                    fig.add_trace(go.Scatter(x=tgl_pred, y=predictions, name='Prediksi', line=dict(color='#F59E0B', dash='dash')))
                    
                    if harga_beli > 0:
                        fig.add_trace(go.Scatter(x=[hist_data['Date'].min(), max(tgl_pred)], y=[harga_beli, harga_beli], name='Entry', line=dict(color='#10B981', dash='dot')))
                        fig.add_trace(go.Scatter(x=[hist_data['Date'].min(), max(tgl_pred)], y=[rekomendasi_sl, rekomendasi_sl], name='SL', line=dict(color='#EF4444')))

                    fig.update_layout(template='plotly_dark', hovermode='x', dragmode=False, height=450)
                    fig.update_xaxes(fixedrange=True); fig.update_yaxes(fixedrange=True)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    # Tabel Mentah
                    df_raw = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
                    df_raw['Volume'] = df_raw['Volume'] / 100
                    df_raw.index = df_raw.index.strftime('%d-%m-%Y')
                    for col in df_raw.columns:
                        df_raw[col] = df_raw[col].apply(format_rp)
                    st.dataframe(df_raw, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# MODE 2: PORTOFOLIO LIVE (FITUR BARU)
# ==========================================
elif menu == "Portofolio Live":
    st.title("Brankas Portofolio")
    
    try:
        df_porto = pd.read_csv('portofolio_aktif.csv')
        total_modal, total_nilai = 0, 0
        live_list = []

        with st.spinner("Memperbarui harga pasar..."):
            for _, row in df_porto.iterrows():
                kode = str(row['Kode']).strip().upper()
                kode_yf = kode if kode.endswith('.JK') else f"{kode}.JK"
                
                # Ambil data live
                t_data = yf.download(kode_yf, period="1d", progress=False)
                last_p = t_data['Close'].iloc[-1] if not t_data.empty else row['Harga_Average']
                
                modal = (row['Lot'] * 100) * row['Harga_Average']
                nilai = (row['Lot'] * 100) * last_p
                pnl_rp = nilai - modal
                pnl_pct = (pnl_rp / modal * 100) if modal > 0 else 0
                
                total_modal += modal
                total_nilai += nilai
                live_list.append({"Kode": kode.replace(".JK",""), "Lot": row['Lot'], "Avg": row['Harga_Average'], 
                                  "Last": last_p, "Nilai": nilai, "PnL_Rp": pnl_rp, "PnL_Pct": pnl_pct})

        # Ringkasan Atas
        st.markdown("---")
        total_pnl = total_nilai - total_modal
        total_pnl_pct = (total_pnl / total_modal * 100) if total_modal > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Nilai", format_rp(total_nilai))
        c2.metric("Total Modal", format_rp(total_modal))
        c3.metric("Total Return", format_rp(total_pnl), f"{total_pnl_pct:.2f}%")

        st.markdown("### Detail Aset")
        for d in live_list:
            with st.container():
                col_a, col_b, col_c = st.columns([1, 1, 1])
                warna = "#10B981" if d['PnL_Rp'] > 0 else "#EF4444"
                
                col_a.markdown(f"**{d['Kode']}** \n{d['Lot']:g} Lot")
                col_b.markdown(f"Avg: {format_rp(d['Avg'])}  \nLast: {format_rp(d['Last'])}")
                col_c.markdown(f"<span style='color:{warna}; font-weight:bold;'>{format_rp(d['PnL_Rp'])} ({d['PnL_Pct']:.2f}%)</span>  \nVal: {format_rp(d['Nilai'])}", unsafe_allow_html=True)
                st.markdown("---")

    except Exception as e:
        st.info("Pastikan file 'portofolio_aktif.csv' sudah diunggah dan formatnya benar.")
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
