import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os
import json
import warnings
import geopandas as gpd
from openai import OpenAI

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Prediksi Pangan Multi-Komoditas", layout="wide", page_icon="🌾")
st.title("🌾 Sistem Prediksi & Analisis Spasio-Temporal Pangan Nasional")

# --- 1A. LOAD GEOJSON ---
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
    try:
        gdf = gpd.read_file(url)
    except Exception as e:
        st.error(f"Gagal memuat peta GeoJSON: {e}")
        st.stop()

    def bersihkan_nama(nama):
        if not nama: return "UNKNOWN"
        nama = str(nama).upper().strip()
        kamus = {
            'JAKARTA RAYA': 'DKI JAKARTA', 'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            'BANGKA BELITUNG': 'KEP. BANGKA BELITUNG', 'KEPULAUAN BANGKA BELITUNG': 'KEP. BANGKA BELITUNG',
            'KEPULAUAN RIAU': 'KEP. RIAU', 'SUMATRA UTARA': 'SUMATERA UTARA',
            'SUMATRA BARAT': 'SUMATERA BARAT', 'SUMATRA SELATAN': 'SUMATERA SELATAN',
            'DI. ACEH': 'ACEH', 'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
            'NUSATENGGARA TIMUR': 'NUSA TENGGARA TIMUR'
        }
        return kamus.get(nama, nama)

    gdf['Propinsi'] = gdf['Propinsi'].apply(bersihkan_nama)
    gdf['geometry'] = gdf['geometry'].make_valid()
    gdf = gdf.dissolve(by='Propinsi').reset_index()
    return json.loads(gdf.to_json())

# --- 1B. ADVANCED DATA LOADER FOR PROD/KONS FILES ---
@st.cache_data
def load_and_preprocess_data():
    base_dir = 'sample_data'
    file_hujan = os.path.join(base_dir, 'Curah_Hujan_Bulanan.csv')
    file_hbkn = os.path.join(base_dir, 'HBKN_Bulanan.csv')

    if not os.path.exists(file_hujan) or not os.path.exists(file_hbkn):
        st.error("Kritis: Berkas Curah Hujan atau HBKN tidak ditemukan di folder 'sample_data/'.")
        st.stop()

    df_hujan = pd.read_csv(file_hujan)
    df_hbkn = pd.read_csv(file_hbkn)
    df_hujan.columns = df_hujan.columns.str.strip()
    df_hbkn.columns = df_hbkn.columns.str.lower().str.strip()

    list_df_harga = []
    file_pengecualian = ['Curah_Hujan_Bulanan.csv', 'HBKN_Bulanan.csv', 'Harga_Produsen_Bulanan.csv', 'Konsumen_Bulanan.csv', 'baca.txt']
    
    if os.path.exists(base_dir):
        all_files = os.listdir(base_dir)
        for f in all_files:
            if f.endswith('.csv') and f not in file_pengecualian:
                file_path = os.path.join(base_dir, f)
                try:
                    df_temp = pd.read_csv(file_path)
                    df_temp.columns = df_temp.columns.str.lower().str.strip()
                    
                    # Parsing nama file (Contoh: "Cabai Rawit Merah_Konsumen_Harian.csv")
                    nama_bersih = f.replace('.csv', '').replace('_Harian', '')
                    
                    if '_Konsumen' in nama_bersih:
                        komoditas = nama_bersih.replace('_Konsumen', '').replace('_', ' ').strip()
                        tipe_pasar = 'Konsumen'
                    elif '_Produsen' in nama_bersih:
                        komoditas = nama_bersih.replace('_Produsen', '').replace('_', ' ').strip()
                        tipe_pasar = 'Produsen'
                    else:
                        komoditas = nama_bersih.replace('_', ' ').strip()
                        tipe_pasar = 'Konsumen'
                    
                    df_temp['komoditas'] = komoditas
                    df_temp['tipe_pasar'] = tipe_pasar
                    list_df_harga.append(df_temp)
                except Exception as e:
                    st.warning(f"Gagal memproses berkas {f}: {e}")

    if not list_df_harga:
        st.error("Kritis: Tidak ada file data harga komoditas harian yang valid di folder 'sample_data/'.")
        st.stop()

    df_raw_harga = pd.concat(list_df_harga, ignore_index=True)
    df_raw_harga['harga'] = df_raw_harga['harga'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df_raw_harga['harga'] = pd.to_numeric(df_raw_harga['harga'], errors='coerce')
    df_raw_harga = df_raw_harga.dropna(subset=['harga', 'tanggal', 'provinsi'])
    df_raw_harga['tanggal'] = pd.to_datetime(df_raw_harga['tanggal'], errors='coerce')
    df_raw_harga = df_raw_harga.dropna(subset=['tanggal'])

    koreksi_papua = {'PAPUA BARAT DAYA': 'PAPUA BARAT', 'PAPUA SELATAN': 'PAPUA', 'PAPUA TENGAH': 'PAPUA', 'PAPUA PEGUNUNGAN': 'PAPUA'}
    df_raw_harga['provinsi'] = df_raw_harga['provinsi'].astype(str).str.upper().str.strip().replace(koreksi_papua)

    # Agregasi Mingguan menyertakan tipe_pasar
    df_weekly = df_raw_harga.groupby(['komoditas', 'tipe_pasar', 'provinsi', pd.Grouper(key='tanggal', freq='W-MON')]).agg({'harga': 'mean'}).reset_index()

    # Gabung HBKN
    df_hbkn['tanggal'] = pd.to_datetime(df_hbkn['tanggal'], errors='coerce')
    df_hbkn_weekly = df_hbkn.groupby(pd.Grouper(key='tanggal', freq='W-MON')).agg({
        'hbkn': 'max',
        'keterangan': lambda x: 'Normal' if all(x.astype(str).str.lower() == 'tidak ada') else x[x.astype(str).str.lower() != 'tidak ada'].iloc[0]
    }).reset_index()

    df_combined = pd.merge(df_weekly, df_hbkn_weekly, on='tanggal', how='left')
    df_combined['hbkn'] = df_combined['hbkn'].fillna(0)
    df_combined['keterangan'] = df_combined['keterangan'].fillna('Normal')

    # Gabung Cuaca
    bulan_map = {'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12}
    df_hujan['Bulan'] = df_hujan['Bulan'].map(bulan_map).fillna(1)
    df_hujan['Nama Provinsi'] = df_hujan['Nama Provinsi'].astype(str).str.upper().str.strip().replace(koreksi_papua)
    df_hujan['Curah Hujan'] = pd.to_numeric(df_hujan['Curah Hujan'].astype(str).str.replace(',', '.', regex=False), errors='coerce').fillna(0)
    df_hujan['Tahun'] = pd.to_numeric(df_hujan['Tahun'], errors='coerce').fillna(2024).astype(int)
    df_hujan['tanggal'] = pd.to_datetime(df_hujan[['Tahun', 'Bulan']].assign(day=15).rename(columns={'Tahun':'year', 'Bulan':'month'}), errors='coerce')
    
    df_hujan_proxy = df_hujan[['Nama Provinsi', 'tanggal', 'Curah Hujan']].rename(columns={'Nama Provinsi': 'provinsi', 'Curah Hujan': 'Curah_Hujan'})

    df_final = pd.merge(df_combined, df_hujan_proxy, on=['provinsi', 'tanggal'], how='left')
    df_final['Curah_Hujan'] = df_final.groupby(['komoditas', 'tipe_pasar', 'provinsi'])['Curah_Hujan'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())

    df_final = df_final.rename(columns={'harga': 'Harga_Riil', 'keterangan': 'Momen', 'provinsi': 'Provinsi', 'tanggal': 'Tanggal', 'komoditas': 'Komoditas', 'tipe_pasar': 'Tipe_Pasar'})
    df_final['Tanggal_Str'] = df_final['Tanggal'].dt.strftime('%Y-%m-%d')
    df_final['Rata_Nasional'] = df_final.groupby(['Komoditas', 'Tipe_Pasar', 'Tanggal'])['Harga_Riil'].transform('mean')
    
    return df_final

with st.spinner("Menyelaraskan data produsen & konsumen nasional..."):
    df_all = load_and_preprocess_data()
    geojson_indo = load_geojson()

# --- 2. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🎯 Parameter Analisis")
    
    # Dropdown 1: Komoditas Pokok
    list_komoditas = sorted(df_all['Komoditas'].unique())
    komoditas_terpilih = st.selectbox("Komoditas:", list_komoditas)
    
    # Dropdown 2: Tingkat Rantai Pasok (Produsen vs Konsumen)
    list_pasar = sorted(df_all[df_all['Komoditas'] == komoditas_terpilih]['Tipe_Pasar'].unique())
    pasar_terpilih = st.selectbox("Tingkat Rantai Pasok:", list_pasar)
    
    # Filter Wilayah Berdasarkan Pilihan Atas
    df_sub = df_all[(df_all['Komoditas'] == komoditas_terpilih) & (df_all['Tipe_Pasar'] == pasar_terpilih)]
    provinsi_list = sorted(df_sub['Provinsi'].unique())
    default_prov = 'DKI JAKARTA' if 'DKI JAKARTA' in provinsi_list else provinsi_list[0]
    prov_terpilih = st.selectbox("Provinsi Fokus:", provinsi_list, index=provinsi_list.index(default_prov))
    
    api_key = st.text_input("OpenAI API Key (Opsional):", type="password")

    st.markdown("---")
    st.header("🗺️ Dimensi Spasial")
    list_minggu = sorted(df_sub['Tanggal_Str'].unique(), reverse=True)
    minggu_peta = st.selectbox("Minggu Peta:", list_minggu)

    df_filtered = df_sub[df_sub['Provinsi'] == prov_terpilih]

# --- 3. ML FORECASTING (RANDOM FOREST) ---
@st.cache_resource(show_spinner="Membuat Pemodelan AI...")
def train_model(df_prov):
    df_prov = df_prov.sort_values('Tanggal').reset_index(drop=True)
    if len(df_prov) < 8: return None, None, None, df_prov, [0]*8, []
    
    df_ml = pd.get_dummies(df_prov[['Curah_Hujan', 'Momen']], drop_first=True)
    feat_names = df_ml.columns.tolist()
    X, y = df_ml.values, df_prov['Harga_Riil'].values
    split_idx = int(len(X) * 0.8)
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X[:split_idx], y[:split_idx])
    rf_pred = rf.predict(X[split_idx:])
    
    future_steps = 8
    future_dates = [df_prov['Tanggal'].iloc[-1] + pd.Timedelta(days=7*i) for i in range(1, future_steps+1)]
    future_X = np.tile(X[-1], (future_steps, 1))
    rf_future = rf.predict(future_X).tolist()
    
    return rf, rf_pred, split_idx, df_prov, rf_future, future_dates

rf_model, rf_pred, split_idx, df_prov, rf_future, future_dates = train_model(df_filtered)

# --- 4. GRAPHICAL DASHBOARD INTERFACE ---
col1, col2 = st.columns([7, 3])

with col1:
    st.subheader(f"📊 Tren Harga {komoditas_terpilih} di Tingkat {pasar_terpilih}")
    
    tab1, tab2 = st.tabs(["🗺️ Peta Spasial Indonesia", "🎉 Koreksi Fluktuasi Mingguan HBKN"])
    
    with tab1:
        df_map = df_sub[df_sub['Tanggal_Str'] == minggu_peta]
        if df_map.empty:
            st.info("Data spasial tidak tersedia untuk minggu ini.")
        else:
            fig_map = px.choropleth(df_map, geojson=geojson_indo, locations="Provinsi", featureidkey="properties.Propinsi", 
                        color="Harga_Riil", hover_name="Provinsi", color_continuous_scale="Reds",
                        title=f"Distribusi Harga {komoditas_terpilih} ({pasar_terpilih}) - {minggu_peta}")
            fig_map.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.markdown("##### 📈 Hubungan Siklus HBKN Terhadap Lonjakan Harga Mingguan")
        fig_hbkn_trend = go.Figure()
        fig_hbkn_trend.add_trace(go.Scatter(x=df_prov['Tanggal'], y=df_prov['Harga_Riil'], mode='lines+markers', name='Harga Aktual (Rp)', line=dict(color='darkred', width=2)))
        fig_hbkn_trend.add_trace(go.Bar(x=df_prov['Tanggal'], y=df_prov['hbkn'] * df_prov['Harga_Riil'].max() * 0.12, name='Indikator HBKN', marker_color='orange', opacity=0.4))
        fig_hbkn_trend.update_layout(xaxis_title="Periode Tanggal", yaxis_title="Harga Tingkat Pasar (Rp)", barmode='overlay')
        st.plotly_chart(fig_hbkn_trend, use_container_width=True)

    st.markdown(f"##### 🔮 Prediksi AI Proyeksi 8 Minggu ke Depan")
    if rf_model is not None and len(rf_pred) > 0:
        test_dates = df_prov['Tanggal'].iloc[-len(rf_pred):]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df_prov['Tanggal'], y=df_prov['Harga_Riil'], mode='lines', name='Harga Sebenarnya', line=dict(color='blue')))
        fig_line.add_trace(go.Scatter(x=test_dates, y=rf_pred, mode='lines', name='Validasi Historis AI', line=dict(color='green', dash='dot')))
        fig_line.add_trace(go.Scatter(x=[df_prov['Tanggal'].iloc[-1]] + future_dates, y=[df_prov['Harga_Riil'].iloc[-1]] + rf_future,
                                      mode='lines+markers', name='Forecasting AI 2 Bulan', line=dict(color='purple', width=3, dash='dash')))
        fig_line.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Data wilayah terpilih belum mencukupi batas minimum permodelan AI jangka panjang.")

with col2:
    st.subheader("🤖 Analisis Rantai Pasok AI")
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    prompt = st.chat_input("Diskusikan intervensi harga pasokan...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if not api_key:
                st.warning("⚠️ Masukkan OpenAI API Key di Sidebar.")
            else:
                client = OpenAI(api_key=api_key)
                system_prompt = f"Anda adalah pakar pemodelan ketahanan pangan komoditas {komoditas_terpilih} di pasar {pasar_terpilih} wilayah {prov_terpilih}."
                messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_api, stream=True)
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
