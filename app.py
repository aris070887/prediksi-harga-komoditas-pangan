import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import os
import json
import warnings
import geopandas as gpd
from openai import OpenAI

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Analisis Pangan Komprehensif", layout="wide", page_icon="🌾")
st.title("🌾 Sistem Analisis Spasio-Temporal & Prediksi Pangan Nasional")

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

# --- 1B. LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_preprocess_data():
    base_dir = 'sample_data'
    file_hujan = os.path.join(base_dir, 'Curah_Hujan_Bulanan.csv')
    file_hbkn = os.path.join(base_dir, 'HBKN_Bulanan.csv')

    if not os.path.exists(file_hujan) or not os.path.exists(file_hbkn):
        st.error("Kritis: Berkas Curah Hujan atau HBKN tidak ditemukan.")
        st.stop()

    df_hujan = pd.read_csv(file_hujan)
    df_hbkn = pd.read_csv(file_hbkn)
    df_hujan.columns = df_hujan.columns.str.strip()
    df_hbkn.columns = df_hbkn.columns.str.lower().str.strip()

    list_df_harga = []
    file_pengecualian = ['Curah_Hujan_Bulanan.csv', 'HBKN_Bulanan.csv', 'Harga_Produsen_Bulanan.csv', 'Konsumen_Bulanan.csv', 'baca.txt']
    
    if os.path.exists(base_dir):
        for f in os.listdir(base_dir):
            if f.endswith('.csv') and f not in file_pengecualian:
                file_path = os.path.join(base_dir, f)
                try:
                    df_temp = pd.read_csv(file_path)
                    df_temp.columns = df_temp.columns.str.lower().str.strip()
                    
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

    df_raw_harga = pd.concat(list_df_harga, ignore_index=True)
    df_raw_harga['harga'] = df_raw_harga['harga'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df_raw_harga['harga'] = pd.to_numeric(df_raw_harga['harga'], errors='coerce')
    df_raw_harga = df_raw_harga.dropna(subset=['harga', 'tanggal', 'provinsi'])
    
    df_raw_harga['tanggal'] = pd.to_datetime(df_raw_harga['tanggal'], errors='coerce')
    df_raw_harga = df_raw_harga.dropna(subset=['tanggal'])

    koreksi_papua = {'PAPUA BARAT DAYA': 'PAPUA BARAT', 'PAPUA SELATAN': 'PAPUA', 'PAPUA TENGAH': 'PAPUA', 'PAPUA PEGUNUNGAN': 'PAPUA'}
    df_raw_harga['provinsi'] = df_raw_harga['provinsi'].astype(str).str.upper().str.strip().replace(koreksi_papua)

    df_weekly = df_raw_harga.groupby(['komoditas', 'tipe_pasar', 'provinsi', pd.Grouper(key='tanggal', freq='W-MON')]).agg({'harga': 'mean'}).reset_index()

    df_hbkn['tanggal'] = pd.to_datetime(df_hbkn['tanggal'], errors='coerce')
    df_hbkn_weekly = df_hbkn.groupby(pd.Grouper(key='tanggal', freq='W-MON')).agg({'hbkn': 'max', 'keterangan': 'first'}).reset_index()
    df_hbkn_weekly['keterangan'] = df_hbkn_weekly['keterangan'].fillna('Normal')

    df_combined = pd.merge(df_weekly, df_hbkn_weekly, on='tanggal', how='left')
    df_combined['hbkn'] = df_combined['hbkn'].fillna(0)

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
    df_final['Selisih_Nasional'] = df_final['Harga_Riil'] - df_final['Rata_Nasional']
    
    df_final = df_final.sort_values(['Komoditas', 'Tipe_Pasar', 'Provinsi', 'Tanggal'])
    df_final['Harga_Minggu_Lalu'] = df_final.groupby(['Komoditas', 'Tipe_Pasar', 'Provinsi'])['Harga_Riil'].shift(1)
    df_final['Selisih_Minggu_Lalu'] = (df_final['Harga_Riil'] - df_final['Harga_Minggu_Lalu']).fillna(0)

    df_final['Harga_Lag_2'] = df_final.groupby(['Komoditas', 'Tipe_Pasar', 'Provinsi'])['Harga_Riil'].shift(2).bfill()
    return df_final

with st.spinner("Memproses sinkronisasi data spasial & cuaca..."):
    df_all = load_and_preprocess_data()
    geojson_indo = load_geojson()

# --- 2. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🎯 Parameter")
    list_komoditas = sorted(df_all['Komoditas'].unique())
    komoditas_terpilih = st.selectbox("Komoditas:", list_komoditas)
    
    list_pasar = sorted(df_all[df_all['Komoditas'] == komoditas_terpilih]['Tipe_Pasar'].unique())
    pasar_terpilih = st.selectbox("Rantai Pasok:", list_pasar)
    
    df_sub = df_all[(df_all['Komoditas'] == komoditas_terpilih) & (df_all['Tipe_Pasar'] == pasar_terpilih)].copy()
    provinsi_list = sorted(df_sub['Provinsi'].unique())
    prov_terpilih = st.selectbox("Provinsi Fokus:", provinsi_list, index=0 if 'DKI JAKARTA' not in provinsi_list else provinsi_list.index('DKI JAKARTA'))
    
    st.markdown("---")
    st.header("📋 Pengaturan HET Wilayah")
    uploaded_het = st.file_uploader("Upload File HET Spasial (CSV):", type=["csv"])
    
    # Membaca basis data HET multi-wilayah
    df_het_mapped = pd.DataFrame()
    use_uploaded_het = False
    
    if uploaded_het is not None:
        try:
            df_het_uploaded = pd.read_csv(uploaded_het)
            df_het_uploaded.columns = df_het_uploaded.columns.str.lower().str.strip()
            if 'komoditas' in df_het_uploaded.columns and 'provinsi' in df_het_uploaded.columns and 'het' in df_het_uploaded.columns:
                df_het_uploaded['komoditas'] = df_het_uploaded['komoditas'].str.strip()
                df_het_uploaded['provinsi'] = df_het_uploaded['provinsi'].str.upper().str.strip()
                df_het_mapped = df_het_uploaded
                use_uploaded_het = True
                st.sidebar.success("✅ HET Spasial Berhasil Diterapkan!")
            else:
                st.sidebar.error("❌ Format kolom salah. Harus ada: 'komoditas', 'provinsi', 'het'.")
        except Exception as e:
            st.sidebar.error(f"Gagal membaca berkas HET: {e}")

    st.markdown("---")
    st.header("🗺️ Opsi Peta")
    metrik_peta = st.radio("Metrik Visualisasi Peta:", 
                           ["Harga Riil", "Selisih vs Rata-Rata Nasional", "Selisih vs Minggu Lalu", "Selisih vs HET Wilayah"])
    list_minggu = sorted(df_sub['Tanggal_Str'].unique(), reverse=True)
    minggu_peta = st.selectbox("Periode Minggu Peta:", list_minggu)

    df_filtered = df_sub[df_sub['Provinsi'] == prov_terpilih]

# --- 3. DYNAMIC CALCULATION FOR SPATIAL HET ---
def kalkulasi_selisih_het(row, minggu_ref):
    if use_uploaded_het:
        # Cari HET berdasarkan Komoditas DAN Provinsi spesifik
        match = df_het_mapped[(df_het_mapped['komoditas'] == row['Komoditas']) & (df_het_mapped['provinsi'] == row['Provinsi'])]
        if not match.empty:
            return row['Harga_Riil'] - match['het'].iloc[0]
            
    # Fallback jika tidak di-upload atau kombinasi wilayah tidak ditemukan: Pakai rata-rata nasional minggu tersebut
    return row['Harga_Riil'] - row['Rata_Nasional']

df_sub['Selisih_HET'] = df_sub.apply(lambda r: kalkulasi_selisih_het(r, minggu_peta), axis=1)

# --- 4. TRAIN ML MODEL ---
@st.cache_resource
def dapatkan_feature_importance(df_prov):
    if len(df_prov) < 5: return None
    fitur = ['Curah_Hujan', 'hbkn', 'Harga_Lag_2']
    X = df_prov[fitur].fillna(0)
    y = df_prov['Harga_Riil']
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    return pd.DataFrame({
        'Faktor Pengaruh': ['Curah Hujan (Weather)', 'Siklus Hari Raya (HBKN)', 'Tren Harga Historis (Lag)'],
        'Tingkat Pengaruh (%)': rf.feature_importances_ * 100
    }).sort_values(by='Tingkat Pengaruh (%)', ascending=True)

df_imp = dapatkan_feature_importance(df_filtered)

# --- 5. LAYOUT UTAMA ---
col1, col2 = st.columns([7, 3])

with col1:
    st.subheader(f"📊 Dashboard Analisis Dinamis: {komoditas_terpilih} ({prov_terpilih})")
    
    # Keterangan status HET dinamis untuk provinsi fokus
    if use_uploaded_het:
        fokus_match = df_het_mapped[(df_het_mapped['komoditas'] == komoditas_terpilih) & (df_het_mapped['provinsi'] == prov_terpilih)]
        if not fokus_match.empty:
            st.info(f"📋 **Status HET Wilayah ({prov_terpilih}):** Rp {fokus_match['het'].iloc[0]:,.0f} *(Berdasarkan Berkas CSV)*")
        else:
            st.warning(f"⚠️ **Status HET Wilayah ({prov_terpilih}):** Data tidak ada di CSV. Menggunakan Fallback Rata-rata Peta Nasional.")
    else:
        st.info("💡 **Status Acuan:** HET tidak di-upload, sistem otomatis menggunakan Rata-rata Nasional sebagai tolok ukur.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Peta Distribusi Nasional", 
        "🌧️ Korelasi Cuaca & Harga", 
        "🎉 Tren Siklus HBKN",
        "🎯 Feature Importance AI"
    ])
    
    with tab1:
        df_map = df_sub[df_sub['Tanggal_Str'] == minggu_peta]
        if df_map.empty:
            st.info("Data tidak tersedia untuk minggu terpilih.")
        else:
            if metrik_peta == "Harga Riil":
                kolom_peta = "Harga_Riil"
                skala_warna = "Reds"
                judul_peta = f"Harga Riil {komoditas_terpilih} (Rp)"
            elif metrik_peta == "Selisih vs Rata-Rata Nasional":
                kolom_peta = "Selisih_Nasional"
                skala_warna = "RdBu_r"
                judul_peta = f"Selisih Harga vs Rata-Rata Nasional (Rp)"
            elif metrik_peta == "Selisih vs Minggu Lalu":
                kolom_peta = "Selisih_Minggu_Lalu"
                skala_warna = "Cwtex"
                judul_peta = f"Perubahan Harga Dibanding Minggu Lalu (Rp)"
            else:
                kolom_peta = "Selisih_HET"
                skala_warna = "PRGn" # Ungu = di atas HET (mahal/waspada), Hijau = di bawah HET (aman)
                judul_peta = f"Selisih Harga vs Target HET Regional Spasial (Rp)"

            fig_map = px.choropleth(df_map, geojson=geojson_indo, locations="Provinsi", featureidkey="properties.Propinsi", 
                                    color=kolom_peta, hover_name="Provinsi", color_continuous_scale=skala_warna,
                                    title=f"{judul_peta} - Periode {minggu_peta}")
            fig_map.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.markdown("##### 🌧️ Grafik Korelasi Curah Hujan Mingguan vs Tren Harga")
        fig_korelasi = go.Figure()
        fig_korelasi.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Harga_Riil'], mode='lines+markers',
                                          name='Harga Riil (Sumbu Kiri)', line=dict(color='darkred', width=2.5)))
        fig_korelasi.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Curah_Hujan'], mode='lines',
                                          name='Curah Hujan (Sumbu Kanan)', line=dict(color='deepskyblue', width=1.5, dash='dash'),
                                          yaxis='y2'))
        # KODE YANG BENAR & VALID UNTUK PLOTLY DUAL-AXIS
        fig_korelasi.update_layout(
            title=f"Hubungan Curah Hujan vs Harga {komoditas_terpilih} di {prov_terpilih}",
            xaxis=dict(title="Periode Waktu"),
            yaxis=dict(
                title="Harga Tingkat Pasar (Rp)", 
                titlefont=dict(color='darkred'), 
                tickfont=dict(color='darkred')
            ),
            yaxis2=dict(
                title="Estimasi Curah Hujan (mm)", 
                titlefont=dict(color='deepskyblue'), 
                tickfont=dict(color='deepskyblue'),
                overlaying='y', 
                side='right'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_korelasi, use_container_width=True)

    with tab3:
        st.markdown("##### 📈 Dampak Siklus HBKN terhadap Fluktuasi")
        fig_hbkn = go.Figure()
        fig_hbkn.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Harga_Riil'], mode='lines+markers', name='Harga Aktual', line=dict(color='orange')))
        fig_hbkn.add_trace(go.Bar(x=df_filtered['Tanggal'], y=df_filtered['hbkn'] * df_filtered['Harga_Riil'].max() * 0.1, name='Indikator HBKN', marker_color='red', opacity=0.3))
        st.plotly_chart(fig_hbkn, use_container_width=True)

    with tab4:
        st.markdown("##### 🎯 Feature Importance (Faktor Utama Penentu Fluktuasi Harga)")
        if df_imp is not None:
            fig_imp = px.bar(df_imp, x='Tingkat Pengaruh (%)', y='Faktor Pengaruh', orientation='h',
                             title=f"Analisis Kontribusi Faktor Terhadap Harga {komoditas_terpilih} di {prov_terpilih}",
                             color='Tingkat Pengaruh (%)', color_continuous_scale='Viridis')
            fig_imp.update_layout(xaxis_title="Persentase Kontribusi Pengaruh (%)", yaxis_title="")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Data tidak mencukupi untuk melakukan ekstraksi Feature Importance AI.")

with col2:
    st.subheader("🤖 Konsultan Ketahanan Pangan AI")
    api_key = st.text_input("OpenAI API Key:", type="password")
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    prompt = st.chat_input("Tanya strategi pasokan pangan...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if not api_key:
                st.warning("⚠️ Masukkan OpenAI API Key di Sidebar.")
            else:
                client = OpenAI(api_key=api_key)
                system_prompt = f"Anda adalah analis ketahanan pangan komoditas {komoditas_terpilih} di pasar {pasar_terpilih} wilayah {prov_terpilih}."
                messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_api, stream=True)
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
